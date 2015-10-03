#pragma once
/*
    This implementation is based on the beta regression published on:
    Ferrari, Silvia, and Francisco Cribari-Neto. "Beta regression for modelling rates and proportions." Journal of Applied Statistics 31.7 (2004): 799-815.
*/

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>
#include <ctime>
#include <chrono>

#include <Eigen/Eigen>
#include <Eigen/SVD>

#define LBFGS_FLOAT 64
#include <lbfgs.h>

#include "omp.h"

namespace adrianyu{
    
    template<typename Scalar>
    void dumpArr(std::ostream &outss, const Scalar *arr, size_t n){
        for (size_t i = 0; i < n; ++i){
            outss << arr[i] << "\n";
        }
    }

    double logitfunc(const double x){
        return std::log(x / (1.0 - x));
    }
    double logitDevRep(const double x){
        return x * (1.0 - x);
    }
    double sigmoidfunc(double x){
        // avoid overflow error
        if (x > 100){
            x = 100;
        }
        else{
            if (x < -100){
                x = -100;
            }
        }
        return 1.0 / (1.0 + std::exp(-x));
    }

    // see https://en.wikipedia.org/wiki/Digamma_function
    // for details
    double digamma(double x){
        // when near zero, appr. -1/x
        // when x is too small, it is probably not needed.
        if (x < 1.0e-20){
            return -1.0e20;
        }
        double res = 0;
        while (x < 10){
            res -= 1.0 / x;
            x++;
        }
        // for large x, we don't need so many digits
        if (x < 100){
            res += std::log(x) - 1.0 / 2.0 / x - 1.0 / 12.0 / std::pow(x, 2)
                + 1.0 / 120.0 / std::pow(x, 4) - 1.0 / 252.0 / std::pow(x, 6)
                + 1.0 / 240.0 / std::pow(x, 8) - 5.0 / 660.0 / std::pow(x, 10)
                + 691.0 / 32760.0 / std::pow(x, 12) - 1.0 / 12.0 / std::pow(x, 14);
            return res;
        }
        if (x < 1e3){
            res += std::log(x) - 1.0 / 2.0 / x - 1.0 / 12.0 / std::pow(x, 2)
                + 1.0 / 120.0 / std::pow(x, 4) - 1.0 / 252.0 / std::pow(x, 6)
                + 1.0 / 240.0 / std::pow(x, 8) - 5.0 / 660.0 / std::pow(x, 10);
            return res;
        }
        if (x < 1e4){
            res += std::log(x) - 1.0 / 2.0 / x - 1.0 / 12.0 / std::pow(x, 2)
                + 1.0 / 120.0 / std::pow(x, 4) - 1.0 / 252.0 / std::pow(x, 6)
                + 1.0 / 240.0 / std::pow(x, 8);
            return res;
        }
        if (x < 1e5){
            res += std::log(x) - 1.0 / 2.0 / x - 1.0 / 12.0 / std::pow(x, 2)
                + 1.0 / 120.0 / std::pow(x, 4) - 1.0 / 252.0 / std::pow(x, 6);
            return res;
        }
        if (x < 1e8){
            res += std::log(x) - 1.0 / 2.0 / x - 1.0 / 12.0 / std::pow(x, 2)
                + 1.0 / 120.0 / std::pow(x, 4);
            return res;
        }
        if (x < 1e15){
            res += std::log(x) - 1.0 / 2.0 / x - 1.0 / 12.0 / std::pow(x, 2);
            return res;
        }
        res += std::log(x);
        return res;
    }

    template<typename Vec, typename Scalar>
    void vec2arr(const Vec &in, Scalar *out){
        for (size_t i = 0; i < in.size(); ++i){
            out[i] = in[i];
        }
    }
    template<typename Vec, typename Scalar>
    void arr2vec(const Scalar *in, Vec &out, size_t n){
        for (size_t i = 0; i < n; ++i){
            out[i] = in[i];
        }
    }

    template<class MatType>
    void pinv(const MatType &inMat, MatType &inMatPinv)
    {
        Eigen::JacobiSVD<MatType> jsvd(inMat, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        struct reciprocalNonZero{
            typename MatType::Scalar operator()(typename MatType::Scalar a) const {
                const double EPSILON = 1.0e-20;
                if (std::abs(a) > std::abs(EPSILON)){
                    return 1.0 / a;
                }
                else{
                    return 0;
                }
            }
        };
        inMatPinv.resize(inMat.rows(), inMat.cols());
        inMatPinv.noalias() = jsvd.matrixV() * jsvd.singularValues().unaryExpr(reciprocalNonZero()).asDiagonal() * jsvd.matrixU().adjoint();
    }

    class betareg
    {
    public:

        bool loadTrainData(std::istream &datass, bool appendBias){
            using namespace std;
            using namespace Eigen;

            if (!datass){
                cerr << "open sample data failed" << endl;
                return false;
            }
            string line;
            stringstream liness;
            // n is sample number, k is feature number.
            size_t n;
            size_t k;
            if (getline(datass, line)){
                liness.clear();
                liness.str(line);
                liness >> n >> k;
            }
            else{
                cerr << "read from data failed at line " << 1 << endl;
                return false;
            }
            if (appendBias){
                trainSampFeat.resize(n, k + 1);
            }
            else{
                trainSampFeat.resize(n, k);
            }
            featWeight.resize(trainSampFeat.cols());
            trainSampWeight.resize(n);
            trainSampResp.resize(n);
            VectorXd linefeats(k + 2);
            for (size_t i = 0; i < n; ++i){
                if (!getline(datass, line)){
                    cerr << "read from data failed at line " << i + 2 << endl;
                    return false;
                }
                liness.clear();
                liness.str(line);
                // the first column is the sample weight, second column is the sample response.
                for (size_t j = 0; j < k + 2; ++j){
                    double feat;
                    if (liness >> feat){
                        linefeats(j) = feat;
                    }
                    else{
                        cerr << "read from data failed at line " << i + 2 << ", column " << j + 1 << endl;
                        return false;
                    }
                }
                trainSampWeight(i) = linefeats(0);
                trainSampResp(i) = linefeats(1);
                const double EPSILON = 1.0e-20;
                if (trainSampResp(i) > 1.0 - EPSILON){
                    trainSampResp(i) = 1.0 - EPSILON;
                }
                else{
                    if (trainSampResp(i) < EPSILON){
                        trainSampResp(i) = EPSILON;
                    }
                }
                trainSampFeat.row(i).head(k).noalias() = linefeats.tail(k);
                if (appendBias){
                    trainSampFeat(i, k) = 1.0;
                }
            }

            //double maxw = trainSampWeight.maxCoeff();
            //trainSampWeight /= maxw;
            return true;
        }

        bool betafit(const std::string outweightfile){
            this->outweightfile = outweightfile;

            // get initial values
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            init();
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::cerr << std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() << " seconds used for init weights" << std::endl;

            lbfgs_parameter_t lbfgs_param;
            lbfgs_parameter_init(&lbfgs_param);
            lbfgs_param.m = 15;
            lbfgs_param.max_linesearch = 50;
            size_t n = featWeight.size() + 1;
            double *weights_all = lbfgs_malloc(n);
            vec2arr(featWeight, weights_all);
            weights_all[n - 1] = precisionParam;

            double finalLoss = 0;
            int suc = lbfgs(n, weights_all, &finalLoss, funcValGrad, fitProg, this, &lbfgs_param);
            std::cerr << "lbfgs routine finished with code: " << suc << std::endl;

            std::ofstream outweight(this->outweightfile);
            dumpArr(outweight, weights_all, n);
            arr2vec(weights_all, featWeight, n - 1);
            precisionParam = weights_all[n - 1];

            lbfgs_free(weights_all);
            return true;
        }

    private:
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> trainSampFeat;
        //Eigen::MatrixXd trainSampFeat;
        Eigen::VectorXd trainSampWeight;
        Eigen::VectorXd trainSampResp;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> testSampFeat;
        //Eigen::MatrixXd testSampFeat;
        Eigen::VectorXd featWeight;
        Eigen::VectorXd featWeightGrad;
        double precisionParam; // the phi
        double precisionParamGrad;
        std::chrono::high_resolution_clock::time_point clk;
        std::string outweightfile;

        bool init(void){
            Eigen::MatrixXd trainSampHInv;
            pinv<Eigen::MatrixXd>(trainSampFeat.transpose() * trainSampWeight.asDiagonal() * trainSampFeat, trainSampHInv);
            Eigen::VectorXd zt = trainSampResp.unaryExpr(std::ptr_fun(logitfunc));
            featWeight = trainSampHInv * trainSampFeat.transpose() * trainSampWeight.asDiagonal() * zt;
            Eigen::VectorXd yt = trainSampFeat * featWeight;
            Eigen::VectorXd mut = yt.unaryExpr(std::ptr_fun(sigmoidfunc));
            Eigen::VectorXd et = (zt - yt).cwiseProduct(trainSampWeight);

            /*
            Eigen::VectorXd sigt = et.transpose() * et / (trainSampWeight.sum() - trainSampFeat.cols())
            * mut.unaryExpr(std::ptr_fun(logitDevRep)).square();
            precisionParam = mut.unaryExpr(std::ptr_fun(logitDevRep)).cwiseQuotient(sigt).cwiseProduct(trainSampWeight).sum()
            / trainSampWeight.sum() - 1.0;
            */

            // can also be computed
            double sigt = et.squaredNorm() / (trainSampWeight.sum() - trainSampFeat.cols());
            precisionParam = trainSampWeight.cwiseQuotient(mut.unaryExpr(std::ptr_fun(logitDevRep)) * sigt).sum()
                / trainSampWeight.sum() - 1.0;

            return true;
        }

        double lossfunc(bool calcGrad = false){
            clk = std::chrono::high_resolution_clock::now();
            double loglikely = 0;
            Eigen::VectorXd mus(trainSampWeight.size());
            Eigen::VectorXd mupprec(trainSampWeight.size());
            double lgmPreP = std::lgamma(precisionParam);

            /*
            #pragma omp parallel for reduction(+:loglikely)
            for (size_t i = 0; i < trainSampWeight.size(); ++i){
                mus(i) = sigmoidfunc(trainSampFeat.row(i) * featWeight);
                mupprec(i) = mus(i) * precisionParam;
                double logloss = lgmPreP - std::lgamma(mupprec(i))
                    - std::lgamma(precisionParam - mupprec(i))
                    + (mupprec(i) - 1.0) * std::log(trainSampResp(i))
                    + (precisionParam - mupprec(i) - 1.0)
                    * std::log(1.0 - trainSampResp(i));
                loglikely += logloss * trainSampWeight(i);
            }*/

            
            // not using openmp to compute log loss.
            mus = (trainSampFeat * featWeight).unaryExpr(std::ptr_fun(sigmoidfunc));
            mupprec = precisionParam * mus;
            loglikely = (lgmPreP*Eigen::VectorXd::Ones(mupprec.size())
                - mupprec.unaryExpr(std::ptr_fun<double, double>(std::lgamma))
                - (precisionParam * Eigen::VectorXd::Ones(mupprec.size()) - mupprec).unaryExpr(std::ptr_fun<double, double>(std::lgamma))
                + (mupprec - Eigen::VectorXd::Ones(mupprec.size())).cwiseProduct(trainSampResp.unaryExpr(std::ptr_fun<double, double>(std::log)))
                + ((precisionParam - 1.0)*Eigen::VectorXd::Ones(mupprec.size()) - mupprec)
                .cwiseProduct((Eigen::VectorXd::Ones(trainSampResp.size()) - trainSampResp)
                .unaryExpr(std::ptr_fun<double, double>(std::log))))
                .cwiseProduct(trainSampWeight).sum();
            // let's remember that the objective is to maximize the loglikely above,
            // but the optimization tool minimizes functions.
            loglikely *= -1.0;

            if (calcGrad){
                Eigen::VectorXd mut_p_m1 = (precisionParam * Eigen::VectorXd::Ones(mupprec.size())
                    - mupprec).unaryExpr(std::ptr_fun(digamma));
                Eigen::VectorXd mus_star = mupprec.unaryExpr(std::ptr_fun(digamma)) - mut_p_m1;
                Eigen::VectorXd resp_star = trainSampResp.unaryExpr(std::ptr_fun(logitfunc));
                featWeightGrad.noalias() = trainSampFeat.transpose()*trainSampWeight.asDiagonal()
                    * mus.unaryExpr(std::ptr_fun(logitDevRep)).asDiagonal() * (resp_star - mus_star);
                featWeightGrad *= precisionParam;

                precisionParamGrad = ((resp_star - mus_star).cwiseProduct(mus)
                    + (Eigen::VectorXd::Ones(trainSampResp.size()) - trainSampResp).unaryExpr(std::ptr_fun<double, double>(std::log))
                    - mut_p_m1 + digamma(precisionParam)*Eigen::VectorXd::Ones(trainSampResp.size()))
                    .cwiseProduct(trainSampWeight).sum();

                // for the same reason above, we need to multiply by -1
                featWeightGrad *= -1.0;
                precisionParamGrad *= -1.0;
            }

            return loglikely;
        }

        static lbfgsfloatval_t funcValGrad(void *betaregVar, const lbfgsfloatval_t *weigts,
            lbfgsfloatval_t *gradient, const int n, const lbfgsfloatval_t step){
            betareg *beta = reinterpret_cast<betareg *>(betaregVar);
            // the last element of the weights is the precision parameter.
            arr2vec(weigts, beta->featWeight, n - 1);
            beta->precisionParam = weigts[n - 1];
            double loss = beta->lossfunc(true);
            vec2arr(beta->featWeightGrad, gradient);
            gradient[n - 1] = beta->precisionParamGrad;
            return loss;
        }

        static int fitProg(void *betaregVar, const lbfgsfloatval_t *weights,
            const lbfgsfloatval_t *gradient, const lbfgsfloatval_t lossval,
            const lbfgsfloatval_t weightNorm, const lbfgsfloatval_t weightGradNorm,
            const lbfgsfloatval_t step, int n, int iter, int evalNum){
            betareg *beta = reinterpret_cast<betareg *>(betaregVar);

            if (iter % 100 == 0){
                // dump intermediate result to file, just in case something should happen.
                std::ofstream outweighttmp(beta->outweightfile+".tmp");
                dumpArr(outweighttmp, weights, n);
            }
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::cerr << "iteration count: " << iter << ", loss function value: " << lossval
                    << ", gradient norm: " << weightGradNorm << ", step: " << step
                    << ", loss function evaluated times: " << evalNum
                    << ", time used for one loss compute: " << std::chrono::duration_cast<std::chrono::duration<double> >(t2 - beta->clk).count()
                    << " seconds" << std::endl;
            return 0;
        }
    };

}