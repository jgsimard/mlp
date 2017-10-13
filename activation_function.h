#pragma once

#include <Eigen/Dense>

// set of activation function
static double identity(const double& x) { return x; };
static double logistic(const double& x) { return 1 / (1 + std::exp(-x)); };
static double binary(const double& x) { return x < 0 ? 0 : 1; };
static double relu(const double& x) { return x < 0 ? 0 : x; };

// set of the derivate of activation function
static double identityD(const double& x) { return 1; };
static double logisticD(const double& x) { return logistic(x)*(1 - logistic(x)); };
static double binaryD(const double& x) { return 0; };
static double reluD(const double& x) { return x < 0 ? 0 : 1; };

// set of cost function
void euclidian(double *cost, Eigen::MatrixXd *labels, Eigen::MatrixXd *predictions) { *cost = (*labels - *predictions).squaredNorm(); };
void crossEntropy(double *cost, Eigen::MatrixXd *labels, Eigen::MatrixXd *predictions) { *cost = 1; }; // TODO