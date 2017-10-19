#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <memory>

typedef std::shared_ptr<Eigen::MatrixXd> p_matrix;
typedef std::vector <p_matrix> vec_matrix;

inline auto make_random(const int& rows, const int& cols) { return p_matrix(new Eigen::MatrixXd(Eigen::MatrixXd::Random(rows,cols))); }
inline auto make_zero(const int& rows, const int& cols) { return p_matrix(new Eigen::MatrixXd(Eigen::MatrixXd::Zero(rows, cols))); }


// set of activation function
inline double identity(const double& x){ return x; };
inline double logistic(const double& x){ return 1 / (1 + std::exp(-x)); };
inline double binary(const double& x)  { return x < 0 ? 0 : 1; };
inline double relu(const double& x)    { return std::max(0.0, x); };

// set of the derivate of activation function
inline double identityD(const double& x){ return 1; };
inline double logisticD(const double& x){ return logistic(x)*(1 - logistic(x)); };
inline double binaryD(const double& x)  { return 0; };
inline double reluD(const double& x)    { return x < 0.0 ? 0.0 : 1.0; };

// set of cost function
inline void euclidian(double *cost, Eigen::MatrixXd *labels, Eigen::MatrixXd *predictions) { *cost = (*labels - *predictions).squaredNorm();};
inline void crossEntropy(double *cost, Eigen::MatrixXd *labels, Eigen::MatrixXd *predictions) { *cost = 1; }; // TODO

inline void print_state(unsigned epoch, double cost, double accuracy, double temps_ms)
{
	printf("Epoch #%d, Training error : %3.3f, Accuracy : %1.3f, Time : %1.0f ms \n", epoch, cost, accuracy, temps_ms);
	std::cout << std::flush;
}

//set of gradient descent algo
/*
void SGD(weights, bias, local_gradients, y, input, leargning_rate) {
	for (unsigned k = 0; k < weights_.size(); k++) {
		*weights_[k] += -learning_rate_ / batch_size_ * (*local_gradients_[k] * (k == 0 ? *batch_input_ : *y_[k - 1])).transpose();
		*bias_[k] += -learning_rate_ *((*local_gradients_[k]).transpose()).colwise().mean();
	}
}
void dnnJG::momentum_GD() {
	for (unsigned k = 0; k < weights_.size(); k++) {
		*weights_past_update_[k] = momentum_ * (*weights_past_update_[k])
			+ learning_rate_ / batch_size_ * (*local_gradients_[k] * (k == 0 ? *batch_input_ : *y_[k - 1])).transpose();
		*weights_[k] += -*weights_past_update_[k];

		*bias_past_update_[k] = momentum_ * (*bias_past_update_[k])
			+ learning_rate_ *((*local_gradients_[k]).transpose()).colwise().mean();
		*bias_[k] += -*bias_past_update_[k];
	}
}
*/