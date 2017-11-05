#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <cmath>

using p_matrix  = std::shared_ptr<Eigen::MatrixXd>;
using vec_matrix = std::vector <p_matrix>;

inline auto make_random(const int& rows, const int& cols) { return p_matrix(new Eigen::MatrixXd(Eigen::MatrixXd::Random(rows, cols))); }
inline auto make_zero(const int& rows, const int& cols) { return p_matrix(new Eigen::MatrixXd(Eigen::MatrixXd::Zero(rows, cols))); }

// set of activation function
inline void logistic(p_matrix in, p_matrix out) {*out =  1 / (1 + (-in->array()).exp()); }
inline void relu(p_matrix in, p_matrix out) {*out = in->cwiseMax(0); }
inline void softmax(p_matrix input, p_matrix output) {
	*output = input->array().exp();
	*output = output->array().colwise() / output->array().rowwise().sum();
}

// set of the derivate of activation function
inline void logisticD(p_matrix in, p_matrix out) { logistic(in, out); *out = out->array() * (1 - out->array()); };
inline void reluD(p_matrix in, p_matrix out) {}; //TODO


// set of cost function
inline double euclidian(p_matrix labels, p_matrix predictions) { return (labels->array() - predictions->array()).matrix().rowwise().squaredNorm().sum(); };
inline double crossEntropy(p_matrix labels, p_matrix predictions) { return 1; }; // TODO

//print training accuracy
inline void print_state(unsigned epoch, double cost, double accuracy, double temps_ms)
{
	printf("Epoch #%d, Training error : %3.3f, Accuracy : %1.3f, Time : %1.0f ms \n", epoch, cost, accuracy, temps_ms);
	std::cout << std::flush;
}

//set of gradient descent algo
inline void SGD(vec_matrix weights, vec_matrix bias, vec_matrix local_gradients, vec_matrix y, p_matrix input,
	vec_matrix weights_past_update, vec_matrix bias_past_update, int input_size, double learning_rate, double momentum) {
	for (unsigned k = 0; k < weights.size(); k++) {
		*weights[k] += -learning_rate / input_size * (*local_gradients[k] * (k == 0 ? *input : *y[k - 1])).transpose();
		*bias[k] += -learning_rate *((*local_gradients[k]).transpose()).colwise().mean();
	}
}

inline void momentum_GD(vec_matrix weights_, vec_matrix bias_, vec_matrix local_gradients_, vec_matrix y_, p_matrix input_,
	vec_matrix weights_past_update_, vec_matrix bias_past_update_, int input_size_, double learning_rate_, double momentum_) {
	for (unsigned k = 0; k < weights_.size(); k++) {
		*weights_past_update_[k] = momentum_ * (*weights_past_update_[k])
			+ learning_rate_ / input_size_ * (*local_gradients_[k] * (k == 0 ? *input_ : *y_[k - 1])).transpose();
		*weights_[k] += -*weights_past_update_[k];

		*bias_past_update_[k] = momentum_ * (*bias_past_update_[k])
			+ learning_rate_ *((*local_gradients_[k]).transpose()).colwise().mean();
		*bias_[k] += -*bias_past_update_[k];
	}
}