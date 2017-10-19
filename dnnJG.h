#pragma once

#include <vector>
#include <iostream>
#include <Eigen/Dense>	//matrix library
#include <ctime>		//calculate time of epoch
#include <omp.h>		//multithreading
#include <memory>

#include "activation_function.h"

class dnnJG
{
public:
	//consructor
	dnnJG(std::vector<int> layers, 
			p_matrix train_data,
			p_matrix train_labels, 
			p_matrix test_data, 
			p_matrix test_labels,
			unsigned batchSize = 50, 
			int activation_function = -1, 
			int cost_function = -1, 
			double learning_rate = 0.05, 
			double momentum = 0.01, 
			int weight_update_function = -1);

	virtual ~dnnJG();

	void print_structure();

	// wraps all stage of learning
	int train(unsigned nbEpochs);

	// wraps inference for new data
	void inference();

	//different stage of learning
	void forward(Eigen::MatrixXd *data, Eigen::MatrixXd *input, unsigned inputSize, int position);
	void add_bias(Eigen::MatrixXd *bias, Eigen::MatrixXd *output, unsigned inputSize);
	void backward(Eigen::MatrixXd *input);
	void updateWeight();

	// gradient descent algorithms
	void naiveSGD();
	void momentumSGD();

	// show state of network
	double check_accuracy(Eigen::MatrixXd *pred, Eigen::MatrixXd *labels);

protected:
	//hyper-parameters
	unsigned nb_layers_, batch_size_;
	unsigned input_size_, test_size_;	//dataset size
	double learning_rate_, momentum_;	//learning factors

	double cost_;
	double accuracy_;
	int weight_update_function_;

	//parameters to be tuned
	vec_matrix weights_, bias_;

	vec_matrix weights_past_update_, bias_past_update_;	//for SGD
	vec_matrix v_, y_;								    //batch predictions
	vec_matrix y_d_, local_gradients_;					//for backpropagation
	vec_matrix v_I, y_I;								//inference predictions

	//data
	p_matrix train_data_,  train_labels_;		//train data
	p_matrix test_data_,   test_labels_;		//test data	
	p_matrix batch_input_, batch_labels_;		//batch data

	// pointers to the functions used in backpropagation
	std::string activation_function_name_;
	std::string cost_function_name_;
	double(*activation_function_) (const double&);
	double(*activation_function_derivative_) (const double&);
	void(*cost_function_) (double*, Eigen::MatrixXd*, Eigen::MatrixXd*);
};