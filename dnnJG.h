#pragma once

#include <vector>
#include <iostream>
#include <Eigen/Dense>	//matrix library
#include <ctime>		//calculate time of epoch
#include <omp.h>		//multithreading
#include <memory>
#include <algorithm>
#include <functional>

#include "dnn_essentials.h"

class dnnJG
{
public:
	//consructor
	dnnJG(std::vector<int> layers, p_matrix train_data,	p_matrix train_labels, p_matrix test_data, 	p_matrix test_labels,
			unsigned batchSize = 50, int activation_function = -1, 	int cost_function = -1, double learning_rate = 0.1, 
			double momentum = 0.01, int weight_update_function = 1);

	virtual ~dnnJG();

	void print_structure();

	// wraps all stage of learning
	void train(unsigned nb_epochs);

	// wraps inference for new data
	void inference();

	//different stage of learning
	void forward (p_matrix input, bool inference = false);
	void add_bias(p_matrix bias, p_matrix output, unsigned inputSize);
	void backward(p_matrix labels);
	void update_weights();

	// show state of network
	double check_accuracy(p_matrix pred,  p_matrix labels);

	void shape_for_new_size(int size);

protected:

	std::vector<int> layers_;

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

	//data
	p_matrix train_data_,  train_labels_;		//train data
	p_matrix test_data_,   test_labels_;		//test data	
	p_matrix input_, labels_;

	// pointers to the functions used in backpropagation
	std::string activation_function_name_;
	std::string cost_function_name_;

	double(*activation_function_) (const double&);
	double(*activation_function_derivative_) (const double&);
	void(*cost_function_) (double*, Eigen::MatrixXd*, Eigen::MatrixXd*);

	std::function<void(vec_matrix weights, vec_matrix bias, vec_matrix local_gradients, vec_matrix y, p_matrix input,
					   vec_matrix weights_past_update, vec_matrix bias_past_update, int input_size, double learning_rate, double momentum)> GD_algo;
};

