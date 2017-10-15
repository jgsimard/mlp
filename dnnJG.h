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
			std::shared_ptr<Eigen::MatrixXd> train_data,
			std::shared_ptr<Eigen::MatrixXd> train_labels, 
			std::shared_ptr<Eigen::MatrixXd> test_data, 
			std::shared_ptr<Eigen::MatrixXd> test_labels,
			unsigned batchSize = 100, 
			int activation_function = -1, 
			int cost_function = -1, 
			double learning_rate = 1.0, 
			double momentum = 0.3, 
			int weight_update_function = -1);

	virtual ~dnnJG();

	//
	void print_state();

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
	void printStateE(unsigned currentEpoch, double accuracy, double temps);
	void printStateEB(unsigned currentEpoch, unsigned currentBatch, double accuracy, double temps);
	void printStateEBCostOnly(unsigned currentEpoch, unsigned currentBatch, double temps);

protected:
	//hyper-parameters
	unsigned nb_layers_, batch_size_;
	unsigned input_size_, test_size_;	//dataset size
	double learning_rate_, momentum_;	//learning factors

	double cost_;
	double accuracy_;
	int weight_update_function_;

	//parameters to be tuned
	std::vector <Eigen::MatrixXd*> weights_, bias_;

	std::vector <Eigen::MatrixXd*> weights_past_update_, bias_past_update_;	//for SGD
	std::vector <Eigen::MatrixXd*> v_, y_;								    //batch predictions
	std::vector <Eigen::MatrixXd*> y_d_, local_gradients_;					//for backpropagation
	std::vector <Eigen::MatrixXd*> v_I, y_I;								//inference predictions

	//data
	std::shared_ptr<Eigen::MatrixXd> train_data_,  train_labels_;		//train data
	std::shared_ptr<Eigen::MatrixXd> test_data_,   test_labels_;		//test data	
	std::shared_ptr<Eigen::MatrixXd> batch_input_, batch_labels_;		//batch data

	// pointers to the functions used in backpropagation
	std::string activation_function_name_;
	std::string cost_function_name_;
	double(*activation_function_) (const double&);
	double(*activation_function_derivative_) (const double&);
	void(*cost_function_) (double*, Eigen::MatrixXd*, Eigen::MatrixXd*);
};