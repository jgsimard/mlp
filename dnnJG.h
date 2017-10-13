#pragma once
#include <vector>
#include <iostream>
#include <Eigen/Dense>	//matrix library
#include <ctime>		//calculate time of epoch
#include <omp.h>		//multithreading

#include "activation_function.h"
class dnnJG
{
public:
	//consructor
	dnnJG(std::vector<int> layers, 
			Eigen::MatrixXd *trainData,
			Eigen::MatrixXd *trainLabels, 
			Eigen::MatrixXd *testData, 
			Eigen::MatrixXd *testLabels,
			unsigned batchSize = 100, 
			int activationFunction = -1, 
			int costFunction = -1, 
			double learningRate = 1.0, 
			double momentum = 0.3, 
			int updateWeightFunction = -1);

	virtual ~dnnJG();

	// wraps all stage of learning
	int train(unsigned nbEpochs);

	// wraps inference for new data
	void inference();

	//different stage of learning
	void forward(Eigen::MatrixXd *data, Eigen::MatrixXd *input, unsigned inputSize, int position);
	void addBias(Eigen::MatrixXd *bias, Eigen::MatrixXd *output, unsigned inputSize);
	void backward(Eigen::MatrixXd *input);
	void updateWeight();

	// gradient descent algorithms
	void naiveSGD();
	void momentumSGD();

	// show state of network
	double checkAccuracy(Eigen::MatrixXd *pred, Eigen::MatrixXd *labels);
	void printStateE(unsigned currentEpoch, double accuracy, double temps);
	void printStateEB(unsigned currentEpoch, unsigned currentBatch, double accuracy, double temps);
	void printStateEBCostOnly(unsigned currentEpoch, unsigned currentBatch, double temps);

protected:
	//hyper-parameters
	unsigned m_nbLayers, m_batchSize;
	unsigned m_inputSize, m_testSize;	//dataset size
	double m_learningRate, m_momentum;	//learning factors

	double m_cost;
	double m_accuracy;
	int m_updateWeightFunction;

	//parameters to be tuned
	std::vector <Eigen::MatrixXd*> m_weights, m_bias;

	std::vector <Eigen::MatrixXd*> m_weightsPastUpdate, m_biasPastUpdate;	//for SGD
	std::vector <Eigen::MatrixXd*> m_v, m_y;								//batch predictions
	std::vector <Eigen::MatrixXd*> m_yD, m_localGradients;					//for backpropagation
	std::vector <Eigen::MatrixXd*> m_vI, m_yI;								//inference predictions

																			//data
	Eigen::MatrixXd *m_trainData, *m_trainLabels;		//train data
	Eigen::MatrixXd *m_testData, *m_testLabels;		//test data	
	Eigen::MatrixXd *m_batchInput, *m_batchLabels;		//batch data

														// pointers to the functions used in backpropagation
	double(*m_activationFunction) (const double&);
	double(*m_activationFunctionDerivative) (const double&);
	void(*m_costFunction) (double*, Eigen::MatrixXd*, Eigen::MatrixXd*);
};