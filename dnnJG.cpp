#include "dnnJG.h"

using namespace Eigen;

dnnJG::dnnJG(std::vector<int> layers, MatrixXd *trainData, MatrixXd *trainLabels, MatrixXd *testData, MatrixXd *testLabels,
	unsigned batchSize, int activationFunction, int costFunction, double learningRate, double momentum, int updateWeightFunction) :
	m_batchSize(batchSize), m_trainData(trainData), m_trainLabels(trainLabels), m_nbLayers(unsigned(layers.size())),
	m_learningRate(learningRate), m_momentum(momentum), m_inputSize(unsigned(trainData->rows())), m_updateWeightFunction(updateWeightFunction),
	m_testData(testData), m_testLabels(testLabels),
	m_testSize(unsigned(testData->rows()))
{
	m_batchInput = new MatrixXd(MatrixXd::Zero(m_batchSize, layers.front()));
	m_batchLabels = new MatrixXd(MatrixXd::Zero(m_batchSize, layers.back()));

	for (unsigned i = 0; i < m_nbLayers - 1; i++) {
		m_v.push_back(new MatrixXd(MatrixXd::Zero(m_batchSize, layers[i + 1])));
		m_y.push_back(new MatrixXd(MatrixXd::Zero(m_batchSize, layers[i + 1])));

		m_vI.push_back(new MatrixXd(MatrixXd::Zero(m_testSize, layers[i + 1])));
		m_yI.push_back(new MatrixXd(MatrixXd::Zero(m_testSize, layers[i + 1])));

		m_yD.push_back(new MatrixXd(MatrixXd::Zero(layers[i + 1], m_batchSize)));
		m_localGradients.push_back(new MatrixXd(MatrixXd::Zero(layers[i + 1], m_batchSize)));

		m_weights.push_back(new MatrixXd(MatrixXd::Random(layers[i], layers[i + 1])));
		m_bias.push_back(new MatrixXd(MatrixXd::Random(1, layers[i + 1])));

		m_weightsPastUpdate.push_back(new MatrixXd(MatrixXd::Random(layers[i], layers[i + 1])));
		m_biasPastUpdate.push_back(new MatrixXd(MatrixXd::Random(1, layers[i + 1])));
	}
	switch (activationFunction) {
	case 1:  m_activationFunction = &identity;	m_activationFunctionDerivative = &identityD; break;
	case 2:  m_activationFunction = &logistic;	m_activationFunctionDerivative = &logisticD; break;
	case 3:  m_activationFunction = &binary;	m_activationFunctionDerivative = &binaryD;	 break;
	case 4:  m_activationFunction = &relu;		m_activationFunctionDerivative = &reluD;	 break;
	default: m_activationFunction = &logistic;	m_activationFunctionDerivative = &logisticD; break;
	}

	switch (costFunction) {
	case 1:  m_costFunction = &crossEntropy;	break;
	default: m_costFunction = &euclidian;		break;
	}

	printf("Initialization complete \n");
}
dnnJG::~dnnJG()
{
	delete m_batchLabels;
	delete m_batchInput;

	for (unsigned i = 0; i < m_nbLayers - 1; i++) {
		delete m_v.back();
		delete m_y.back();
		delete m_localGradients.back();
		delete m_weights.back();
		delete m_bias.back();

		m_v.pop_back();
		m_y.pop_back();
		m_localGradients.pop_back();
		m_weights.pop_back();
		m_bias.pop_back();
	}
}

void dnnJG::printStateE(unsigned currentEpoch, double accuracy, double temps)
{
	printf("Epoch #%d, Training error : %3.3f, Accuracy : %1.3f, Time : %1.0f ms \n", currentEpoch, m_cost, accuracy, temps);
	std::cout << std::flush;
}
void dnnJG::printStateEB(unsigned currentEpoch, unsigned currentBatch, double accuracy, double temps)
{
	printf("Epoch #%d, Batch #%d, Training error : %3.3f, Accuracy : %1.3f, Time : %1.0f ms \n", currentEpoch, currentBatch, m_cost, accuracy, temps);
	std::cout << std::flush;
}
void dnnJG::printStateEBCostOnly(unsigned currentEpoch, unsigned currentBatch, double temps)
{
	printf("Epoch #%d, Batch #%d, Training error : %3.3f,  Time : %1.0f ms \n", currentEpoch, currentBatch, m_cost, temps);
	std::cout << std::flush;
}

int dnnJG::train(unsigned nbEpochs) {
	omp_set_num_threads(2);
	setNbThreads(2);
	m_inputSize = unsigned(m_trainData->rows());
	for (unsigned i = 0; i < nbEpochs; i++) {
		//set up permatation matrix
		//PermutationMatrix <Dynamic, Dynamic> perm(m_trainData->size());
		//perm.setIdentity();
		//std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		//permutate the rows
		//*m_trainData   = perm * (*m_trainData);
		//*m_trainLabels = perm * (*m_trainLabels);
		clock_t start = clock();
		for (unsigned j = 0; j < m_inputSize / m_batchSize; j++) {
			m_cost = 0;
			forward(m_trainData, m_batchInput, m_batchSize, j * m_batchSize);
			*m_batchLabels = m_trainLabels->block(j*m_batchSize, 0, m_batchSize, m_trainLabels->cols());
			backward(m_batchLabels);
			m_costFunction(&m_cost, m_batchLabels, m_y.back());
			updateWeight();
			/*m_accuracy = checkAccuracy(m_y.back(), m_batchLabels);
			printStateEB(i, j, m_accuracy, clock() - start);
			printStateEBCostOnly(i, j, clock() - start);
			start = clock();
			printStateE(i * m_inputSize/m_batchSize + j, m_accuracy);*/
		}
		inference();
		m_accuracy = checkAccuracy(m_yI.back(), m_testLabels);
		printStateE(i, m_accuracy, clock() - start);
		start = clock();
	}
	return 0;
}


void dnnJG::addBias(Eigen::MatrixXd *bias, Eigen::MatrixXd *output, unsigned inputSize) {
	for (unsigned k = 0; k < inputSize; k++)
		output->block(k, 0, 1, output->cols()) += *bias;
}

void dnnJG::forward(Eigen::MatrixXd *data, Eigen::MatrixXd *input, unsigned inputSize, int position) {
	*input = data->block(position, 0, inputSize, data->cols()); //must use temp because eigen do not support matrix multiplication of block
	for (unsigned k = 0; k < m_weights.size(); k++) {
		*m_v[k] = (k == 0 ? (*input) * (*m_weights[k]) : (*m_y[k - 1]) * (*m_weights[k]));
		addBias(m_bias[k], m_v[k], m_batchSize);
		*m_y[k] = m_v[k]->unaryExpr(m_activationFunction);
		*m_yD[k] = (m_v[k]->unaryExpr(m_activationFunctionDerivative)).transpose();
	}
}
void dnnJG::backward(Eigen::MatrixXd *labels) {

	*m_localGradients.back() = (*m_y.back() - *labels).transpose();
	for (size_t k = m_localGradients.size() - 1; k > 0; k--)
		*m_localGradients[k - 1] = ((*m_yD[k - 1]).array() * (*m_weights[k] * (*m_localGradients[k])).array()).matrix();
}
void dnnJG::inference() {
	for (unsigned k = 0; k < m_weights.size(); k++) {
		*m_vI[k] = k == 0 ? (*m_testData) * (*m_weights[k]) : (*m_yI[k - 1]) * (*m_weights[k]);
		addBias(m_bias[k], m_vI[k], m_testSize);
		*m_yI[k] = m_vI[k]->unaryExpr(m_activationFunction);
	}
}
double dnnJG::checkAccuracy(Eigen::MatrixXd *predic, Eigen::MatrixXd *labels) {
	int count = 0;
	MatrixXd::Index dummy, maxLabelsCol, maxPredicCol;
	for (int i = 0; i < labels->rows(); i++) {
		labels->block(i, 0, 1, labels->cols()).maxCoeff(&dummy, &maxLabelsCol);
		predic->block(i, 0, 1, predic->cols()).maxCoeff(&dummy, &maxPredicCol);
		if (maxLabelsCol == maxPredicCol)
			count++;
	}
	return (double)count / labels->rows();
}
void dnnJG::updateWeight() {
	switch (m_updateWeightFunction) {
	case 0: naiveSGD();		break;
	case 1: momentumSGD();	break;
	default: naiveSGD();	break;
	}
}
void dnnJG::naiveSGD() {
	for (unsigned k = 0; k < m_weights.size(); k++) {
		*m_weights[k] += -m_learningRate / m_batchSize * (*m_localGradients[k] * (k == 0 ? *m_batchInput : *m_y[k - 1])).transpose();
		*m_bias[k] += -m_learningRate *((*m_localGradients[k]).transpose()).colwise().mean();
	}
}
void dnnJG::momentumSGD() {
	for (unsigned k = 0; k < m_weights.size(); k++) {
		*m_weightsPastUpdate[k] = m_momentum * (*m_weightsPastUpdate[k]) + m_learningRate / m_batchSize * (*m_localGradients[k] * (k == 0 ? *m_batchInput : *m_y[k - 1])).transpose();
		*m_weights[k] += -*m_weightsPastUpdate[k];

		*m_biasPastUpdate[k] = m_momentum * (*m_biasPastUpdate[k]) + m_learningRate *((*m_localGradients[k]).transpose()).colwise().mean();
		*m_bias[k] += -*m_biasPastUpdate[k];
	}
}
