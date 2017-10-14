#include "dnnJG.h"

using namespace Eigen;

// set of cost function
void euclidian(double *cost, MatrixXd *labels, MatrixXd *predictions) { *cost = (*labels - *predictions).squaredNorm(); };
void crossEntropy(double *cost, MatrixXd *labels, MatrixXd *predictions) { *cost = 1; }; // TODO

dnnJG::dnnJG(std::vector<int> layers, MatrixXd *train_data, MatrixXd *train_labels, MatrixXd *test_data, MatrixXd *test_labels,
	unsigned batchSize, int activation_function, int cost_function, double learning_rate, double momentum, int updateWeightFunction) :
	batch_size_(batchSize), train_data_(train_data), train_labels_(train_labels), nb_layers_(unsigned(layers.size())),
	learning_rate_(learning_rate), momentum_(momentum), input_size_(unsigned(train_data->rows())), weight_update_function_(updateWeightFunction),
	test_data_(test_data), test_labels_(test_labels),
	test_size_(unsigned(test_data->rows()))
{
	batch_input_ = new MatrixXd(MatrixXd::Zero(batch_size_, layers.front()));
	batch_labels_ = new MatrixXd(MatrixXd::Zero(batch_size_, layers.back()));

	for (unsigned i = 0; i < nb_layers_ - 1; i++) {
		v_.push_back(new MatrixXd(MatrixXd::Zero(batch_size_, layers[i + 1])));
		y_.push_back(new MatrixXd(MatrixXd::Zero(batch_size_, layers[i + 1])));

		v_I.push_back(new MatrixXd(MatrixXd::Zero(test_size_, layers[i + 1])));
		y_I.push_back(new MatrixXd(MatrixXd::Zero(test_size_, layers[i + 1])));

		y_d_.push_back(new MatrixXd(MatrixXd::Zero(layers[i + 1], batch_size_)));
		local_gradients_.push_back(new MatrixXd(MatrixXd::Zero(layers[i + 1], batch_size_)));

		weights_.push_back(new MatrixXd(MatrixXd::Random(layers[i], layers[i + 1])));
		bias_.push_back(new MatrixXd(MatrixXd::Random(1, layers[i + 1])));

		weights_past_update_.push_back(new MatrixXd(MatrixXd::Random(layers[i], layers[i + 1])));
		bias_past_update_.push_back(new MatrixXd(MatrixXd::Random(1, layers[i + 1])));
	}
	switch (activation_function) {
	case 1:  m_activation_function = &identity;	m_activation_functionDerivative = &identityD; break;
	case 2:  m_activation_function = &logistic;	m_activation_functionDerivative = &logisticD; break;
	case 3:  m_activation_function = &binary;	m_activation_functionDerivative = &binaryD;	 break;
	case 4:  m_activation_function = &relu;		m_activation_functionDerivative = &reluD;	 break;
	default: m_activation_function = &logistic;	m_activation_functionDerivative = &logisticD; break;
	}

	switch (cost_function) {
	case 1:  cost__function = &crossEntropy;	break;
	default: cost__function = &euclidian;		break;
	}

	printf("Initialization complete \n");
}
dnnJG::~dnnJG()
{
	delete batch_labels_;
	delete batch_input_;

	for (unsigned i = 0; i < nb_layers_ - 1; i++) {
		delete v_.back();
		delete y_.back();
		delete local_gradients_.back();
		delete weights_.back();
		delete bias_.back();

		v_.pop_back();
		y_.pop_back();
		local_gradients_.pop_back();
		weights_.pop_back();
		bias_.pop_back();
	}
}

void dnnJG::printStateE(unsigned currentEpoch, double accuracy, double temps)
{
	printf("Epoch #%d, Training error : %3.3f, Accuracy : %1.3f, Time : %1.0f ms \n", currentEpoch, cost_, accuracy, temps);
	std::cout << std::flush;
}
void dnnJG::printStateEB(unsigned currentEpoch, unsigned currentBatch, double accuracy, double temps)
{
	printf("Epoch #%d, Batch #%d, Training error : %3.3f, Accuracy : %1.3f, Time : %1.0f ms \n", currentEpoch, currentBatch, cost_, accuracy, temps);
	std::cout << std::flush;
}
void dnnJG::printStateEBCostOnly(unsigned currentEpoch, unsigned currentBatch, double temps)
{
	printf("Epoch #%d, Batch #%d, Training error : %3.3f,  Time : %1.0f ms \n", currentEpoch, currentBatch, cost_, temps);
	std::cout << std::flush;
}

int dnnJG::train(unsigned nbEpochs) {
	omp_set_num_threads(2);
	setNbThreads(2);
	input_size_ = unsigned(train_data_->rows());
	for (unsigned i = 0; i < nbEpochs; i++) {
		//set up permatation matrix
		//PermutationMatrix <Dynamic, Dynamic> perm(train_data_->size());
		//perm.setIdentity();
		//std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		//permutate the rows
		//*train_data_   = perm * (*train_data_);
		//*train_labels_ = perm * (*train_labels_);
		clock_t start = clock();
		for (unsigned j = 0; j < input_size_ / batch_size_; j++) {
			cost_ = 0;
			forward(train_data_, batch_input_, batch_size_, j * batch_size_);
			*batch_labels_ = train_labels_->block(j*batch_size_, 0, batch_size_, train_labels_->cols());
			backward(batch_labels_);
			cost__function(&cost_, batch_labels_, y_.back());
			updateWeight();
			/*accuracy_ = checkAccuracy(y_.back(), batch_labels_);
			printStateEB(i, j, accuracy_, clock() - start);
			printStateEBCostOnly(i, j, clock() - start);
			start = clock();
			printStateE(i * input_size_/batch_size_ + j, accuracy_);*/
		}
		inference();
		accuracy_ = checkAccuracy(y_I.back(), test_labels_);
		printStateE(i, accuracy_, clock() - start);
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
	for (unsigned k = 0; k < weights_.size(); k++) {
		*v_[k] = (k == 0 ? (*input) * (*weights_[k]) : (*y_[k - 1]) * (*weights_[k]));
		addBias(bias_[k], v_[k], batch_size_);
		*y_[k] = v_[k]->unaryExpr(m_activation_function);
		*y_d_[k] = (v_[k]->unaryExpr(m_activation_functionDerivative)).transpose();
	}
}
void dnnJG::backward(Eigen::MatrixXd *labels) {

	*local_gradients_.back() = (*y_.back() - *labels).transpose();
	for (size_t k = local_gradients_.size() - 1; k > 0; k--)
		*local_gradients_[k - 1] = ((*y_d_[k - 1]).array() * (*weights_[k] * (*local_gradients_[k])).array()).matrix();
}
void dnnJG::inference() {
	for (unsigned k = 0; k < weights_.size(); k++) {
		*v_I[k] = k == 0 ? (*test_data_) * (*weights_[k]) : (*y_I[k - 1]) * (*weights_[k]);
		addBias(bias_[k], v_I[k], test_size_);
		*y_I[k] = v_I[k]->unaryExpr(m_activation_function);
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
	switch (weight_update_function_) {
	case 0: naiveSGD();		break;
	case 1: momentumSGD();	break;
	default: naiveSGD();	break;
	}
}
void dnnJG::naiveSGD() {
	for (unsigned k = 0; k < weights_.size(); k++) {
		*weights_[k] += -learning_rate_ / batch_size_ * (*local_gradients_[k] * (k == 0 ? *batch_input_ : *y_[k - 1])).transpose();
		*bias_[k] += -learning_rate_ *((*local_gradients_[k]).transpose()).colwise().mean();
	}
}
void dnnJG::momentumSGD() {
	for (unsigned k = 0; k < weights_.size(); k++) {
		*weights_past_update_[k] = momentum_ * (*weights_past_update_[k]) + learning_rate_ / batch_size_ * (*local_gradients_[k] * (k == 0 ? *batch_input_ : *y_[k - 1])).transpose();
		*weights_[k] += -*weights_past_update_[k];

		*bias_past_update_[k] = momentum_ * (*bias_past_update_[k]) + learning_rate_ *((*local_gradients_[k]).transpose()).colwise().mean();
		*bias_[k] += -*bias_past_update_[k];
	}
}
