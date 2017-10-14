#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include "dnnJG.h"
#include "read_mnist.h"

using namespace std;
using namespace Eigen;

int main()
{
	//datasets
	int size_train_data = 3000,  size_test_data = 1000;
	std::string folder = "C:\\Users\\Jean-Gabriel Simard\\source\\repos\\mlp\\data\\";

	auto train_data   = jg_mnist::read_data(size_train_data, folder, true);
	auto train_labels = jg_mnist::read_labels(size_train_data,folder, true);
	auto test_data    = jg_mnist::read_data(size_test_data, folder, false);
	auto test_labels  = jg_mnist::read_labels(size_test_data, folder, false);

	//set up MLP structure
	std::vector<int> layers = {jg_mnist::INPUT_SIZE, 50, 50, jg_mnist::NB_CLASSES};
	
	dnnJG NN(layers, train_data, train_labels, test_data, test_labels, 10, -1, -1, 0.05, 0.01, 1);
	
	NN.train(10);

	system("pause");
}