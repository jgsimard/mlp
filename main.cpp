#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include "dnnJG.h"
#include "read_mnist.h"

using namespace std;
using namespace Eigen;


int main()
{
	//datasets
	int size_train_data = 200,  size_test_data = 100;
	std::string folder = "C:\\Users\\Jean-Gabriel Simard\\source\\repos\\mlp\\data\\";
	try {
		auto train_data   = read_data_MNIST(size_train_data, folder, true);
		auto train_labels = read_labels_MNIST(size_train_data,folder, true);
		auto test_data    = read_data_MNIST(size_test_data, folder, false);
		auto test_labels  = read_labels_MNIST(size_test_data, folder, false);
	}
	catch (string e) {
		cout << "ERROR : " << e << " is not read proprely \n";
		system("pause");
		exit(0);
	}

	//set up MLP structure
	std::vector<int> layers = {INPUT_SIZE, 10, NB_CLASSES};
	
	//dnnJG NN(layers, train_data, train_labels, test_data, test_labels, 100, -1, -1, 0.05, 0.01, 1);
	//NN.train(10);

	system("pause");
}