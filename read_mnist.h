#pragma once

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <string>

const int IMAGE_SIZE = 28;
const int INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE;
const int NB_CLASSES = 10;

static int reverse_int(int& i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
auto read_data_MNIST(const int& size_data, std::string folder, bool train_data)
{
	std::string file_name = folder + (train_data ? "train-images.idx3-ubyte" : "t10k-images.idx3-ubyte");
	std::ifstream file = std::ifstream(file_name, std::ios::binary);
	if (file.is_open()){
		//numbers in header
		int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));

		//Check if little of bid endian is used
		volatile uint32_t i = 0x01234567;
		if ((*((uint8_t*)(&i))) == 0x67) { 		// return 0 for big endian, 1 for little endian.
			magic_number = reverse_int(magic_number);
			number_of_images = reverse_int(number_of_images);
			n_rows = reverse_int(n_rows);
			n_cols = reverse_int(n_cols);
		};

		std::cout << file_name << std::endl;
		printf("magic_number: %d, number_of_images : %d, n_rows: %d, n_cols : %d \n\n", magic_number, number_of_images, n_rows, n_cols);
		
		//read data
		auto data = std::make_shared<Eigen::MatrixXd>(size_data, INPUT_SIZE);
		for (int i = 0; i < size_data; i++) {
			for (int r = 0; r < n_rows; r++) {
				for (int c = 0; c < n_cols; c++) {
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					(*data)(i, (n_rows * r) + c) = (double)temp;
				}
			}
		}
		return data;
	}
	else { throw file_name; }
}
auto read_labels_MNIST(const int& size_data, std::string folder, bool train_labels)
{
	std::string file_name = folder + (train_labels ? "train-labels.idx1-ubyte" : "t10k-labels.idx1-ubyte");
	std::ifstream file = std::ifstream(file_name, std::ios::binary);
	if (file.is_open()){
		//numbers in header
		int magic_number = 0, number_of_items = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_items, sizeof(number_of_items));

		//Check if little of bid endian is used
		volatile uint32_t i = 0x01234567;
		if ((*((uint8_t*)(&i))) == 0x67) { 		// return 0 for big endian, 1 for little endian.
		magic_number = reverse_int(magic_number);
		number_of_items = reverse_int(number_of_items);
		}

		std::cout << file_name << std::endl;
		printf("magic_number: %d, number_of_items : %d \n\n", magic_number, number_of_items);

		//read labels
		auto labels = std::shared_ptr<Eigen::MatrixXd>(new Eigen::MatrixXd(Eigen::MatrixXd::Zero(size_data, NB_CLASSES)));
		for (int i = 0; i < size_data; i++) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			(*labels)(i, temp) = (double)1.0;
		}
		return labels;
	}
	else { throw file_name; }
}
