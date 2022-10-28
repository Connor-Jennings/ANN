//
//  Preprocessing.hpp
//  NeuralNetwork
//
//  Created by Connor Jennings on 10/28/22.
//

#ifndef Preprocessing_hpp
#define Preprocessing_hpp

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>


int pix_value_at(int, int, const cv::Mat);
double conv1_value_at(int, int, const cv::Mat, std::vector<std::vector<double>>);
std::vector<std::vector<double>> convolution1(cv::Mat);
std::vector<std::vector<double>> pool1(std::vector<std::vector<double>>);
std::vector<std::vector<double>> ReLU_2D(std::vector<std::vector<double>>);
std::vector<double> flatten(std::vector<std::vector<double>>);
std::vector<std::vector<double>> init_size(int, int);
double max_of_square(int, int, std::vector<std::vector<double>>);
double vec_value_at(int, int, std::vector<std::vector<double>>);


std::vector<double> processImageAt(std::string);


#endif /* Preprocessing_hpp */
