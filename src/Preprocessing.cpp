//
//  Preprocessing.cpp
//  NeuralNetwork
//
//  Created by Connor Jennings on 10/28/22.
//

#include "Preprocessing.hpp"


std::vector<std::vector<double>> init_size(int rows, int cols)
{
    return std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));
}

int pix_value_at(int row, int col, const cv::Mat image)
{
    if(row < 0  || col < 0) return 0;
    else if(row >= image.rows || col >= image.cols) return 0;
    return (int)image.at<uchar>(row,col);
}

double conv1_value_at(int row, int col, const cv::Mat image, std::vector<std::vector<double>> kernel)
{
    int top_left = pix_value_at(row-1, col-1, image);
    int top_mid = pix_value_at(row-1, col, image);
    int top_right = pix_value_at(row-1, col+1, image);
    int mid_left = pix_value_at(row,col-1, image);
    int mid_mid = pix_value_at(row,col, image);
    int mid_right = pix_value_at(row, col+1, image);
    int bottom_left = pix_value_at(row+1, col-1, image);
    int bottom_mid = pix_value_at(row+1, col, image);
    int bottom_right = pix_value_at(row+1, col+1, image);
    
    double value = top_left*kernel[0][0] + top_mid*kernel[0][1] + top_right*kernel[0][2];
    value += mid_left*kernel[1][0] + mid_mid*kernel[1][1] + mid_right*kernel[1][2];
    value += bottom_left*kernel[2][0] + bottom_mid*kernel[2][1] + bottom_right*kernel[2][2];
    return value;
}

std::vector<std::vector<double>> convolution1(const cv::Mat image)
{
    std::vector<std::vector<double>> conv_image = init_size(image.rows, image.cols);
    std::vector<std::vector<double>> kernel ={
        {-1, -1, -1},
        {-1, 8, -1},
        {-1, -1, -1}};
    
    for(int row = 0; row < image.rows; ++row)
    {
        for(int col = 0; col < image.cols; ++col)
        {
            conv_image[row][col] = conv1_value_at(row, col, image, kernel);
        }
    }
    return conv_image;
}

double vec_value_at(int row, int col, std::vector<std::vector<double>> image)
{
    if(row >= image.size() || col >= image[0].size()) return 0.0;
    return image[row][col];
}

double max_of_square(int row, int col, std::vector<std::vector<double>> image)
{
    double max_val = std::max(vec_value_at(row,col,image),vec_value_at(row,col+1,image));
    max_val = std::max(max_val, vec_value_at(row+1,col,image));
    max_val = std::max(max_val, vec_value_at(row+1,col+1,image));
    return max_val;
}

std::vector<std::vector<double>> pool1(std::vector<std::vector<double>> image)
{
    // 2x2 kernel wiht a step of 2 taking the max of the four values
    int step = 2;
    auto pool = init_size((int)image.size()/step, (int)image[0].size()/step);
    int row_step = 0;
    for(int row = 0; row < pool.size(); ++row, row_step+=step)
    {
        int col_step = 0;
        for(int col = 0; col < pool[0].size(); ++col, col_step+=step)
        {
            pool[row][col] = max_of_square(row_step, col_step, image);
        }
    }
    return pool;
}

std::vector<std::vector<double>> ReLU_2D(std::vector<std::vector<double>> image)
{
    for(int i = 0; i < image.size(); ++i)
    {
        for(int j = 0; j < image[0].size(); ++j)
        {
            image[i][j] = std::max(0.0, image[i][j]);
        }
    }
    return image;
}

std::vector<double> flatten(std::vector<std::vector<double>> image)
{
    std::vector<double> flat_image;
    for(auto vec : image)
    {
        for(auto number : vec)
        {
            flat_image.push_back(number);
        }
    }
    return flat_image;
}



std::vector<double> processImageAt(std::string file_path)
{
    //std::string file_path = "/Users/connorjennings/Code/KNNproject/data/mnist_png/mnist_png/training/5/img00788.png";
    cv::Mat image;
    
    try {
        image = cv::imread(file_path, 0); //read in grayscale
        if(image.empty()){
            std::string error = "Could not read the image...";
            throw error;
        }
        
        auto vec_2D = convolution1(image);
        vec_2D = ReLU_2D(vec_2D);
        vec_2D = pool1(vec_2D);
        auto vec_1D = flatten(vec_2D);
        return vec_1D;
    }
    catch(std::string x){
        std::cout << "Error [ " << x <<  " ]" <<std::endl;
        return std::vector<double>(0);
    }
    catch (...) {
        std::cout << "Something went wrong... \n " << std::endl;
        return std::vector<double>(0);
    }
    return std::vector<double>(0);
}
