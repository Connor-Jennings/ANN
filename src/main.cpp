//
//  main.cpp
//  NeuralNetwork
//
//  Created by Connor Jennings on 10/18/22.
//

#include <iostream>
#include <vector>
#include "Preprocessing.hpp"
#include "ANN.hpp"



int main() {
    
    //set parameters
    const int number_images_to_train = 1;
    const int number_images_to_test = 0;
    ANN network; //initialize the network, this will automatically seed the network with random weights
    
    
    //train the network
    for(int iteration = 0; iteration < number_images_to_train; ++iteration)
    {
        //eventually willl generate file path from other parameters
        std::string file_path = "/Users/connorjennings/Code/KNNproject/data/mnist_png/mnist_png/training/5/img00788.png";
        auto input = processImageAt(file_path); //preprocess image for lighter computations
        
        network.trainImage(input, 5); // will compute the cost function, still working on backpropagation
    }
   
    //test the network
    for(int iteration = 0; iteration < number_images_to_test; ++iteration)
    {
        std::string file_path = "/Users/connorjennings/Code/KNNproject/data/mnist_png/mnist_png/testing/5/img00788.png";
        auto input = processImageAt(file_path); //preprocess image for lighter computations
        
        network.testImage(input, 5); //this will print out results as they are tested
    }
    
    return 0;
}
