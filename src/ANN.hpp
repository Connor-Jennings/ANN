//
//  ANN.hpp
//  NeuralNetwork
//
//  Created by Connor Jennings on 10/20/22.
//

#ifndef ANN_hpp
#define ANN_hpp

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <math.h>       /* exp */

#include <iostream>

std::vector<double> ReLU_1D(std::vector<double>);
std::vector<double> softmax(std::vector<double>);
void print_result(std::vector<double>, bool);
double r_num(int, int); // [low, high]
double r_num_zero_one();


class ANN
{
    const int low = 0; // for seeding the network
    const int high = 10; // ""
    const int num_input_nodes = 196; // ""
    const int num_hiddedLayer_nodes = 103; // ""
    const int num_output_nodes = 10; // ""
    
    
    std::vector<double> input; //196 nodes
    std::vector<double> hiddenLayer; //103 nodes
    std::vector<double> output; //10 nodes (0-9)
    std::vector<std::vector<double>> weight_1; //196x103 weights [103][196]
    std::vector<std::vector<double>> weight_2; //103x10 weights [10][103]
    std::vector<double> bias_1; //103 bias values
    std::vector<double> bias_2; //10 bias values
    
    void calcHiddenLayer(); // for each node sum(inputxweight_1) + bias -> ReLU -> set hiddenLayer
    void calcOutput(); // for each node sum(hiddenLayerxweight_2) + bias -> ReLU -> softmax -> set output
    void backPropagate(short int); //update weights and bias based on cost
    void seedNetwork(); //randomly initalize weights and bias
    void reset();
    
    double calcCost(short int);
    
public:
    ANN();
    
    void trainImage(std::vector<double>, short int);//calc hidden layer -> calc output -> back propagate -> reset
    void testImage(std::vector<double>, short int);
    void test(){
        for (int i =0; i< 11; ++i)
        {
            std::cout << r_num_zero_one() << std::endl;
        }
    }
};




#endif /* ANN_hpp */
