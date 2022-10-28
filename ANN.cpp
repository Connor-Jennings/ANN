//
//  ANN.cpp
//  NeuralNetwork
//
//  Created by Connor Jennings on 10/20/22.
//

#include "ANN.hpp"

ANN::ANN()
{
    seedNetwork();
    
    input = std::vector<double>(num_input_nodes); //196 nodes
    hiddenLayer = std::vector<double>(num_hiddedLayer_nodes); //103 nodes
    output = std::vector<double>(num_output_nodes); //10 nodes (0-9)
}

void ANN::seedNetwork()
{
    //srand(time(NULL)); // uncomment for actual random seeding
    // seed weight_1
    // seed bias_1
    for(int x = 0; x < num_hiddedLayer_nodes; ++x)
    {
        weight_1.push_back(std::vector<double>());
        for(int y = 0; y < num_input_nodes; ++y)
        {
            weight_1[x].push_back(r_num_zero_one());
        }
        bias_1.push_back(r_num_zero_one());
    }
    
    // seed weight_2
    // seed bias_2
    for(int x = 0; x < num_output_nodes; ++x)
    {
        weight_2.push_back(std::vector<double>());
        for(int y = 0; y < num_hiddedLayer_nodes; ++y)
        {
            weight_2[x].push_back(r_num_zero_one());
        }
        bias_2.push_back(r_num_zero_one());
    }
}

//calc hidden layer -> calc output -> back propagate -> reset layers
void ANN::trainImage(std::vector<double> input_, short int actual)
{
    input = input_;
    calcHiddenLayer();
    calcOutput();
    backPropagate(actual);
    reset();
}

//calc hidden layer -> calc output -> print result -> reset layers
void ANN::testImage(std::vector<double> input_, short int actual)
{
    input = input_;
    calcHiddenLayer();
    calcOutput();
    output = softmax(output);
    print_result(output, true);
    reset();
}


// for each node sum(inputxweight_1) + bias -> ReLU -> set hiddenLayer
void ANN::calcHiddenLayer()
{
    for(int node = 0; node < num_hiddedLayer_nodes; ++node)
    {
        for(int input_layer = 0; input_layer < num_input_nodes; ++input_layer)
        {
            hiddenLayer[node] += weight_1[node][input_layer] * input[input_layer];
        }
        hiddenLayer[node] += bias_1[node];
        hiddenLayer[node] = std::max(0.0, hiddenLayer[node]); // ReLU
    }
}

// for each node sum(hiddenLayerxweight_2) + bias -> ReLU -> softmax -> set output
void ANN::calcOutput()
{
    for(int node = 0; node < num_output_nodes; ++node)
    {
        for(int hidden_layer = 0; hidden_layer < num_hiddedLayer_nodes; ++hidden_layer)
        {
            output[node] += weight_2[node][hidden_layer] * input[hidden_layer];
        }
        output[node] += bias_2[node];
        output[node] = std::max(0.0, hiddenLayer[node]); // ReLU
    }
}

void ANN::reset()
{
    input = std::vector<double>(num_input_nodes);
    hiddenLayer = std::vector<double>(num_hiddedLayer_nodes);
    output = std::vector<double>(num_output_nodes);
}

//update weights and bias based on cost
void ANN::backPropagate(short int actual)
{
    double cost = calcCost(actual);
    std::cout << "cost : " << cost << std::endl;
}

double ANN::calcCost(short int actual)
{
    //std::vector<double> correct(num_output_nodes,0);
    //correct[actual] = 1;
    double cost = 0;
    for(int i = 0; i < output.size(); ++i)
    {
        if(i == actual)
        {
            double val = output[i] - 1;
            cost += (val * val);
        }else
        {
            cost += (output[i] * output[i]);
        }
    }
    return cost;
}

std::vector<double> ReLU_1D(std::vector<double> nodes)
{
    for(int i = 0; i < nodes.size(); ++i)
    {
       nodes[i] = std::max(0.0, nodes[i]);
    }
    return nodes;
}

double r_num(int low, int high)
{
    if(low == high) return high;
    double num = rand() % (high+1) + low;
    return num/10;
}

double r_num_zero_one()
{
    double num = rand() % (11);
    return num/10;
}

std::vector<double> softmax(std::vector<double> output)
{
    if(output.size() == 1)
    {
        output[0] = 1.0;
        return output;
    }
    
    double max_val  = std::max(output[0], output[1]);
    for(int i = 2; i < output.size()-1; i+=1)
    {
        max_val = std::max(max_val, output[i]);
    }
    
    int e_sum = 0;
    for(int i = 0; i < output.size(); ++i)
    {
        output[i] /= 1000;
        e_sum += exp(output[i]);
    }
    for(int i = 0; i < output.size(); ++i)
    {
        output[i] = exp(output[i]) / e_sum;
        
    }
    return output;
}

void print_result(std::vector<double> output, bool verbose)
{
    if(verbose) std::cout << " --Results--" << std::endl;
    double winner_val = 0;
    int winner = 0;
    for(int i = 0; i < output.size(); ++i)
    {
        if(verbose)
        {
            std::cout << i << " : " << output[i]*100 << " %" << std::endl;
        }
        if(winner_val < output[i]){
            winner_val = output[i];
            winner = i;
        }
    }
    std::cout << "\nWinner is " << winner << std::endl;
}
