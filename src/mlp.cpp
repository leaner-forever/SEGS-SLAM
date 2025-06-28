/*
 * reference https://pytorch.org/cppdocs/frontend.html
 */

#pragma once
#include <include/mlp.h>


MLP_feature_bank::MLP_feature_bank(int in_features, int mid_feature_1, int mid_feature_2, int out_features){
    fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(in_features, mid_feature_1)));
    fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(mid_feature_1, mid_feature_2)));

    out_features_ = out_features;
}

torch::Tensor MLP_feature_bank::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::log_softmax(fc2->forward(x), /*dim=*/out_features_);

    return x;
}


MLP_opacity::MLP_opacity(int in_features, int mid_feature_1, int mid_feature_2){
    fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(in_features, mid_feature_1)));
    fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(mid_feature_1, mid_feature_2)));
}

torch::Tensor MLP_opacity::forward(torch::Tensor x) {

    x = torch::relu(fc1->forward(x));
    x = torch::tanh(fc2->forward(x));

    return x;
}



MLP_cov::MLP_cov(int in_features, int mid_feature_1, int mid_feature_2){

    fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(in_features, mid_feature_1)));
    fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(mid_feature_1, mid_feature_2)));

}

torch::Tensor MLP_cov::forward(torch::Tensor x) {
    
    x = torch::relu(fc1->forward(x));
    x = fc2->forward(x);

    return x;
}


MLP_color::MLP_color(int in_features, int mid_feature_1, int mid_feature_2){

    fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(in_features, mid_feature_1)));
    fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(mid_feature_1, mid_feature_2)));

}

torch::Tensor MLP_color::forward(torch::Tensor x) {
    
    x = torch::relu(fc1->forward(x));
    x = torch::sigmoid(fc2->forward(x));

    return x;
}