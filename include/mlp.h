/*
 * reference https://pytorch.org/cppdocs/frontend.html
 */

#pragma once
#include <torch/torch.h>


class MLP_feature_bank: public torch::nn::Module{
public:
    MLP_feature_bank(int in_features, int mid_feature_1, int mid_feature_2, int out_features);
    torch::Tensor forward(torch::Tensor x);


private:
    int out_features_;
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};



class MLP_opacity: public torch::nn::Module{
public:
    MLP_opacity(int in_features, int mid_feature_1, int mid_feature_2);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};



class MLP_cov: public torch::nn::Module{
public:
    MLP_cov(int in_features, int mid_feature_1, int mid_feature_2);
    torch::Tensor forward(torch::Tensor x);


private:

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};


class MLP_color: public torch::nn::Module{
public:
    MLP_color(int in_features, int mid_feature_1, int mid_feature_2);
    torch::Tensor forward(torch::Tensor x);


private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};