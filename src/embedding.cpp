#include "include/embedding.h"

using Shaped = torch::Tensor;


FieldComponent::FieldComponent(int in_dim, int out_dim)
    : in_dim_(in_dim), out_dim_(out_dim) {}

void FieldComponent::set_in_dim(int in_dim) {
    if (in_dim <= 0) {
        throw std::invalid_argument("Input dimension should be greater than zero");
    }
    in_dim_ = in_dim;
}

int FieldComponent::get_out_dim() const {
    if (out_dim_ == -1) {
        throw std::logic_error("Output dimension has not been set");
    }
    return out_dim_;
}


Embedding::Embedding(int in_dim, int out_dim)
    : FieldComponent(in_dim, out_dim) {
        embedding = register_module("embedding", torch::nn::Embedding(in_dim, out_dim));
    }


Shaped Embedding::forward(Shaped in_tensor) {
    return embedding->forward(in_tensor);
}

torch::nn::Embedding Embedding::get_embedding()
{
    return embedding;
}

torch::Tensor Embedding::mean(int dim) {
    return embedding->weight.mean(dim);
}
