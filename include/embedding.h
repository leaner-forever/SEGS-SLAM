// We reference the code in https://github.com/nerfstudio-project/nerfstudio/blob/a8e6f8fa3fd6c0ad2f3e681dcf1519e74ad2230f/nerfstudio/field_components/embedding.py
// Thanks to their great work!

#include <torch/torch.h>
#include <stdexcept>

using Shaped = torch::Tensor;

class FieldComponent : public torch::nn::Module {
public:
    FieldComponent(int in_dim = -1, int out_dim = -1);
    virtual ~FieldComponent() = default;

    void set_in_dim(int in_dim);
    int get_out_dim() const;

    virtual Shaped forward(Shaped in_tensor) = 0;

protected:
    int in_dim_;
    int out_dim_;
};

class Embedding : public FieldComponent {
public:
    Embedding(int in_dim = 1, int out_dim = 1);

    Shaped forward(Shaped in_tensor) override;

    torch::Tensor mean(int dim = 0);

    torch::nn::Embedding get_embedding();

private:
    torch::nn::Embedding embedding{nullptr};
};