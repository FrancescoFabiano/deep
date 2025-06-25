#include <iostream>
#include <torch/torch.h>

int main() {
  try {
    // Create two tensors
    torch::Tensor a = torch::randn({2, 3});
    torch::Tensor b = torch::randn({3, 4});

    // Matrix multiplication
    torch::Tensor c = torch::mm(a, b);

    // Apply ReLU activation
    torch::Tensor d = torch::relu(c);

    // Second batach of tests

    std::vector<int64_t> edge_src = {0, 1, 2};
    std::vector<int64_t> edge_dst = {1, 2, 3};
    std::vector<int64_t> edge_labels = {10, 20, 30};

    const auto options = torch::TensorOptions().dtype(torch::kInt64);

    auto src_tensor = torch::from_blob(edge_src.data(), {3}, options);
    auto dst_tensor = torch::from_blob(edge_dst.data(), {3}, options);
    auto labels_tensor = torch::from_blob(edge_labels.data(), {3, 1}, options);

    auto edge_ids = torch::stack({src_tensor, dst_tensor});
    auto edge_attrs =
        labels_tensor.clone(); // Make sure to copy if you need lifetime safety

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] LibTorch test failed: " << e.what() << std::endl;
    return 1;
  }
}
