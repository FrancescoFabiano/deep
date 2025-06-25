#include <iostream>
#include <torch/torch.h>

int main() {
  try {
    torch::Tensor tensor = torch::rand({2, 3});
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Libtorch C++ test failed: " << e.what() << std::endl;
    return 1;
  }
}
