#ifndef CUBU_GPU_CHECK_HPP
#define CUBU_GPU_CHECK_HPP

#include <stdexcept>

#define gpu_check (cubu::internal::_gpu_check(__FILE__, __LINE__)) =

namespace cubu::internal {
class _gpu_check
{
public:
  _gpu_check(const char* file, int line)
    : file_(file)
    , line_(line)
  {
  }

  _gpu_check& operator=(cudaError_t err)
  {
    if (err != cudaSuccess) {
      char str[256];
      sprintf(str,
              "CUDA error at %s:%d: %s\n",
              file_,
              line_,
              cudaGetErrorString(err));
      throw std::runtime_error(str);
    }

    return *this;
  }

private:
  const char* file_;
  int line_;
};
} // namespace cubu::internal

#endif // CUBU_GPU_CHECK_HPP
