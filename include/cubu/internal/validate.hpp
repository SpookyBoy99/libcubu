#ifndef CUBU_VALIDATE_HPP
#define CUBU_VALIDATE_HPP

#include <stdexcept>

#define validate (cubu::internal::_gpu_validate(__FILE__, __LINE__)) =

namespace cubu::internal {
class _gpu_validate
{
public:
  _gpu_validate(const char* file, int line)
    : file_(file)
    , line_(line)
  {
  }

  _gpu_validate& operator=(cudaError_t err)
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

#endif // CUBU_VALIDATE_HPP
