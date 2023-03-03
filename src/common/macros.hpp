#ifndef CUBU_MACROS_HPP
#define CUBU_MACROS_HPP

// todo: Replace this with a function call that throws an exception to prevent
//  the server from crashing upon errors and destructors to be called
#ifdef NDEBUG
#define gpuAssert(call) call
#else
#include <cstdio>

#define gpuAssert(call)                                                        \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "CUDA error at %s %d: %s\n",                                     \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

#endif // CUBU_MACROS_HPP
