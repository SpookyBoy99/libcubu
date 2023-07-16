#include "cubu/internal/validate.hpp"

namespace cubu::internal {
template<class T>
linear_resource<T>::linear_resource()
  : initialized_{ false }
  , size_{}
  , tex_{}
  , texRes_{}
  , texDescr_{}
{
}

template<class T>
linear_resource<T>::linear_resource(size_t size, cudaChannelFormatDesc desc)
  : initialized_{ true }
  , size_{ size }
  , tex_{}
  , texRes_{}
  , texDescr_{}
{
  // *** Allocate memory for the device data
  validate cudaMalloc((void**)&devData_, size * sizeof(T));

  // *** Configure the resource description
  texRes_.resType = cudaResourceTypeLinear;
  texRes_.res.linear.devPtr = devData_;
  texRes_.res.linear.desc = desc;
  texRes_.res.linear.sizeInBytes = size * sizeof(T);

  // *** Create the texture object
  validate cudaCreateTextureObject(&tex_, &texRes_, &texDescr_, nullptr);
}

template<class T>
linear_resource<T>::linear_resource(const std::vector<T>& hostData,
                                    cudaChannelFormatDesc desc)
  : linear_resource(hostData.data(), hostData.size(), desc)
{
}

template<class T>
linear_resource<T>::linear_resource(const T* data,
                                    size_t size,
                                    cudaChannelFormatDesc desc)
  : linear_resource(size, desc)
{
  copy_to_device(data);
}

template<class T>
linear_resource<T>::linear_resource(linear_resource&& o) noexcept
  : initialized_{ std::move(o.initialized_) }
  , devData_{ std::move(o.devData_) }
  , size_{ std::move(o.size_) }
  , tex_{ std::move(o.tex_) }
  , texRes_{ std::move(o.texRes_) }
  , texDescr_{ std::move(o.texDescr_) }
{
  // *** Set the initialized to false on the other object to prevent freeing of
  // resources
  o.initialized_ = false;
}

template<class T>
linear_resource<T>&
linear_resource<T>::operator=(linear_resource&& o) noexcept
{
  // *** Swap all values (this way unused resources get freed)
  std::swap(initialized_, o.initialized_);
  std::swap(devData_, o.devData_);
  std::swap(size_, o.size_);
  std::swap(tex_, o.tex_);
  std::swap(texRes_, o.texRes_);
  std::swap(texDescr_, o.texDescr_);

  // *** Return this
  return *this;
}

template<class T>
linear_resource<T>::~linear_resource()
{
  if (!initialized_) {
    return;
  }

  validate cudaDestroyTextureObject(tex_);
  validate cudaFree(devData_);
}

template<class T>
void
linear_resource<T>::copy_to_device(const std::vector<T>& hostData) const
{
  if (hostData.size() != size_) {
    throw std::out_of_range("Size of host data not equal to device data");
  }

  copy_to_device(hostData.data());
}

template<class T>
void
linear_resource<T>::copy_to_device(const T* hostData) const
{
  validate cudaMemcpy(
    devData_, hostData, size_ * sizeof(T), cudaMemcpyHostToDevice);
}

template<class T>
void
linear_resource<T>::copy_to_host(std::vector<T>& destData) const
{
  // *** Resize the vector to fit all the data
  destData.resize(size_);

  // *** Copy the data
  copy_to_host(destData.data());
}

template<class T>
void
linear_resource<T>::copy_to_host(T* destData) const
{
  validate cudaMemcpy(
    destData, devData_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
}

template<class T>
T*
linear_resource<T>::dev_ptr() const
{
  return devData_;
}

template<class T>
cudaTextureObject_t
linear_resource<T>::tex() const
{
  return tex_;
}

template<class T>
size_t
linear_resource<T>::size() const
{
  return size_;
}

template<class T>
linear_resource<T>::operator bool() const
{
  return initialized_;
}
} // namespace cubu::internal