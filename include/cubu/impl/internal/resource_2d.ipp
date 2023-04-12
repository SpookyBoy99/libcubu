#include "cubu/internal/validate.hpp"

namespace cubu::internal {
template<class T>
resource_2d<T>::resource_2d()
  : initialized_{ false }
  , width_{}
  , height_{}
  , tex_{}
  , texRes_{}
  , texDescr_{}
{
}

template<class T>
resource_2d<T>::resource_2d(int width, int height, cudaChannelFormatDesc desc)
  : initialized_{ true }
  , width_{ width }
  , height_{ height }
  , tex_{}
  , texRes_{}
  , texDescr_{}
{

  // *** Allocate memory for the device data
  validate cudaMallocPitch(
    (void**)&devData_, &pitch_, width * sizeof(T), height);

  // *** Configure the resource description
  texRes_.resType = cudaResourceTypePitch2D;
  texRes_.res.pitch2D.devPtr = devData_;
  texRes_.res.pitch2D.desc = desc;
  texRes_.res.pitch2D.width = width;
  texRes_.res.pitch2D.height = height;
  texRes_.res.pitch2D.pitchInBytes = pitch_;

  // *** Create the texture object
  validate cudaCreateTextureObject(&tex_, &texRes_, &texDescr_, nullptr);
}

template<class T>
resource_2d<T>::resource_2d(const std::vector<T>& hostData,
                            int width,
                            int height,
                            cudaChannelFormatDesc desc)
  : resource_2d(hostData.data(), width, height, desc)
{
}

template<class T>
resource_2d<T>::resource_2d(const T* data,
                            int width,
                            int height,
                            cudaChannelFormatDesc desc)
  : resource_2d(width, height, desc)
{
  copy_to_device(data);
}

template<class T>
resource_2d<T>::resource_2d(resource_2d&& o) noexcept
  : initialized_{ std::move(o.initialized_) }
  , devData_{ std::move(o.devData_) }
  , pitch_{ std::move(o.pitch_) }
  , width_{ std::move(o.width_) }
  , height_{ std::move(o.height_) }
  , tex_{ std::move(o.tex_) }
  , texRes_{ std::move(o.texRes_) }
  , texDescr_{ std::move(o.texDescr_) }
{
  // *** Set the initialized to false on the other object to prevent freeing of
  // resources
  o.initialized_ = false;
}

template<class T>
resource_2d<T>&
resource_2d<T>::operator=(resource_2d&& o) noexcept
{
  // *** Swap all values (this way unused resources get freed)
  std::swap(initialized_, o.initialized_);
  std::swap(devData_, o.devData_);
  std::swap(pitch_, o.pitch_);
  std::swap(width_, o.width_);
  std::swap(height_, o.height_);
  std::swap(tex_, o.tex_);
  std::swap(texRes_, o.texRes_);
  std::swap(texDescr_, o.texDescr_);

  // *** Return this
  return *this;
}

template<class T>
resource_2d<T>::~resource_2d()
{
  if (!initialized_) {
    return;
  }

  validate cudaDestroyTextureObject(tex_);
  validate cudaFree(devData_);
}

template<class T>
void
resource_2d<T>::copy_to_device(const std::vector<T>& hostData)
{
  if (hostData.size() != width_ * height_) {
    throw std::out_of_range("Size of host data not equal to device data");
  }

  copy_to_device(hostData.data());
}

template<class T>
void
resource_2d<T>::copy_to_device(const T* hostData)
{
  validate cudaMemcpy2D(devData_,
                         pitch_,
                         hostData,
                         width_ * sizeof(T),
                         width_ * sizeof(T),
                         height_,
                         cudaMemcpyHostToDevice);
}

template<class T>
void
resource_2d<T>::copy_to_host(std::vector<T>& destData)
{
  // *** Resize the vector to fit all the data
  destData.resize(width_ * height_);

  // *** Copy the data
  copy_to_host(destData.data());
}

template<class T>
void
resource_2d<T>::copy_to_host(T* destData)
{
  validate cudaMemcpy2D(destData,
                         width_ * sizeof(T),
                         devData_,
                         pitch_,
                         width_ * sizeof(T),
                         height_,
                         cudaMemcpyDeviceToHost);
}

template<class T>
T*
resource_2d<T>::dev_ptr() const
{
  return devData_;
}

template<class T>
size_t
resource_2d<T>::pitch() const
{
  return pitch_ / sizeof(T);
}

template<class T>
cudaTextureObject_t
resource_2d<T>::tex() const
{
  return tex_;
}

template<class T>
size_t
resource_2d<T>::size() const
{
  return width_ * height_;
}

template<class T>
int
resource_2d<T>::width() const
{
  return width_;
}

template<class T>
int
resource_2d<T>::height() const
{
  return height_;
}

template<class T>
resource_2d<T>::operator bool() const
{
  return initialized_;
}
} // namespace cubu::internal