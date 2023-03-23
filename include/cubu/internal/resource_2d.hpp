#ifndef CUBU_RESOURCE_2D_HPP
#define CUBU_RESOURCE_2D_HPP

#ifdef __NVCC__
#include <vector>

namespace cubu::internal {
template<class T>
class resource_2d
{
public:
  /**
   * Type of the linear resource.
   */
  typedef T value_type;

  /**
   * Reference type of the linear resource.
   */
  typedef value_type& reference;

  /**
   * Const reference type of the linear resource.
   */
  typedef const value_type& const_reference;

  /**
   * Creates an empty resource object, cannot be used until reassigned.
   */
  explicit resource_2d();

  /**
   * Creates a new cuda texture object and allocates the device data.
   *
   * @param size Size of the texture data.
   * @param desc Description of the channel format for the texture data.
   */
  explicit resource_2d(int width,
                       int height,
                       cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>());

  /**
   * Creates a new cuda texture object and uploads the host data to the device.
   *
   * @param hostData Host data to upload to the device.
   * @param desc     Description of the channel format for the texture data.
   */
  explicit resource_2d(const std::vector<T>& hostData,
                       int width,
                       int height,
                       cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>());

  /**
   * Creates a new cuda texture object and uploads the host data to the device.
   *
   * @param data Host data to upload to the device.
   * @param size Size of the texture data.
   * @param desc Description of the channel format for the texture data.
   */
  explicit resource_2d(const T* data,
                       int width,
                       int height,
                       cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>());

  /**
   * Move constructor.
   *
   * @param o Other.
   */
  resource_2d(resource_2d&& o) noexcept;

  /**
   * Move assignment.
   *
   * @param o Other.
   *
   * @returns Self.
   */
  resource_2d& operator=(resource_2d&& o) noexcept;

  /* Delete the copy constructors */
  resource_2d(const resource_2d&) = delete;
  resource_2d& operator=(const resource_2d&) = delete;

  /**
   * Destroys texture and device buffer.
   */
  ~resource_2d();

  /**
   * Uploads host data to the device. Must be the same size as allocated.
   *
   * @param hostData Host data to upload to the device.
   */
  void copy_to_device(const std::vector<T>& hostData);

  /**
   * Uploads host data to the device. Must be the same size as allocated.
   *
   * @param hostData Host data to upload to the device.
   */
  void copy_to_device(const T* hostData);

  /**
   * Copies the device data to the host. The vector will automatically be
   * resized to fit the data.
   *
   * @param destData Destination for the data on the host.
   */
  void copy_to_host(std::vector<T>& destData);

  /**
   * Copies the device data to the host.
   *
   * @param destData Destination for the data on the host.
   */
  void copy_to_host(T* destData);

  /**
   * Getter for the device pointer.
   *
   * @returns Device pointer.
   */
  [[nodiscard]] T* dev_ptr() const;

  /**
   * Getter for the pitch of the resource.
   *
   * @returns Pitch of the underlying device array.
   */
  [[nodiscard]] size_t pitch() const;

  /**
   * Getter for the underlying texture object.
   *
   * @returns Cuda texture object for kernel use.
   */
  [[nodiscard]] cudaTextureObject_t tex() const;

  /**
   * Getter for the size of the texture.
   *
   * @returns Size of the texture resource.
   */
  [[nodiscard]] size_t size() const;

  /**
   * Getter for the width of the texture.
   *
   * @returns Width of the texture.
   */
  [[nodiscard]] int width() const;

  /**
   * Getter for the height of the texture.
   *
   * @returns Height of the texture.
   */
  [[nodiscard]] int height() const;

  /**
   * Checks if the resource is initialized and valid.
   *
   * @returns True if the resource is initialized.
   */
  explicit operator bool() const;

private:
  bool initialized_;

  T* devData_;
  size_t pitch_;
  int width_, height_;

  cudaTextureObject_t tex_;
  cudaResourceDesc texRes_;
  cudaTextureDesc texDescr_;
};
} // namespace cubu::internal

#include "cubu/impl/internal/resource_2d.ipp"
#else
namespace cubu::internal {
template<class T>
class resource_2d;
}
#endif

#endif // CUBU_RESOURCE_2D_HPP
