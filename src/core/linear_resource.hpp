#ifndef CUBU_LINEAR_RESOURCE_HPP
#define CUBU_LINEAR_RESOURCE_HPP

#ifdef __NVCC__
#include <stdexcept>
#include <vector>
#include "common/macros.hpp"

namespace cubu {
/**
 * Linear resource such as an array on the GPU.
 *
 * @tparam T Type of the array.
 */
template<class T>
class linear_resource
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
  explicit linear_resource();

  /**
   * Creates a new cuda texture object and allocates the device data.
   *
   * @param size Size of the texture data.
   * @param desc Description of the channel format for the texture data.
   */
  explicit linear_resource(
    size_t size,
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>());

  /**
   * Creates a new cuda texture object and uploads the host data to the device.
   *
   * @param hostData Host data to upload to the device.
   * @param desc     Description of the channel format for the texture data.
   */
  explicit linear_resource(
    const std::vector<T>& hostData,
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>());

  /**
   * Creates a new cuda texture object and uploads the host data to the device.
   *
   * @param data Host data to upload to the device.
   * @param size Size of the texture data.
   * @param desc Description of the channel format for the texture data.
   */
  explicit linear_resource(
    const T* data,
    size_t size,
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>());

  /**
   * Move constructor.
   *
   * @param o Other.
   */
  linear_resource(linear_resource&& o) noexcept;

  /**
   * Move assignment.
   *
   * @param o Other.
   *
   * @returns Self.
   */
  linear_resource& operator=(linear_resource&& o) noexcept;

  /* Delete the copy constructors */
  linear_resource(const linear_resource&) = delete;
  linear_resource& operator=(const linear_resource&) = delete;

  /**
   * Destroys texture and device buffer.
   */
  ~linear_resource();

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
   * Checks if the resource is initialized and valid.
   *
   * @returns True if the resource is initialized.
   */
  explicit operator bool() const;

private:
  bool initialized_;

  T* devData_;
  size_t size_;

  cudaTextureObject_t tex_;
  cudaResourceDesc texRes_;
  cudaTextureDesc texDescr_;
};
} // namespace cubu

#include "linear_resource.ipp"
#else
namespace cubu {
template<class T>
class linear_resource;
}
#endif

#endif // CUBU_LINEAR_RESOURCE_HPP
