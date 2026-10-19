#include "../include/lib.c"

storage *storage_align_allocate(size_t n_elements, device_type device);
float *storage_cpu_alloc(size_t size);						      // allocating CPU storage
void storage_cpu_dealloc(void *s);						      // deallocating CPU storage
float *storage_gpu_alloc(size_t size);						      // allocting GPU storage
float *storage_gpu_dealloc(size_t size);					      // deallocting GPU storage
void storage_no_op_dealloc(void *s);						      // if the data is not owned by the storage itself
void storage_copy_cpu_to_gpu(storage *dst, const storage *src);			      // CPU to GPU
void storage_copy_gpu_to_cpu(storage *dst, const storage *src);			      // GPU to CPU
storage *storage_wrap(void *ptr, size_t n_elements, size_t size, device_type device); // the call is responsible for the memory
storage *storage_retain(storage *s);
void storage_release(storage *s);
