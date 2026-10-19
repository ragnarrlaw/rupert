#ifndef LIB_H
#define LIB_H

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

// random number generation
typedef struct rng_state {
	uint64_t state;
} rng_state;

rng_state *rng_from_seed(uint64_t seed);
rng_state *rng_from_entropy();
void rng_free(rng_state *ptr);
/* 64-bit raw output from SplitMix64 (advances state) */
uint64_t rng_next_u64(rng_state *r);
/* 32-bit output */
uint32_t rng_next_u32(rng_state *r);
/* uniform float32 in [0,1) */
float rng_next_f32(rng_state *r);
/* uniform integer in [0, n) unbiased using rejection sampling. assumes 1 <= n <= 2^32. */
uint32_t rng_uniform_int(rng_state *r, uint32_t n);
/* Box-Muller normal sampler (mean, std) returning float32 -> to get curves from flat lines. Caches the second sample for next call. */
void rng_normal_f32(rng_state *r, float mean, float stddev, float *out_1, float *out_2);
/* block allocation -> fill contiguous float32 buffer with uniform in [a,b). */
void rng_linear_block_fill_uniform_f32(rng_state *r, float *dst, size_t n, float a, float b);
/* block allocation -> fill contiguous float32 buffer with normal(mean,std) */
void rng_linear_block_fill_normal_f32(rng_state *r, float *dst, size_t n, float mean, float stddev);
/* shuffle in position -> fisher-yates shuffle for integer array (in-place) */
void rng_shuffle_float_array(rng_state *r, float *arr, size_t n);

#define MAX_DIMS 8

typedef enum device_type { CPU, GPU } device_type;

/* flags related to storage struct */
typedef uint8_t storage_flags;

#define STORAGE_DATA_OWNED (1u << 0)
#define STORAGE_DATA_BORROWED (1u << 1)

typedef float *(*storage_alloc_fn)(size_t size);
typedef void (*storage_dealloc_fn)(void *ptr);

typedef struct storage {
	float *data;
	size_t n_elements; // number of elements
	size_t size;	   // total allocation size
	atomic_int refs;
	device_type stored_on; // CPU or GPU
	storage_flags flags;
	storage_alloc_fn allocator;
	storage_dealloc_fn deallocator;
} storage;

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

typedef uint8_t matrix_flags;

#define MAT_CONTIGUOUS (1u << 0)
#define MAT_IS_VIEW (1u << 1)

typedef struct matrix {
	size_t ndims;
	storage *data;	 // reference counted storage
	float *data_ptr; // pointer to the actual starting point of the data
	size_t dims[MAX_DIMS];
	size_t strides[MAX_DIMS];
	size_t nelems;
	device_type device;
	matrix_flags flags;
} matrix;

matrix *matrix_create(size_t ndim, const size_t *dims, device_type device);
matrix *matrix_fill(size_t ndim, const size_t *dims, device_type, float *fill_with);
matrix *matrix_zero(size_t ndim, const size_t *dims, device_type device);
matrix *matrix_ones(size_t ndim, const size_t *dims, device_type device);
matrix *matrix_eye(size_t ndim, const size_t *dims, device_type device); // last two dimensions must be the same
matrix *matrix_rand_uniform(rng_state *rs, size_t ndims, const size_t *dims, float min, float max);
matrix *matrix_rand_normal(rng_state *rs, size_t ndims, const size_t *dims, float mean, float stddev);

matrix *matrix_rand_kaiming_uniform(rng_state *rs, int ndims, const size_t *dims, device_type device);
matrix *matrix_rand_kaiming_normal(rng_state *rs, int ndims, const size_t *dims, device_type device);
matrix *matrix_rand_glorot_uniform(rng_state *rs, int ndims, const size_t *dims, device_type device);
matrix *matrix_rand_glorot_normal(rng_state *rs, int ndims, const size_t *dims, device_type device);

matrix *matrix_wrap(void *data, int ndim, const size_t *dims, device_type device);
matrix *matrix_clone(const matrix *src); // deep copy
matrix *matrix_copy(const matrix *src);	 // shallow copy
void matrix_copy_to(const matrix *src, matrix *dst);
void matrix_free(matrix *m);

#endif // LIB_H
