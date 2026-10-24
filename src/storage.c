#include "../include/lib.c"
#include <assert.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>

#define CPU_ALLOCATION_SIZE(n_elements) (n_elements + (VECTOR_ALIGNMENT - 1) / VECTOR_ALIGNMENT) * VECTOR_ALIGNMENT;

storage *storage_allocate(size_t n_elements, device_type device) {
	float *data = NULL;
	size_t size = 0;
	storage_alloc_fn alloc_fn = NULL;
	storage_dealloc_fn dealloc_fn = NULL;
	switch (device) {
	case CPU: {
		data = storage_cpu_alloc(n_elements);
		size = CPU_ALLOCATION_SIZE(n_elements);
		alloc_fn = storage_cpu_alloc;
		dealloc_fn = storage_cpu_dealloc;
	} break;
	case GPU:
		data = storage_gpu_alloc(n_elements);
		size = 0;
		alloc_fn = storage_gpu_alloc;
		dealloc_fn = storage_gpu_dealloc;
	}
	if (!data) {
		return NULL;
	}
	storage *s = (storage *)calloc(1, sizeof(storage));
	if (!s) {
		free(data);
		return NULL;
	}
	s->data = data;
	s->n_elements = n_elements;
	s->size = size;
	atomic_init(&s->refs, 1);
	s->stored_on = device;
	s->flags = STORAGE_DATA_OWNED;
	s->allocator = alloc_fn;
	s->deallocator = dealloc_fn;
	return s;
}

storage *storage_retain(storage *s) {
	if (!s) {
		return NULL;
	}
	atomic_fetch_add(&s->refs, 1);
	return s;
}

void storage_release(storage *s) {
	if (!s)
		return;
	int old = atomic_fetch_sub(&s->refs, 1);
	if (old == 1) {
		s->deallocator(s->data);
		free(s);
	}
}

// allocating CPU storage
float *storage_cpu_alloc(size_t n_elements) {
	size_t size = CPU_ALLOCATION_SIZE(n_elements);
	float *data = (float *)aligned_alloc(VECTOR_ALIGNMENT, size);
	return data;
}

// the call is responsible for the memory
storage *storage_wrap(void *ptr, size_t n_elements, size_t size, device_type device) {
	storage *s = (storage *)calloc(1, sizeof(storage));
	if (!s) {
		return NULL;
	}
	s->data = (float *)(ptr);
	s->n_elements = n_elements;
	s->size = size;
	atomic_init(&s->refs, 1);
	s->stored_on = device;
	s->flags = STORAGE_DATA_BORROWED;
	s->allocator = storage_no_op_alloc;
	s->deallocator = storage_no_op_dealloc;
	return s;
}

// deallocating CPU storage
void storage_cpu_dealloc(void *s) {
	if (s) {
		free(s);
	}
}

// allocting GPU storage
float *storage_gpu_alloc(size_t size);

// deallocting GPU storage
void storage_gpu_dealloc(void *s);

// if the data is not owned by the storage itself
float *storage_no_op_alloc(size_t size) { return NULL; }
// if the data is not owned by the storage itself
void storage_no_op_dealloc(void *s) {}

void storage_copy_cpu_to_gpu(storage *dst, const storage *src); // CPU to GPU
void storage_copy_gpu_to_cpu(storage *dst, const storage *src); // GPU to CPU
