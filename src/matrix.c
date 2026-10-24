#include "../include/lib.c"
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void calculate_strides(const size_t ndims, const size_t *dims, size_t *out_strides);
static size_t calculate_nelements(const size_t ndims, const size_t *dims);

matrix *matrix_init(size_t ndims, const size_t *dims, device_type device) {
	if (ndims <= 0 || !dims)
		return NULL;
	size_t nelements = calculate_nelements(ndims, dims);
	storage *s = (storage *)storage_init(nelements, device);
	if (!s)
		return NULL;
	matrix *m = (matrix *)calloc(1, sizeof(matrix));
	if (!m) {
		storage_release(s);
		return NULL;
	}
	calculate_strides(ndims, dims, m->strides);
	memcpy(m->dims, dims, ndims);
	m->ndims = ndims;
	m->device = device;
	m->data_ptr = s->data;
	m->data = s;
	m->flags = MAT_CONTIGUOUS;
	return m;
}

void matrix_free(matrix *m) {
	if (!m) {
		return;
	}
	storage_release(m->data);
	free(m);
}

matrix *matrix_fill(size_t ndims, const size_t *dims, device_type device, int fill_with) {
	matrix *m = matrix_init(ndims, dims, device);
	if (!m) {
		return NULL;
	}
	memset(m->data_ptr, fill_with, m->data->size);
	return m;
}

matrix *matrix_zero(size_t ndims, const size_t *dims, device_type device) {
	matrix *m = matrix_init(ndims, dims, device);
	if (!m) {
		return NULL;
	}
	memset(m->data_ptr, 0, m->data->size);
	return m;
}

matrix *matrix_ones(size_t ndims, const size_t *dims, device_type device) {
	matrix *m = matrix_init(ndims, dims, device);
	if (!m) {
		return NULL;
	}
	memset(m->data_ptr, 1, m->data->size);
	return m;
}

matrix *matrix_eye(size_t ndims, const size_t *dims, device_type device) {
	if (ndims < 2 || dims[ndims - 2] != dims[ndims - 1])
		return NULL;
	matrix *m = matrix_init(ndims, dims, device);
	if (!m) {
		return NULL;
	}
	size_t batches = 1;
	for (int i = 0; i < ndims - 2; i++) {
		batches *= m->dims[i];
	}
	size_t batch_stride = 1;
	for (int i = 0; i < ndims - 2; i++) {
		batch_stride *= m->strides[i];
	}
	size_t row_stride = m->strides[m->ndims - 2];
	size_t col_stride = m->strides[m->ndims - 1];
	size_t max_dim = dims[ndims - 2] > dims[ndims - 1] ? dims[ndims - 2] : dims[ndims - 1];
	float *tmp_ptr = m->data_ptr;
	for (int i = 0; i < batches; i++) {
		for (int j = 0; j < max_dim; j++) {
			*(tmp_ptr + row_stride * j + col_stride * j) = 1.0f;
		}
		tmp_ptr += batch_stride;
	}
	return m;
}

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

static size_t calculate_nelements(const size_t ndims, const size_t *dims) {
	size_t nelements = 1;
	for (int i = ndims - 1; i > -1; i--) {
		nelements *= dims[i];
	}
	return nelements;
}

static void calculate_strides(const size_t ndims, const size_t *dims, size_t *out_strides) {
	size_t n = 1;
	for (int i = ndims - 1; i > -1; i--) {
		out_strides[i] = n;
		n *= dims[i];
	}
}
