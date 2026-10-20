#include "../include/lib.c"

static uint64_t splitmix64_next(uint64_t *state);

static uint64_t splitmix64_next(uint64_t *state) {
	uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

rng_state *rng_from_seed(uint64_t seed) {
	rng_state *r = calloc(1, sizeof(rng_state));
	if (!r)
		return NULL;
	uint64_t s = seed;
	r->state = splitmix64_next(&s);
	r->state ^= splitmix64_next(&s);
	return r;
}

rng_state *rng_from_entropy() {
	uint64_t seed = 0;
	int fd = open("/dev/urandom", O_RDONLY);
	if (fd >= 0) {
		ssize_t n = read(fd, &seed, sizeof(seed));
		close(fd);
		if (n == sizeof(seed)) {
			return rng_from_seed(seed);
		}
	}
	uint64_t t = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)&seed;
	return rng_from_seed(t);
}

void rng_free(rng_state *r) {
	if (!r)
		return;
	r->state = 0;
	free(r);
}

/* 64-bit raw output from SplitMix64 (advances state) */
uint64_t rng_next_u64(rng_state *r) {
	// splitmix as both seeder and generator
	uint64_t s = r->state;
	uint64_t out = splitmix64_next(&s);
	r->state = s;
	return out;
}

/* 32-bit output (take upper 32 bits of previous 64-bit output) */
uint32_t rng_next_u32(rng_state *r) {
	uint64_t u64 = rng_next_u64(r);
	return (uint32_t)(u64 >> 32);
}

/* uniform float32 in [0,1). Avoid producing exact 1.0. */
float rng_next_f32(rng_state *r) {
	uint32_t u = rng_next_u32(r);
	// -> convert using double to reduce rounding bias, then cast to float
	double x = (double)u / 4294967296.0; // 2^32
	return (float)x;
}

/* uniform integer in [0, n) unbiased using rejection sampling. assumes 1 <= n <= 2^32. */
uint32_t rng_uniform_int(rng_state *r, uint32_t n) {
	if (n == 0) {
		return 0;
	}
	const uint64_t limit = 4294967296ULL; // 2^32
	uint64_t bound = (limit / n) * n;     // largest multiple of n < 2^32
	uint32_t x;
	while (1) {
		x = rng_next_u32(r);
		if ((uint64_t)x < bound)
			return x % n;
	}
}

/* Box-Muller normal sampler (mean, std) returning float32 -> to get curves from flat lines. Caches the second sample for next call. */
void rng_normal_f32(rng_state *r, float mean, float stddev, float *out_1, float *out_2) {
	// generate two uniform floats in (0,1], avoid zero.
	// using u in (0,1] by mapping u = (u32 + 1) / 2^32 to avoid log(0).
	uint32_t a = rng_next_u32(r);
	uint32_t b = rng_next_u32(r);
	double u1 = ((double)a + 1.0) / 4294967296.0; // max 2^32
	double u2 = ((double)b + 1.0) / 4294967296.0; // max 2^32
	double rads = sqrt(-2.0 * log(u1));
	double theta = 2.0 * M_PI * u2;
	double z1 = rads * cos(theta);
	double z2 = rads * sin(theta);
	*out_1 = mean + stddev * (float)z1;
	*out_2 = mean + stddev * (float)z2;
}

/* block allocation -> fill contiguous float32 buffer with uniform in [a,b). */
void rng_linear_block_fill_uniform_f32(rng_state *r, float *dst, size_t n, float a, float b) {
	float range = b - a;
	for (size_t i = 0; i < n; ++i) {
		float u = rng_next_f32(r);
		dst[i] = a + range * u;
	}
}

/* block allocation -> fill contiguous float32 buffer with normal(mean,std) */
void rng_linear_block_fill_normal_f32(rng_state *r, float *dst, size_t n, float mean, float stddev) {
	size_t i = 0;
	for (; i + 1 < n; i += 2) {
		float out_1, out_2;
		rng_normal_f32(r, mean, stddev, &out_1, &out_2);
		dst[i] = out_1;
		dst[i + 1] = out_2;
	}

	// incase if n is odd
	if (i < n) {
		float out_1, _;
		rng_normal_f32(r, mean, stddev, &out_1, &_);
		dst[i] = out_1;
	}
}

/* shuffle in position -> fisher-yates shuffle for integer array (in-place) */
void rng_shuffle_float_array(rng_state *r, float *arr, size_t n) {
	if (n <= 1)
		return;
	for (size_t i = n - 1; i > 0; --i) {
		uint32_t j = rng_uniform_int(r, (uint32_t)(i + 1)); // [0, i]
		int t = arr[i];
		arr[i] = arr[j];
		arr[j] = t;
	}
}
