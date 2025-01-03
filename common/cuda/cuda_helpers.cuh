#pragma once

extern __device__ void *calloc_cuda(size_t nmemb, size_t size);

__device__ void *realloc_cuda(void *old_ptr, size_t old_ptr_size, size_t size);

__device__ void *memmove_cuda(void *src, void *dest, int32_t n);

__device__ int memcmp_cuda(const void *a, const void *b, int32_t n);

__device__ float fmax_cuda(const float a, const float b);

__device__ float fmin_cuda(const float a, const float b);

__device__ double strtod_cuda(const char *nptr, char **endptr);

__device__ int strcmp_cuda(const char *str1, const char *str2);

__device__ int isspace_cuda(int x);

__device__ int isdigit_cuda(int x);

__device__ void quick_sort_descending_cuda(double arr[], int low, int high);

__device__ void quick_sort_ascending_cuda(double arr[], int low, int high);
