#include <stdlib.h>
#include <stdint.h> 

#include "cuda_helpers.cuh"

#define qswap(A,B) { double temp = A; A = B; B = temp;}

__device__ void *calloc_cuda(size_t nmemb, size_t size)
{
    register uint8_t* mem = (uint8_t*) malloc(nmemb * size);
    if (mem) {
        for (int32_t i = 0; i < nmemb * size; i++) {
            mem[i] = 0;
        }
    }
    return mem;
}

__device__ void *realloc_cuda(void *old_ptr, size_t old_ptr_size, size_t size)
{
    register uint8_t *e;
    e = (uint8_t*) malloc(size);
    for (int i = 0; i < old_ptr_size; i++) {
        e[i] = ((uint8_t*) old_ptr)[i];
    }
    free(old_ptr);

    return (void *) e;
}

__device__ void *memmove_cuda(void *src, void *dest, int32_t n) {
    register uint8_t *src_u = (uint8_t *) src;
    register uint8_t *dest_u = (uint8_t *) dest;

    for (uint64_t i = 0; i < n; i++) {
        *(((uint8_t*) dest) + i) = *(((uint8_t *) src) + i);
        dest_u[i] = src_u[i];
    }

    return dest;
}

__device__ int memcmp_cuda(const void *a, const void *b, int32_t n) {
    register const unsigned char *s1 = (const unsigned char*) a;
    register const unsigned char *s2 = (const unsigned char*) b;

    while (n-- > 0) {
        if (*s1++ != *s2++)
        return s1[-1] < s2[-1] ? -1 : 1;
    }

    return 0;
  }

 __device__ float fmax_cuda(const float a, const float b) {
    return a > b ? a : b;
 }

__device__ float fmin_cuda(const float a, const float b) {
    return a < b ? a : b;
}

__device__ int isdigit_cuda(int x) {
    if (x >= '0' && x <= '9') {
        return 1;
    }
    return 0;
}

__device__ double strtod_cuda(const char *nptr, char **endptr) {
    const char *s = nptr;
    double result = 0.0;
    int sign = 1;
    int exp_sign = 1;
    int exponent = 0;
    int fractional_part = 0;
    double fractional_divisor = 1.0;

    // Skip leading whitespace
    while (isspace_cuda(*s)) {
        s++;
    }

    // Handle optional sign
    if (*s == '-') {
        sign = -1;
        s++;
    } else if (*s == '+') {
        s++;
    }

    // Parse the integer and fractional part
    while (isdigit_cuda(*s) || *s == '.') {
        if (*s == '.') {
            fractional_part = 1;
            s++;
            continue;
        }

        if (!fractional_part) {
            result = result * 10.0 + (*s - '0');
        } else {
            fractional_divisor *= 10.0;
            result += (*s - '0') / fractional_divisor;
        }

        s++;
    }

    // Handle optional exponent part
    if (*s == 'e' || *s == 'E') {
        s++;

        // Handle optional exponent sign
        if (*s == '-') {
            exp_sign = -1;
            s++;
        } else if (*s == '+') {
            s++;
        }

        // Parse the exponent value
        while (isdigit_cuda(*s)) {
            exponent = exponent * 10 + (*s - '0');
            s++;
        }

        // Apply the exponent
        if (exp_sign == 1) {
            result *= pow(10.0, exponent);
        } else {
            result /= pow(10.0, exponent);
        }
    }

    // Apply the overall sign
    result *= sign;

    // Set endptr if needed
    if (endptr) {
        *endptr = (char *)s;
    }

    return result;
}

__device__ int strcmp_cuda(const char *str1, const char *str2) {
    while (*str1 && (*str1 == *str2)) {
        str1++;
        str2++;
    }
    return *(unsigned char *)str1 - *(unsigned char *)str2;
}

__device__ int isspace_cuda(int x) {
    switch (x) {
        case ' ':
        case '\n':
        case '\t':
        case '\v':
        case '\f':
        case '\r':
            return 1;
        default: 
            return 0;
    }
}

__device__ int partition_descending(double arr[], int low, int high) {

    // Initialize pivot to be the first element
    int p = arr[low];
    int i = low;
    int j = high;

   while (i < j) {

        // Find the first element smaller than
        // or equal to the pivot (from the start)
        while (arr[i] >= p && i <= high - 1) {
            i++;
        }

        // Find the first element greater than
        // the pivot (from the end)
        while (arr[j] < p && j >= low + 1) {
            j--;
        }

        // Swap the elements if indices have not crossed
        if (i < j) {
            qswap(arr[i], arr[j]);
        }
    }

    qswap(arr[low], arr[j]);
    return j;
}

__device__ int partition_ascending(double arr[], int low, int high) {

    // Initialize pivot to be the first element
    int p = arr[low];
    int i = low;
    int j = high;

   while (i < j) {

        // Find the first element smaller than
        // or equal to the pivot (from the start)
        while (arr[i] <= p && i <= high - 1) {
            i++;
        }

        // Find the first element greater than
        // the pivot (from the end)
        while (arr[j] > p && j >= low + 1) {
            j--;
        }

        // Swap the elements if indices have not crossed
        if (i < j) {
            qswap(arr[i], arr[j]);
        }
    }

    qswap(arr[low], arr[j]);
    return j;
}


__device__ void quick_sort_descending_cuda(double arr[], int low, int high) {
    if (low < high) {

        // call partition function to find Partition Index
        int pi = partition_descending(arr, low, high);

        // Recursively call quickSort() for left and right
        // half based on Partition Index
        quick_sort_descending_cuda(arr, low, pi - 1);
        quick_sort_descending_cuda(arr, pi + 1, high);
    }
}

__device__ void quick_sort_ascending_cuda(double arr[], int low, int high) {
    if (low < high) {

        // call partition function to find Partition Index
        int pi = partition_ascending(arr, low, high);

        // Recursively call quickSort() for left and right
        // half based on Partition Index
        quick_sort_ascending_cuda(arr, low, pi - 1);
        quick_sort_ascending_cuda(arr, pi + 1, high);
    }
}
