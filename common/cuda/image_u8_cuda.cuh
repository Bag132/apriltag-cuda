/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.
This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#pragma once

#include <stdint.h>
#include "image_types_cuda.cuh"

typedef struct image_u8_lut_cuda image_u8_lut_cuda_t;
struct image_u8_lut_cuda
{
    // When drawing, we compute the squared distance between a given pixel and a filled region.
    // int idx = squared_distance * scale;
    // We then index into values[idx] to obtain the color. (If we must index beyond nvalues,
    // no drawing is performed.)
    float    scale;

    int      nvalues;
    uint8_t *values;
};


// Create or load an image. returns NULL on failure. Uses default
// stride alignment.
__device__ image_u8_cuda_t *image_u8_create_stride_cuda(unsigned int width, unsigned int height, unsigned int stride);
__device__ image_u8_cuda_t *image_u8_create_cuda(unsigned int width, unsigned int height);
__device__ image_u8_cuda_t *image_u8_create_alignment_cuda(unsigned int width, unsigned int height, unsigned int alignment);
// __device__ image_u8_cuda_t *image_u8_create_from_f32_cuda(image_f32_cuda_t *fim);

// image_u8_t *image_u8_create_from_pnm(const char *path);
//     image_u8_t *image_u8_create_from_pnm_alignment(const char *path, int alignment);

__device__ image_u8_cuda_t *image_u8_copy_cuda(const image_u8_cuda_t *in);
__device__ void image_u8_draw_line_cuda(image_u8_cuda_t *im, float x0, float y0, float x1, float y1, int v, int width);
__device__ void image_u8_draw_circle_cuda(image_u8_cuda_t *im, float x0, float y0, float r, int v);
__device__ void image_u8_draw_annulus_cuda(image_u8_cuda_t *im, float x0, float y0, float r0, float r1, int v);

__device__ void image_u8_fill_line_max_cuda(image_u8_cuda_t *im, const image_u8_lut_cuda_t *lut, const float *xy0, const float *xy1);

// __device__ void image_u8_clear_cuda(image_u8_cuda_t *im);
__device__ void image_u8_darken_cuda(image_u8_cuda_t *im);
__device__ void image_u8_convolve_2D_cuda(image_u8_cuda_t *im, const uint8_t *k, int ksz);
__device__ void image_u8_gaussian_blur_cuda(image_u8_cuda_t *im, double sigma, int k);

// 1.5, 2, 3, 4, ... supported
__device__ image_u8_cuda_t *image_u8_decimate_cuda(image_u8_cuda_t *im, float factor);

__device__ void image_u8_destroy_cuda(image_u8_cuda_t *im);

// Write a pnm. Returns 0 on success
// Currently only supports GRAY and RGBA. Does not write out alpha for RGBA
// __device__ int image_u8_write_pnm(const image_u8_t *im, const char *path);

// rotate the image by 'rad' radians. (Rotated in the "intuitive
// sense", i.e., if Y were up. When input values are unavailable, the
// value 'pad' is inserted instead. The geometric center of the output
// image corresponds to the geometric center of the input image.
__device__ image_u8_cuda_t *image_u8_rotate_cuda(const image_u8_cuda_t *in, double rad, uint8_t pad);
