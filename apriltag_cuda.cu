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

#include "apriltag_cuda.cuh"

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "common/cuda/image_u8_cuda.cuh"
#include "common/cuda/image_u8x3_cuda.cuh"
#include "common/cuda/zarray_cuda.cuh"
#include "common/cuda/matd_cuda.cuh"
#include "common/cuda/homography_cuda.cuh"
// #include "common/timeprofile.h"
#include "common/cuda/math_util_cuda.cuh"
#include "common/cuda/g2d_cuda.cuh"
#include "common/cuda/unionfind_cuda.cuh"
// #include "common/debug_print.h"

#include "cuda_helpers.cuh"

#include "apriltag_math_cuda.cuh"

#include "tag36h11_cuda.cuh"

// #include "common/postscript_utils.h"

#define T0_print(format) if (threadIdx.x == 0) printf(format)
#define T0_printf(format, ...) if (threadIdx.x == 0) printf(format, __VA_ARGS__)

///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// START CUDA version
//
///////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef _WIN32
static inline void srandom(unsigned int seed)
{
        srand(seed);
}

static inline long int random(void)
{
        return rand();
}
#endif

#define APRILTAG_U64_ONE ((uint64_t) 1)

__device__ extern zarray_cuda_t *apriltag_quad_thresh_cuda(apriltag_detector_cuda_t *td, image_u8_cuda_t *im, int32_t num_threads, image_u8x3_cuda_t **dbg);

// Regresses a model of the form:
// intensity(x,y) = C0*x + C1*y + CC2
// The J matrix is the:
//    J = [ x1 y1 1 ]
//        [ x2 y2 1 ]
//        [ ...     ]
//  The A matrix is J'J

struct graymodel
{
    double A[3][3];
    double B[3];
    double C[3];
};

__device__ void graymodel_init_cuda(struct graymodel *gm)
{
    memset(gm, 0, sizeof(struct graymodel));
}

__device__ void graymodel_add_cuda(struct graymodel *gm, double x, double y, double gray)
{
    // update upper right entries of A = J'J
    gm->A[0][0] += x*x;
    gm->A[0][1] += x*y;
    gm->A[0][2] += x;
    gm->A[1][1] += y*y;
    gm->A[1][2] += y;
    gm->A[2][2] += 1;

    // update B = J'gray
    gm->B[0] += x * gray;
    gm->B[1] += y * gray;
    gm->B[2] += gray;
}

__device__ static void graymodel_solve_cuda(struct graymodel *gm)
{
    mat33_sym_solve_cuda((double*) gm->A, gm->B, gm->C);
}

__device__ static double graymodel_interpolate_cuda(struct graymodel *gm, double x, double y)
{
    return gm->C[0]*x + gm->C[1]*y + gm->C[2];
}

struct quick_decode_entry
{
    uint64_t rcode;   // the queried code
    uint16_t id;      // the tag ID (a small integer)
    uint8_t hamming;  // how many errors corrected?
    uint8_t rotation; // number of rotations [0, 3]
};

struct quick_decode
{
    int nentries;
    struct quick_decode_entry *entries;
};

/**
 * Assuming we are drawing the image one quadrant at a time, what would the rotated image look like?
 * Special care is taken to handle the case where there is a middle pixel of the image.
 */
__device__ static uint64_t rotate90_cuda(uint64_t w, int numBits)
{
    int p = numBits;
    uint64_t l = 0;
    if (numBits % 4 == 1) {
	p = numBits - 1;
	l = 1;
    }
    w = ((w >> l) << (p/4 + l)) | (w >> (3 * p/ 4 + l) << l) | (w & l);
    w &= ((APRILTAG_U64_ONE << numBits) - 1);
    return w;
}

__device__ static void quad_destroy_cuda(struct quad_cuda *quad)
{
    if (!quad)
        return;

    matd_destroy_cuda(quad->H);
    matd_destroy_cuda(quad->Hinv);
    free(quad);
}

__device__ struct quad_cuda *quad_copy_cuda(struct quad_cuda *quad)
{
    struct quad_cuda *q = (struct quad_cuda *) calloc_cuda(1, sizeof(struct quad_cuda));
    memcpy(q, quad, sizeof(struct quad_cuda));
    if (quad->H)
        q->H = matd_copy_cuda(quad->H);
    if (quad->Hinv)
        q->Hinv = matd_copy_cuda(quad->Hinv);
    return q;
}

__host__ __device__ void quick_decode_add_cuda(struct quick_decode *qd, uint64_t code, int id, int hamming)
{
    uint32_t bucket = code % qd->nentries;

    while (qd->entries[bucket].rcode != UINT64_MAX) {
        bucket = (bucket + 1) % qd->nentries;
    }

    qd->entries[bucket].rcode = code;
    qd->entries[bucket].id = id;
    qd->entries[bucket].hamming = hamming;
}

__device__ void quick_decode_uninit_cuda(apriltag_family_cuda_t *fam)
{
    if (!fam->impl)
        return;

    struct quick_decode *qd = (struct quick_decode*) fam->impl;
    free(qd->entries);
    free(qd);
    fam->impl = NULL;
}

__host__ __device__ void quick_decode_init_cuda(apriltag_family_cuda_t *family, int maxhamming)
{
    assert(family->impl == NULL);
    assert(family->ncodes < 65536);

    struct quick_decode *qd = (struct quick_decode *) calloc_cuda(1, sizeof(struct quick_decode));
    int capacity = family->ncodes;

    int nbits = family->nbits;

    if (maxhamming >= 1)
        capacity += family->ncodes * nbits;

    if (maxhamming >= 2)
        capacity += family->ncodes * nbits * (nbits-1);

    if (maxhamming >= 3)
        capacity += family->ncodes * nbits * (nbits-1) * (nbits-2);

    qd->nentries = capacity * 3;

//    debug_print("capacity %d, size: %.0f kB\n",
//           capacity, qd->nentries * sizeof(struct quick_decode_entry) / 1024.0);

    qd->entries = (struct quick_decode_entry *) calloc_cuda(qd->nentries, sizeof(struct quick_decode_entry));
    if (qd->entries == NULL) {
        printf("Failed to allocate hamming decode table\n");
        // errno already set to ENOMEM (Error No MEMory) by calloc() failure
        return;
    }

    for (int i = 0; i < qd->nentries; i++)
        qd->entries[i].rcode = UINT64_MAX;

    // errno = 0;

    for (uint32_t i = 0; i < family->ncodes; i++) {
        uint64_t code = family->codes[i];

        // add exact code (hamming = 0)
        quick_decode_add_cuda(qd, code, i, 0);

        if (maxhamming >= 1) {
            // add hamming 1
            for (int j = 0; j < nbits; j++)
                quick_decode_add_cuda(qd, code ^ (APRILTAG_U64_ONE << j), i, 1);
        }

        if (maxhamming >= 2) {
            // add hamming 2
            for (int j = 0; j < nbits; j++)
                for (int k = 0; k < j; k++)
                    quick_decode_add_cuda(qd, code ^ (APRILTAG_U64_ONE << j) ^ (APRILTAG_U64_ONE << k), i, 2);
        }

        if (maxhamming >= 3) {
            // add hamming 3
            for (int j = 0; j < nbits; j++)
                for (int k = 0; k < j; k++)
                    for (int m = 0; m < k; m++)
                        quick_decode_add_cuda(qd, code ^ (APRILTAG_U64_ONE << j) ^ (APRILTAG_U64_ONE << k) ^ (APRILTAG_U64_ONE << m), i, 3);
        }

        if (maxhamming > 3) {
            printf("\"maxhamming\" beyond 3 not supported\n");
            // set errno to Error INvalid VALue
            // errno = EINVAL;
            return;
        }
    }

    family->impl = qd;

    #if 0
        int longest_run = 0;
        int run = 0;
        int run_sum = 0;
        int run_count = 0;

        // This accounting code doesn't check the last possible run that
        // occurs at the wrap-around. That's pretty insignificant.
        for (int i = 0; i < qd->nentries; i++) {
            if (qd->entries[i].rcode == UINT64_MAX) {
                if (run > 0) {
                    run_sum += run;
                    run_count ++;
                }
                run = 0;
            } else {
                run ++;
                longest_run = imax(longest_run, run);
            }
        }

        printf("quick decode: longest run: %d, average run %.3f\n", longest_run, 1.0 * run_sum / run_count);
    #endif
}

// returns an entry with hamming set to 255 if no decode was found.
__device__ static void quick_decode_codeword_cuda(apriltag_family_cuda_t *tf, uint64_t rcode,
                                  struct quick_decode_entry *entry)
{
    struct quick_decode *qd = (struct quick_decode*) tf->impl;

    // qd might be null if detector_add_family_bits() failed
    for (int ridx = 0; qd != NULL && ridx < 4; ridx++) {

        for (int bucket = rcode % qd->nentries;
             qd->entries[bucket].rcode != UINT64_MAX;
             bucket = (bucket + 1) % qd->nentries) {

            if (qd->entries[bucket].rcode == rcode) {
                *entry = qd->entries[bucket];
                entry->rotation = ridx;
                return;
            }
        }

        rcode = rotate90_cuda(rcode, tf->nbits);
    }

    entry->rcode = 0;
    entry->id = 65535;
    entry->hamming = 255;
    entry->rotation = 0;
}

__device__ static inline int detection_compare_function_cuda(const void *_a, const void *_b)
{
    apriltag_detection_cuda_t *a = *(apriltag_detection_cuda_t**) _a;
    apriltag_detection_cuda_t *b = *(apriltag_detection_cuda_t**) _b;

    return a->id - b->id;
}

__device__ void apriltag_detector_remove_family_cuda(apriltag_detector_cuda_t *td, apriltag_family_cuda_t *fam)
{
    quick_decode_uninit_cuda(fam);
    zarray_remove_value_cuda(td->tag_families, &fam, 0);
}

__host__ __device__ void apriltag_detector_add_family_bits_cuda(apriltag_detector_cuda_t *td, apriltag_family_cuda_t *fam, int bits_corrected)
{
    zarray_add_cuda(td->tag_families, &fam);

    if (!fam->impl)
        quick_decode_init_cuda(fam, bits_corrected);
}

__device__ void apriltag_detector_clear_families_cuda(apriltag_detector_cuda_t *td)
{
    for (int i = 0; i < zarray_size_cuda(td->tag_families); i++) {
        apriltag_family_cuda_t *fam;
        zarray_get_cuda(td->tag_families, i, &fam);
        quick_decode_uninit_cuda(fam);
    }
    zarray_clear_cuda(td->tag_families);
}

__host__ __device__ apriltag_detector_cuda_t *apriltag_detector_create_cuda()
{
    apriltag_detector_cuda_t *td = (apriltag_detector_cuda_t*) calloc_cuda(1, sizeof(apriltag_detector_cuda_t));

    td->nthreads = 1;
    td->quad_decimate = 2.0;
    td->quad_sigma = 0.0;

    td->qtp.max_nmaxima = 10;
    td->qtp.min_cluster_pixels = 5;

    td->qtp.max_line_fit_mse = 10.0;
    td->qtp.cos_critical_rad = cos(10 * M_PI / 180);
    td->qtp.deglitch = false;
    td->qtp.min_white_black_diff = 5;

    td->tag_families = zarray_create_cuda(sizeof(apriltag_family_cuda_t*));

    // pthread_mutex_init(&td->mutex, NULL);

    // td->tp = timeprofile_create();

    td->refine_edges = true;
    td->decode_sharpening = 0.25;


    td->debug = false;

    // NB: defer initialization of td->wp so that the user can
    // override td->nthreads.

    return td;
}

__device__ void apriltag_detector_destroy_cuda(apriltag_detector_cuda_t *td)
{
    // timeprofile_destroy(td->tp);
    // workerpool_destroy(td->wp);

    apriltag_detector_clear_families_cuda(td);

    zarray_destroy_cuda(td->tag_families);
    free(td);
}

struct quad_decode_task
{
    int i0, i1;
    zarray_cuda_t *quads;
    apriltag_detector_cuda_t *td;

    image_u8_cuda_t *im;
    zarray_cuda_t *detections;

    image_u8_cuda_t *im_samples;
};

struct evaluate_quad_ret
{
    int64_t rcode;
    double  score;
    matd_cuda_t  *H, *Hinv;

    int decode_status;
    struct quick_decode_entry e;
};

__device__ static matd_cuda_t* homography_compute2_cuda(double c[4][4]) {
    double A[] =  {
            c[0][0], c[0][1], 1,       0,       0, 0, -c[0][0]*c[0][2], -c[0][1]*c[0][2], c[0][2],
                  0,       0, 0, c[0][0], c[0][1], 1, -c[0][0]*c[0][3], -c[0][1]*c[0][3], c[0][3],
            c[1][0], c[1][1], 1,       0,       0, 0, -c[1][0]*c[1][2], -c[1][1]*c[1][2], c[1][2],
                  0,       0, 0, c[1][0], c[1][1], 1, -c[1][0]*c[1][3], -c[1][1]*c[1][3], c[1][3],
            c[2][0], c[2][1], 1,       0,       0, 0, -c[2][0]*c[2][2], -c[2][1]*c[2][2], c[2][2],
                  0,       0, 0, c[2][0], c[2][1], 1, -c[2][0]*c[2][3], -c[2][1]*c[2][3], c[2][3],
            c[3][0], c[3][1], 1,       0,       0, 0, -c[3][0]*c[3][2], -c[3][1]*c[3][2], c[3][2],
                  0,       0, 0, c[3][0], c[3][1], 1, -c[3][0]*c[3][3], -c[3][1]*c[3][3], c[3][3],
    };

    double epsilon = 1e-10;

    // Eliminate.
    for (int col = 0; col < 8; col++) {
        // Find best row to swap with.
        double max_val = 0;
        int max_val_idx = -1;
        for (int row = col; row < 8; row++) {
            double val = fabs(A[row*9 + col]);
            if (val > max_val) {
                max_val = val;
                max_val_idx = row;
            }
        }

        if (max_val_idx < 0) {
            return NULL;
        }

        if (max_val < epsilon) {
            printf("WRN: Matrix is singular.\n");
            return NULL;
        }

        // Swap to get best row.
        if (max_val_idx != col) {
            for (int i = col; i < 9; i++) {
                double tmp = A[col*9 + i];
                A[col*9 + i] = A[max_val_idx*9 + i];
                A[max_val_idx*9 + i] = tmp;
            }
        }

        // Do eliminate.
        for (int i = col + 1; i < 8; i++) {
            double f = A[i*9 + col]/A[col*9 + col];
            A[i*9 + col] = 0;
            for (int j = col + 1; j < 9; j++) {
                A[i*9 + j] -= f*A[col*9 + j];
            }
        }
    }

    // Back solve.
    for (int col = 7; col >=0; col--) {
        double sum = 0;
        for (int i = col + 1; i < 8; i++) {
            sum += A[col*9 + i]*A[i*9 + 8];
        }
        A[col*9 + 8] = (A[col*9 + 8] - sum)/A[col*9 + col];
    }
    return matd_create_data_cuda(3, 3, (double[]) { A[8], A[17], A[26], A[35], A[44], A[53], A[62], A[71], 1 });
}

// returns non-zero if an error occurs (i.e., H has no inverse)
__device__ static int quad_update_homographies_cuda(struct quad_cuda *quad)
{
    //zarray_t *correspondences = zarray_create(sizeof(float[4]));

    double corr_arr[4][4];

    for (int i = 0; i < 4; i++) {
        corr_arr[i][0] = (i==0 || i==3) ? -1 : 1;
        corr_arr[i][1] = (i==0 || i==1) ? -1 : 1;
        corr_arr[i][2] = quad->p[i][0];
        corr_arr[i][3] = quad->p[i][1];
    }

    if (quad->H)
        matd_destroy_cuda(quad->H);
    if (quad->Hinv)
        matd_destroy_cuda(quad->Hinv);

    // XXX Tunable
    quad->H = homography_compute2_cuda(corr_arr);
    if (quad->H != NULL) {
        quad->Hinv = matd_inverse_cuda(quad->H);
        if (quad->Hinv != NULL) {
	    // Success!
            return 0;
        }
        matd_destroy_cuda(quad->H);
        quad->H = NULL;
    }
    return -1;
}

__device__ static double value_for_pixel_cuda(image_u8_cuda_t *im, double px, double py) {
    int x1 = floor(px - 0.5);
    int x2 = ceil(px - 0.5);
    double x = px - 0.5 - x1;
    int y1 = floor(py - 0.5);
    int y2 = ceil(py - 0.5);
    double y = py - 0.5 - y1;
    if (x1 < 0 || x2 >= im->width || y1 < 0 || y2 >= im->height) {
        return -1;
    }
    return im->buf[y1*im->stride + x1]*(1-x)*(1-y) +
            im->buf[y1*im->stride + x2]*x*(1-y) +
            im->buf[y2*im->stride + x1]*(1-x)*y +
            im->buf[y2*im->stride + x2]*x*y;
}

__device__ static void sharpen_cuda(apriltag_detector_cuda_t* td, double* values, int size) {
    double *sharpened = (double *) malloc(sizeof(double)*size*size);
    double kernel[9] = {
        0, -1, 0,
        -1, 4, -1,
        0, -1, 0
    };

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            sharpened[y*size + x] = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if ((y + i - 1) < 0 || (y + i - 1) > size - 1 || (x + j - 1) < 0 || (x + j - 1) > size - 1) {
                        continue;
                    }
                    sharpened[y*size + x] += values[(y + i - 1)*size + (x + j - 1)]*kernel[i*3 + j];
                }
            }
        }
    }


    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            values[y*size + x] = values[y*size + x] + td->decode_sharpening*sharpened[y*size + x];
        }
    }

    free(sharpened);
}

// returns the decision margin. Return < 0 if the detection should be rejected.
__device__ float quad_decode_cuda(apriltag_detector_cuda_t* td, apriltag_family_cuda_t *family, image_u8_cuda_t *im, struct quad_cuda *quad, struct quick_decode_entry *entry, image_u8_cuda_t *im_samples)
{
    // decode the tag binary contents by sampling the pixel
    // closest to the center of each bit cell.

    // We will compute a threshold by sampling known white/black cells around this tag.
    // This sampling is achieved by considering a set of samples along lines.
    //
    // coordinates are given in bit coordinates. ([0, fam->border_width]).
    //
    // { initial x, initial y, delta x, delta y, WHITE=1 }
    float patterns[] = {
        // left white column
        -0.5, 0.5,
        0, 1,
        1,

        // left black column
        0.5, 0.5,
        0, 1,
        0,

        // right white column
        family->width_at_border + 0.5, .5,
        0, 1,
        1,

        // right black column
        family->width_at_border - 0.5, .5,
        0, 1,
        0,

        // top white row
        0.5, -0.5,
        1, 0,
        1,

        // top black row
        0.5, 0.5,
        1, 0,
        0,

        // bottom white row
        0.5, family->width_at_border + 0.5,
        1, 0,
        1,

        // bottom black row
        0.5, family->width_at_border - 0.5,
        1, 0,
        0

        // XXX double-counts the corners.
    };

    struct graymodel whitemodel, blackmodel;
    graymodel_init_cuda(&whitemodel);
    graymodel_init_cuda(&blackmodel);

    for (long unsigned int pattern_idx = 0; pattern_idx < sizeof(patterns)/(5*sizeof(float)); pattern_idx ++) {
        float *pattern = &patterns[pattern_idx * 5];

        int is_white = pattern[4];

        for (int i = 0; i < family->width_at_border; i++) {
            double tagx01 = (pattern[0] + i*pattern[2]) / (family->width_at_border);
            double tagy01 = (pattern[1] + i*pattern[3]) / (family->width_at_border);

            double tagx = 2*(tagx01-0.5);
            double tagy = 2*(tagy01-0.5);

            double px, py;
            homography_project_cuda(quad->H, tagx, tagy, &px, &py);

            // don't round
            int ix = px;
            int iy = py;
            if (ix < 0 || iy < 0 || ix >= im->width || iy >= im->height)
                continue;

            int v = im->buf[iy*im->stride + ix];

            if (im_samples) {
                im_samples->buf[iy*im_samples->stride + ix] = (1-is_white)*255;
            }

            if (is_white)
                graymodel_add_cuda(&whitemodel, tagx, tagy, v);
            else
                graymodel_add_cuda(&blackmodel, tagx, tagy, v);
        }
    }

    if (family->width_at_border > 1) {
        graymodel_solve_cuda(&whitemodel);
        graymodel_solve_cuda(&blackmodel);
    } else {
        graymodel_solve_cuda(&whitemodel);
        blackmodel.C[0] = 0;
        blackmodel.C[1] = 0;
        blackmodel.C[2] = blackmodel.B[2]/4;
    }

    // XXX Tunable
    if ((graymodel_interpolate_cuda(&whitemodel, 0, 0) - graymodel_interpolate_cuda(&blackmodel, 0, 0) < 0) != family->reversed_border) {
        return -1;
    }

    // compute the average decision margin (how far was each bit from
    // the decision boundary?
    //
    // we score this separately for white and black pixels and return
    // the minimum average threshold for black/white pixels. This is
    // to penalize thresholds that are too close to an extreme.
    float black_score = 0, white_score = 0;
    float black_score_count = 1, white_score_count = 1;

    double *values = (double *) calloc_cuda(family->total_width*family->total_width, sizeof(double));

    int min_coord = (family->width_at_border - family->total_width)/2;
    for (uint32_t i = 0; i < family->nbits; i++) {
        int bity = family->bit_y[i];
        int bitx = family->bit_x[i];

        double tagx01 = (bitx + 0.5) / (family->width_at_border);
        double tagy01 = (bity + 0.5) / (family->width_at_border);

        // scale to [-1, 1]
        double tagx = 2*(tagx01-0.5);
        double tagy = 2*(tagy01-0.5);

        double px, py;
        homography_project_cuda(quad->H, tagx, tagy, &px, &py);

        double v = value_for_pixel_cuda(im, px, py);

        if (v == -1) {
            continue;
        }

        double thresh = (graymodel_interpolate_cuda(&blackmodel, tagx, tagy) + graymodel_interpolate_cuda(&whitemodel, tagx, tagy)) / 2.0;
        values[family->total_width*(bity - min_coord) + bitx - min_coord] = v - thresh;

        if (im_samples) {
            int ix = px;
            int iy = py;
            im_samples->buf[iy*im_samples->stride + ix] = (v < thresh) * 255;
        }
    }

    sharpen_cuda(td, values, family->total_width);

    uint64_t rcode = 0;
    for (uint32_t i = 0; i < family->nbits; i++) {
        int bity = family->bit_y[i];
        int bitx = family->bit_x[i];
        rcode = (rcode << 1);
        double v = values[(bity - min_coord)*family->total_width + bitx - min_coord];

        if (v > 0) {
            white_score += v;
            white_score_count++;
            rcode |= 1;
        } else {
            black_score -= v;
            black_score_count++;
        }
    }

    quick_decode_codeword_cuda(family, rcode, entry);
    free(values);
    return fmin(white_score / white_score_count, black_score / black_score_count);
}

__device__ static void refine_edges_cuda(apriltag_detector_cuda_t *td, image_u8_cuda_t *im_orig, struct quad_cuda *quad)
{
    double lines[4][4]; // for each line, [Ex Ey nx ny]

    for (int edge = 0; edge < 4; edge++) {
        int a = edge, b = (edge + 1) & 3; // indices of the end points.

        // compute the normal to the current line estimate
        double nx = quad->p[b][1] - quad->p[a][1];
        double ny = -quad->p[b][0] + quad->p[a][0];
        double mag = sqrt(nx*nx + ny*ny);
        nx /= mag;
        ny /= mag;

        if (quad->reversed_border) {
            nx = -nx;
            ny = -ny;
        }

        // we will now fit a NEW line by sampling points near
        // our original line that have large gradients. On really big tags,
        // we're willing to sample more to get an even better estimate.
        int nsamples = imax_cuda(16, mag / 8); // XXX tunable

        // stats for fitting a line...
        double Mx = 0, My = 0, Mxx = 0, Mxy = 0, Myy = 0, N = 0;

        for (int s = 0; s < nsamples; s++) {
            // compute a point along the line... Note, we're avoiding
            // sampling *right* at the corners, since those points are
            // the least reliable.
            double alpha = (1.0 + s) / (nsamples + 1);
            double x0 = alpha*quad->p[a][0] + (1-alpha)*quad->p[b][0];
            double y0 = alpha*quad->p[a][1] + (1-alpha)*quad->p[b][1];

            // search along the normal to this line, looking at the
            // gradients along the way. We're looking for a strong
            // response.
            double Mn = 0;
            double Mcount = 0;

            // XXX tunable: how far to search?  We want to search far
            // enough that we find the best edge, but not so far that
            // we hit other edges that aren't part of the tag. We
            // shouldn't ever have to search more than quad_decimate,
            // since otherwise we would (ideally) have started our
            // search on another pixel in the first place. Likewise,
            // for very small tags, we don't want the range to be too
            // big.

            int range = td->quad_decimate + 1;

            // To reduce the overhead of bilinear interpolation, we can
            // reduce the number of steps per unit.
            int steps_per_unit = 4;
            double step_length = 1.0 / steps_per_unit;
            int max_steps = 2 * steps_per_unit * range + 1;
            double delta = 0.5;

            // XXX tunable step size.
            for (int step = 0; step < max_steps; ++step) {
                double n = -range + step_length * step;
                // Because of the guaranteed winding order of the
                // points in the quad, we will start inside the white
                // portion of the quad and work our way outward.
                //
                // sample to points (x1,y1) and (x2,y2) XXX tunable:
                // how far +/- to look? Small values compute the
                // gradient more precisely, but are more sensitive to
                // noise.
                double grange = 1;

                double x1 = x0 + (n + grange)*nx - delta;
                double y1 = y0 + (n + grange)*ny - delta;
                double x1i_d, y1i_d, a1, b1;
                a1 = modf(x1, &x1i_d);
                b1 = modf(y1, &y1i_d);
                int x1i = x1i_d, y1i = y1i_d;

                if (x1i < 0 || x1i + 1 >= im_orig->width || y1i < 0 || y1i + 1 >= im_orig->height)
                    continue;

                double x2 = x0 + (n - grange)*nx - delta;
                double y2 = y0 + (n - grange)*ny - delta;
                double x2i_d, y2i_d, a2, b2;
                a2 = modf(x2, &x2i_d);
                b2 = modf(y2, &y2i_d);
                int x2i = x2i_d, y2i = y2i_d;

                if (x2i < 0 || x2i + 1 >= im_orig->width || y2i < 0 || y2i + 1 >= im_orig->height)
                    continue;

                // interpolate
                double g1 = (1 - a1) * (1 - b1) * im_orig->buf[y1i*im_orig->stride + x1i] +
                                  a1 * (1 - b1) * im_orig->buf[y1i*im_orig->stride + x1i + 1] +
                            (1 - a1) *    b1    * im_orig->buf[(y1i + 1)*im_orig->stride + x1i] +
                                  a1 *    b1    * im_orig->buf[(y1i + 1)*im_orig->stride + x1i + 1];
                double g2 = (1 - a2) * (1 - b2) * im_orig->buf[y2i*im_orig->stride + x2i] +
                                  a2 * (1 - b2) * im_orig->buf[y2i*im_orig->stride + x2i + 1] +
                            (1 - a2) *    b2    * im_orig->buf[(y2i + 1)*im_orig->stride + x2i] +
                                  a2 *    b2    * im_orig->buf[(y2i + 1)*im_orig->stride + x2i + 1];
                if (g1 < g2) // reject points whose gradient is "backwards". They can only hurt us.
                    continue;

                double weight = (g2 - g1)*(g2 - g1); // XXX tunable. What shape for weight=f(g2-g1)?

                // compute weighted average of the gradient at this point.
                Mn += weight*n;
                Mcount += weight;
            }

            // what was the average point along the line?
            if (Mcount == 0)
                continue;

            double n0 = Mn / Mcount;

            // where is the point along the line?
            double bestx = x0 + n0*nx;
            double besty = y0 + n0*ny;

            // update our line fit statistics
            Mx += bestx;
            My += besty;
            Mxx += bestx*bestx;
            Mxy += bestx*besty;
            Myy += besty*besty;
            N++;
        }

        // fit a line
        double Ex = Mx / N, Ey = My / N;
        double Cxx = Mxx / N - Ex*Ex;
        double Cxy = Mxy / N - Ex*Ey;
        double Cyy = Myy / N - Ey*Ey;

        // TODO: Can replace this with same code as in fit_line.
        double normal_theta = .5 * atan2f(-2*Cxy, (Cyy - Cxx));
        nx = cosf(normal_theta);
        ny = sinf(normal_theta);
        lines[edge][0] = Ex;
        lines[edge][1] = Ey;
        lines[edge][2] = nx;
        lines[edge][3] = ny;
    }

    // now refit the corners of the quad
    for (int i = 0; i < 4; i++) {

        // solve for the intersection of lines (i) and (i+1)&3.
        double A00 =  lines[i][3],  A01 = -lines[(i+1)&3][3];
        double A10 =  -lines[i][2],  A11 = lines[(i+1)&3][2];
        double B0 = -lines[i][0] + lines[(i+1)&3][0];
        double B1 = -lines[i][1] + lines[(i+1)&3][1];

        double det = A00 * A11 - A10 * A01;

        // inverse.
        if (fabs(det) > 0.001) {
            // solve
            double W00 = A11 / det, W01 = -A01 / det;

            double L0 = W00*B0 + W01*B1;

            // Compute intersection. Note that line i represents the line from corner i to (i+1)&3, so
	    // the intersection of line i with line (i+1)&3 represents corner (i+1)&3.
            quad->p[(i+1)&3][0] = lines[i][0] + L0*A00;
            quad->p[(i+1)&3][1] = lines[i][1] + L0*A10;
        } else {
            // this is a bad sign. We'll just keep the corner we had.
//            debug_print("bad det: %15f %15f %15f %15f %15f\n", A00, A11, A10, A01, det);
        }
    }
}

__device__ void quad_decode_task_cuda(apriltag_detector_cuda_t *td, image_u8_cuda_t *im, int i0, int i1, zarray_cuda_t *quads, image_u8_cuda_t *im_samples, zarray_cuda_t *detections)
{
    __shared__ bool add_lock;
    if (threadIdx.x == 0) {
        add_lock = false;
    }
    __syncthreads();

    for (int quadidx = i0; quadidx < i1; quadidx++) {
        struct quad_cuda *quad_original;
        zarray_get_volatile_cuda(quads, quadidx, &quad_original);

        // refine edges is not dependent upon the tag family, thus
        // apply this optimization BEFORE the other work.
        //if (td->quad_decimate > 1 && td->refine_edges) {
        if (td->refine_edges) {
            refine_edges_cuda(td, im, quad_original);
        }

        // make sure the homographies are computed...
        if (quad_update_homographies_cuda(quad_original) != 0)
            continue;

        for (int famidx = 0; famidx < zarray_size_cuda(td->tag_families); famidx++) {
            apriltag_family_cuda_t *family;
            zarray_get_cuda(td->tag_families, famidx, &family);

            if (family->reversed_border != quad_original->reversed_border) {
                continue;
            }

            // since the geometry of tag families can vary, start any
            // optimization process over with the original quad.
            struct quad_cuda *quad = quad_copy_cuda(quad_original);

            struct quick_decode_entry entry;

            float decision_margin = quad_decode_cuda(td, family, im, quad, &entry, im_samples);

            if (decision_margin >= 0 && entry.hamming < 255) {
                apriltag_detection_cuda_t *det = (apriltag_detection_cuda_t *) calloc_cuda(1, sizeof(apriltag_detection_cuda_t));

                det->family = family;
                det->id = entry.id;
                det->hamming = entry.hamming;
                det->decision_margin = decision_margin;

                double theta = entry.rotation * M_PI / 2.0;
                double c = cos(theta), s = sin(theta);

                // Fix the rotation of our homography to properly orient the tag
                matd_cuda_t *R = matd_create_cuda(3,3);
                MATD_EL_CUDA(R, 0, 0) = c;
                MATD_EL_CUDA(R, 0, 1) = -s;
                MATD_EL_CUDA(R, 1, 0) = s;
                MATD_EL_CUDA(R, 1, 1) = c;
                MATD_EL_CUDA(R, 2, 2) = 1;

                det->H = matd_op_cuda("M*M", quad->H, R);

                matd_destroy_cuda(R);

                homography_project_cuda(det->H, 0, 0, &det->c[0], &det->c[1]);

                // [-1, -1], [1, -1], [1, 1], [-1, 1], Desired points
                // [-1, 1], [1, 1], [1, -1], [-1, -1], FLIP Y
                // adjust the points in det->p so that they correspond to
                // counter-clockwise around the quad, starting at -1,-1.
                for (int i = 0; i < 4; i++) {
                    int tcx = (i == 1 || i == 2) ? 1 : -1;
                    int tcy = (i < 2) ? 1 : -1;

                    double p[2];

                    homography_project_cuda(det->H, tcx, tcy, &p[0], &p[1]);

                    det->p[i][0] = p[0];
                    det->p[i][1] = p[1];
                }

                while (!add_lock) {
                    add_lock = true;
                    zarray_add_cuda(detections, &det);
                    add_lock = false;
                }
                // pthread_mutex_lock(&td->mutex);
                // zarray_add_cuda(task->detections, &det);
                // pthread_mutex_unlock(&td->mutex);
            }

            quad_destroy_cuda(quad);
        }
    }
}

__device__ void apriltag_detection_destroy_cuda(apriltag_detection_cuda_t *det)
{
    if (det == NULL)
        return;

    matd_destroy_cuda(det->H);
    free(det);
}

__device__ static int prefer_smaller(int pref, double q0, double q1)
{
    if (pref)     // already prefer something? exit.
        return pref;

    if (q0 < q1)
        return -1; // we now prefer q0
    if (q1 < q0)
        return 1; // we now prefer q1

    // no preference
    return 0;
}

// djb2
__device__ uint32_t compute_image_hash(image_u8_cuda_t *im) 
{
    unsigned long hash = 5381;
    int c;

    for (int i = 0; i < im->stride * im->height; i++) {
        c = im->buf[i];
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }

    return hash;
}

// __device__ uint32_t compute_image8x3_hash_cuda(image_u8x3_cuda_t *im) 
// {
//     unsigned long hash = 5381;
//     int c;

//     for (int i = 0; i < im->stride * im->height; i++) {
//         c = im->buf[i];
//         hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
//     }

//     return hash;
// }

// __device__ uint32_t compute_buf_hash(void *buf, uint32_t buf_size) // TODO: deal with other hash func
// {
//     register uint8_t *ubuf = (uint8_t *) buf;
//     unsigned long hash = 5381;
//     int c;

//     for (int i = 0; i < buf_size; i++) {
//         c = ubuf[i];
//         hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
//     }

//     return hash;

// }

__global__ void apriltag_detector_detect_cuda(apriltag_detector_cuda_t *td, image_u8_cuda_t *im_orig, int32_t num_threads, zarray_cuda_t **out, uint8_t *dbg_im_buf, int32_t *dbg_im_w, int32_t *dbg_im_h, int32_t *dbg_im_s)
{
    __shared__ bool abort;
    __shared__ image_u8_cuda_t *quad_im;
    __shared__ apriltag_family_cuda_t *fam;

    if (threadIdx.x == 0) {
        abort = false;

        printf("GPU: Started kernel with %d threads\n", num_threads);

        td->tag_families = zarray_create_cuda(sizeof(apriltag_family_cuda_t*));

        printf("GPU: Detector decode sharpening: %f, quad_decimate: %f\n", td->decode_sharpening, td->quad_decimate);

        uint32_t im_buf_hash = compute_image_hash(im_orig);
        printf("GPU: im_orig width: %d, stride: %d, height %d, buf: 0x%X\n", im_orig->width, im_orig->stride, im_orig->height, im_buf_hash);

        fam = tag36h11_create_cuda();

        apriltag_detector_add_family_cuda(td, fam);

        if (zarray_size_cuda(td->tag_families) == 0) {
            printf("No tag families enabled\n");
            abort = true;
        }

        quad_im = im_orig;

        ///////////////////////////////////////////////////////////
        // Step 1. Detect quads according to requested image decimation
        // and blurring parameters.
        if (td->quad_decimate > 1) {
            quad_im = image_u8_decimate_cuda(im_orig, td->quad_decimate);
        }
    
        if (td->quad_sigma != 0) {
            // compute a reasonable kernel width by figuring that the
            // kernel should go out 2 std devs.
            //
            // max sigma          ksz
            // 0.499              1  (disabled)
            // 0.999              3
            // 1.499              5
            // 1.999              7

            float sigma = fabsf((float) td->quad_sigma);

            int ksz = 4 * sigma; // 2 std devs in each direction
            if ((ksz & 1) == 0)
                ksz++;

            if (ksz > 1) {

                if (td->quad_sigma > 0) {
                    // Apply a blur
                    image_u8_gaussian_blur_cuda(quad_im, sigma, ksz);
                } else {
                    // SHARPEN the image by subtracting the low frequency components.
                    image_u8_cuda_t *orig = image_u8_copy_cuda(quad_im);
                    image_u8_gaussian_blur_cuda(quad_im, sigma, ksz);

                    for (int y = 0; y < orig->height; y++) {
                        for (int x = 0; x < orig->width; x++) {
                            int vorig = orig->buf[y*orig->stride + x];
                            int vblur = quad_im->buf[y*quad_im->stride + x];

                            int v = 2*vorig - vblur;
                            if (v < 0)
                                v = 0;
                            if (v > 255)
                                v = 255;

                            quad_im->buf[y*quad_im->stride + x] = (uint8_t) v;
                        }
                    }
                    image_u8_destroy_cuda(orig);
                }
            }
        }

    }

    __syncthreads();
    
    image_u8x3_cuda_t *dbg;

    zarray_cuda_t *quads = apriltag_quad_thresh_cuda(td, quad_im, num_threads, &dbg);

    if (threadIdx.x == 0) {
        uint32_t dbg_im_hash = compute_image8x3_hash_cuda(dbg);
        printf("GPU: Returning segmentation image, width: %d, stride: %d, height %d, buf: 0x%X\n", dbg->width, dbg->stride, dbg->height, dbg_im_hash);

        uint32_t buf_size = dbg->stride * dbg->height;
        memcpy(dbg_im_buf, dbg->buf, buf_size);

        *dbg_im_w = dbg->width;
        *dbg_im_h = dbg->height;
        *dbg_im_s = dbg->stride;

        for (int32_t i = 0; i < zarray_size_cuda(td->tag_families); i++) {
            apriltag_family_cuda_t f;
            zarray_get_cuda(td->tag_families, i, &f);
            tag36h11_destroy_cuda(&f);
        }
    }

    return;
    // adjust centers of pixels so that they correspond to the
    // original full-resolution image.
    if (td->quad_decimate > 1) {
        for (int i = 0; i < zarray_size_cuda(quads); i++) {
            struct quad_cuda *q;
            zarray_get_volatile_cuda(quads, i, &q);

            for (int j = 0; j < 4; j++) {
                q->p[j][0] *= td->quad_decimate;
                q->p[j][1] *= td->quad_decimate;
            }
        }
    }

    if (quad_im != im_orig)
        image_u8_destroy_cuda(quad_im);

    zarray_cuda_t *detections = zarray_create_cuda(sizeof(apriltag_detection_cuda_t*));

    td->nquads = zarray_size_cuda(quads);


    ////////////////////////////////////////////////////////////////
    // Step 2. Decode tags from each quad.
    if (1) {
        image_u8_cuda_t *im_samples = td->debug ? image_u8_copy_cuda(im_orig) : NULL;

        int chunksize = 1 + zarray_size_cuda(quads) / (APRILTAG_TASKS_PER_THREAD_TARGET * td->nthreads);

        // struct quad_decode_task *tasks = (struct quad_decode_task *) malloc(sizeof(struct quad_decode_task)*(zarray_size_cuda(quads) / chunksize + 1));

        // int ntasks = 0;
        // for (int i = 0; i < zarray_size_cuda(quads); i+= chunksize) {
        //     tasks[ntasks].i0 = i;
        //     tasks[ntasks].i1 = imin_cuda(zarray_size_cuda(quads), i + chunksize);
        //     tasks[ntasks].quads = quads;
        //     tasks[ntasks].td = td;
        //     tasks[ntasks].im = im_orig;
        //     tasks[ntasks].detections = detections;

        //     tasks[ntasks].im_samples = im_samples;

        //     workerpool_add_task(td->wp, quad_decode_task, &tasks[ntasks]);
        //     ntasks++;
        // }

        // workerpool_run(td->wp);

        // free(tasks);

        int32_t quads_len = zarray_size_cuda(quads);
        int32_t chunk_size, row_start, row_end;
        if (quads_len >= num_threads) {
            chunk_size = 1;
        } else {
            chunk_size = quads_len / num_threads;
            row_start = chunk_size * threadIdx.x;
            row_end = imin_cuda(row_start + chunk_size, quads_len);
            if (threadIdx.x == num_threads - 1) {
                row_end = quads_len;
            }
        }

        __syncthreads();

        quad_decode_task_cuda(td, im_orig, row_start, row_end, quads, im_samples, detections);
        
        __syncthreads();

        if (im_samples != NULL) {
            // image_u8_write_pnm(im_samples, "debug_samples.pnm");
            image_u8_destroy_cuda(im_samples);
        }
    }

    ////////////////////////////////////////////////////////////////
    // Step 3. Reconcile detections--- don't report the same tag more
    // than once. (Allow non-overlapping duplicate detections.)
    if (1) {
        zarray_cuda_t *poly0 = g2d_polygon_create_zeros_cuda(4);
        zarray_cuda_t *poly1 = g2d_polygon_create_zeros_cuda(4);

        for (int i0 = 0; i0 < zarray_size_cuda(detections); i0++) {

            apriltag_detection_cuda_t *det0;
            zarray_get_cuda(detections, i0, &det0);

            for (int k = 0; k < 4; k++)
                zarray_set_cuda(poly0, k, det0->p[k], NULL);

            for (int i1 = i0+1; i1 < zarray_size_cuda(detections); i1++) {

                apriltag_detection_cuda_t *det1;
                zarray_get_cuda(detections, i1, &det1);

                if (det0->id != det1->id || det0->family != det1->family)
                    continue;

                for (int k = 0; k < 4; k++)
                    zarray_set_cuda(poly1, k, det1->p[k], NULL);

                if (g2d_polygon_overlaps_polygon_cuda(poly0, poly1)) {
                    // the tags overlap. Delete one, keep the other.

                    int pref = 0; // 0 means undecided which one we'll keep.
                    pref = prefer_smaller(pref, det0->hamming, det1->hamming);     // want small hamming
                    pref = prefer_smaller(pref, -det0->decision_margin, -det1->decision_margin);      // want bigger margins

                    // if we STILL don't prefer one detection over the other, then pick
                    // any deterministic criterion.
                    for (int i = 0; i < 4; i++) {
                        pref = prefer_smaller(pref, det0->p[i][0], det1->p[i][0]);
                        pref = prefer_smaller(pref, det0->p[i][1], det1->p[i][1]);
                    }

                    if (pref == 0) {
                        // at this point, we should only be undecided if the tag detections
                        // are *exactly* the same. How would that happen?
                        printf("uh oh, no preference for overlappingdetection\n");
                    }

                    if (pref < 0) {
                        // keep det0, destroy det1
                        apriltag_detection_destroy_cuda(det1);
                        zarray_remove_index_cuda(detections, i1, 1);
                        i1--; // retry the same index
                        goto retry1;
                    } else {
                        // keep det1, destroy det0
                        apriltag_detection_destroy_cuda(det0);
                        zarray_remove_index_cuda(detections, i0, 1);
                        i0--; // retry the same index.
                        goto retry0;
                    }
                }

              retry1: ;
            }

          retry0: ;
        }

        zarray_destroy_cuda(poly0);
        zarray_destroy_cuda(poly1);
    }


    for (int i = 0; i < zarray_size_cuda(quads); i++) {
        struct quad_cuda *quad;
        zarray_get_volatile_cuda(quads, i, &quad);
        matd_destroy_cuda(quad->H);
        matd_destroy_cuda(quad->Hinv);
    }

    zarray_destroy_cuda(quads);

    for (int i = 0; i < zarray_size_cuda(detections); i++) {
        apriltag_detection_cuda_t *det;
        zarray_get_cuda(detections, i, det); 
        matd_destroy_cuda(det->H);
        
    }

    // TODO: Re-implement this for CUDA
    // zarray_sort_cuda(detections, detection_compare_function_cuda);

    *out = detections;

    if (threadIdx.x == 0) {
        printf("Finished kernel");
    }
}


// Call this method on each of the tags returned by apriltag_detector_detect
__device__ void apriltag_detections_destroy_cuda(zarray_cuda_t *detections)
{
    for (int i = 0; i < zarray_size_cuda(detections); i++) {
        apriltag_detection_cuda_t *det;
        zarray_get_cuda(detections, i, &det);

        apriltag_detection_destroy_cuda(det);
    }

    zarray_destroy_cuda(detections);
}

__device__ image_u8_cuda_t *apriltag_to_image_cuda(apriltag_family_cuda_t *fam, uint32_t idx)
{
    assert(fam != NULL);
    assert(idx < fam->ncodes);

    uint64_t code = fam->codes[idx];

    image_u8_cuda_t *im = image_u8_create_cuda(fam->total_width, fam->total_width);

    int white_border_width = fam->width_at_border + (fam->reversed_border ? 0 : 2);
    int white_border_start = (fam->total_width - white_border_width)/2;
    // Make 1px white border
    for (int i = 0; i < white_border_width - 1; i += 1) {
        im->buf[white_border_start*im->stride + white_border_start + i] = 255;
        im->buf[(white_border_start + i)*im->stride + fam->total_width - 1 - white_border_start] = 255;
        im->buf[(fam->total_width - 1 - white_border_start)*im->stride + white_border_start + i + 1] = 255;
        im->buf[(white_border_start + 1 + i)*im->stride + white_border_start] = 255;
    }

    int border_start = (fam->total_width - fam->width_at_border)/2;
    for (uint32_t i = 0; i < fam->nbits; i++) {
        if (code & (APRILTAG_U64_ONE << (fam->nbits - i - 1))) {
            im->buf[(fam->bit_y[i] + border_start)*im->stride + fam->bit_x[i] + border_start] = 255;
        }
    }
    return im;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// END CUDA version
//
///////////////////////////////////////////////////////////////////////////////////////////////////////
