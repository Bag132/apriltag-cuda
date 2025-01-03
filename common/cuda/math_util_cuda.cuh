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

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#define to_radians(x) ( (x) * (M_PI / 180.0 ))
#define to_degrees(x) ( (x) * (180.0 / M_PI ))

  /* DEPRECATE, threshold meaningless without context.
static inline int dequals(double a, double b)
{
    double thresh = 1e-9;
    return (fabs(a-b) < thresh);
}
  */

__device__ static inline int dequals_mag_cuda(double a, double b, double thresh)
{
    return (fabs(a-b) < thresh);
}

__device__ static inline int isq_cuda(int v)
{
    return v*v;
}

__device__ static inline float fsq_cuda(float v)
{
    return v*v;
}

__device__ static inline double sq_cuda(double v)
{
    return v*v;
}

__device__ static inline double sgn_cuda(double v)
{
    return (v>=0) ? 1 : -1;
}

// random number between [0, 1)
__device__ static inline float randf_cuda()
{
    return (float)(rand() / (RAND_MAX + 1.0));
}


__device__ static inline float signed_randf_cuda()
{
    return randf_cuda()*2 - 1;
}

// return a random integer between [0, bound)
__device__ static inline int irand_cuda(int bound)
{
    int v = (int) (randf_cuda()*bound);
    if (v == bound)
        return (bound-1);
    //assert(v >= 0);
    //assert(v < bound);
    return v;
}

/** Map vin to [0, 2*PI) **/
__device__ static inline double mod2pi_positive_cuda(double vin)
{
    return vin - M_2_PI * floor(vin / M_2_PI);
}

/** Map vin to [-PI, PI) **/
__device__ static inline double mod2pi_cuda(double vin)
{
    return mod2pi_positive_cuda(vin + M_PI) - M_PI;
}

/** Return vin such that it is within PI degrees of ref **/
__device__ static inline double mod2pi_ref_cuda(double ref, double vin)
{
    return ref + mod2pi_cuda(vin - ref);
}

/** Map vin to [0, 360) **/
__device__ static inline double mod360_positive_cuda(double vin)
{
    return vin - 360 * floor(vin / 360);
}

/** Map vin to [-180, 180) **/
__device__ static inline double mod360_cuda(double vin)
{
    return mod360_positive_cuda(vin + 180) - 180;
}

__device__ static inline int mod_positive_cuda(int vin, int mod) {
    return (vin % mod + mod) % mod;
}

__device__ static inline int theta_to_int_cuda(double theta, int max)
{
    theta = mod2pi_ref_cuda(M_PI, theta);
    int v = (int) (theta / M_2_PI * max);

    if (v == max)
        v = 0;

    assert (v >= 0 && v < max);

    return v;
}

__device__ inline int imin_cuda(int a, int b)
{
    return (a < b) ? a : b;
}

__device__ static inline int imax_cuda(int a, int b)
{
    return (a > b) ? a : b;
}

__device__ static inline int64_t imin64_cuda(int64_t a, int64_t b)
{
    return (a < b) ? a : b;
}

__device__ static inline int64_t imax64_cuda(int64_t a, int64_t b)
{
    return (a > b) ? a : b;
}

__device__ static inline int iclamp_cuda(int v, int minv, int maxv)
{
    return imax_cuda(minv, imin_cuda(v, maxv));
}

__device__ static inline double dclamp_cuda(double a, double min, double max)
{
    if (a < min)
        return min;
    if (a > max)
        return max;
    return a;
}

__device__ static inline int fltcmp_cuda (float f1, float f2)
{
    float epsilon = f1-f2;
    if (epsilon < 0.0)
        return -1;
    else if (epsilon > 0.0)
        return  1;
    else
        return  0;
}

__device__ static inline int dblcmp_cuda (double d1, double d2)
{
    double epsilon = d1-d2;
    if (epsilon < 0.0)
        return -1;
    else if (epsilon > 0.0)
        return  1;
    else
        return  0;
}
