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

#include "zarray_cuda.cuh"

// This library tries to avoid needless proliferation of types.
//
// A point is a double[2]. (Note that when passing a double[2] as an
// argument, it is passed by pointer, not by value.)
//
// A polygon is a zarray_t of double[2]. (Note that in this case, the
// zarray contains the actual vertex data, and not merely a pointer to
// some other data. IMPORTANT: A polygon must be specified in CCW
// order.  It is implicitly closed (do not list the same point at the
// beginning at the end.
//
// Where sensible, it is assumed that objects should be allocated
// sparingly; consequently "init" style methods, rather than "create"
// methods are used.

////////////////////////////////////////////////////////////////////
// Lines

typedef struct
{
    // Internal representation: a point that the line goes through (p) and
    // the direction of the line (u).
    double p[2];
    double u[2]; // always a unit vector
} g2d_line_cuda_t;

// initialize a line object.
__device__ void g2d_line_init_from_points_cuda(g2d_line_cuda_t *line, const double p0[2], const double p1[2]);

// The line defines a one-dimensional coordinate system whose origin
// is p. Where is q? (If q is not on the line, the point nearest q is
// returned.
__device__ double g2d_line_get_coordinate_cuda(const g2d_line_cuda_t *line, const double q[2]);

// Intersect two lines. The intersection, if it exists, is written to
// p (if not NULL), and 1 is returned. Else, zero is returned.
__device__ int g2d_line_intersect_line_cuda(const g2d_line_cuda_t *linea, const g2d_line_cuda_t *lineb, double *p);

////////////////////////////////////////////////////////////////////
// Line Segments. line.p is always one endpoint; p1 is the other
// endpoint.
typedef struct
{
    g2d_line_cuda_t line;
    double p1[2];
} g2d_line_segment_cuda_t;

__device__ void g2d_line_segment_init_from_points_cuda(g2d_line_segment_cuda_t *seg, const double p0[2], const double p1[2]);

// Intersect two segments. The intersection, if it exists, is written
// to p (if not NULL), and 1 is returned. Else, zero is returned.
__device__ int g2d_line_segment_intersect_segment_cuda(const g2d_line_segment_cuda_t *sega, const g2d_line_segment_cuda_t *segb, double *p);

__device__ void g2d_line_segment_closest_point_cuda(const g2d_line_segment_cuda_t *seg, const double *q, double *p);
__device__ double g2d_line_segment_closest_point_distance_cuda(const g2d_line_segment_cuda_t *seg, const double *q);

////////////////////////////////////////////////////////////////////
// Polygons

__device__ zarray_cuda_t *g2d_polygon_create_data_cuda(double v[][2], int sz);

__device__ zarray_cuda_t *g2d_polygon_create_zeros_cuda(int sz);

__device__ zarray_cuda_t *g2d_polygon_create_empty_cuda();

__device__ void g2d_polygon_add_cuda(zarray_cuda_t *poly, double v[2]);

// Takes a polygon in either CW or CCW and modifies it (if necessary)
// to be CCW.
__device__ void g2d_polygon_make_ccw_cuda(zarray_cuda_t *poly);

// Return 1 if point q lies within poly.
__device__ int g2d_polygon_contains_point_cuda(const zarray_cuda_t *poly, double q[2]);

// Do the edges of the polygons cross? (Does not test for containment).
__device__ int g2d_polygon_intersects_polygon_cuda(const zarray_cuda_t *polya, const zarray_cuda_t *polyb);

// Does polya completely contain polyb?
__device__ int g2d_polygon_contains_polygon_cuda(const zarray_cuda_t *polya, const zarray_cuda_t *polyb);

// Is there some point which is in both polya and polyb?
__device__ int g2d_polygon_overlaps_polygon_cuda(const zarray_cuda_t *polya, const zarray_cuda_t *polyb);

// returns the number of points written to x. see comments.
__device__ int g2d_polygon_rasterize_cuda(const zarray_cuda_t *poly, double y, double *x);
