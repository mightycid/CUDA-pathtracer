/**
 * Copyright (C) 2014 MightyCid
 *
 * This file is part of CUDA-pathtracer <github.com/mightycid/CUDA-pathtracer>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. 
 */

#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <limits>

#define NDEBUG
#ifdef __CUDACC__
#  define CUDA_HOST __host__
#  define CUDA_DEVICE __device__
#  define CUDA_HOST_DEVICE __host__ __device__
#  define NDEBUG
#else
#  define CUDA_HOST
#  define CUDA_DEVICE
#  define CUDA_HOST_DEVICE
#endif

#ifdef M_PI
#undef M_PI
#endif
#define M_PI       3.14159265358979323846f
#define INV_PI     0.31830988618379067154f
#define INV_TWOPI  0.15915494309189533577f
#define INV_FOURPI 0.07957747154594766788f
#define EPSILON 5E-2f

#ifdef __CUDACPP__
#  define INF 0x7f800000
#else
#  define INF FLT_MAX
#endif

#define isnan(f) _isnan(f)
#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

#include "geometry.h"

#define Clamp(x, xmin, xmax) (x < xmin ? xmin : (x > xmax ? xmax : x))

#if !defined(__GNUC__) && (defined(_WIN32) || defined(_WIN64))
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
#endif

class Camera;
class Material;
class Primitive;
class Light;
class Scene;
class Pathtracer;

inline CUDA_HOST_DEVICE float Radians(float deg) {
    return ((float)M_PI/180.f) * deg;
}

inline CUDA_HOST_DEVICE float Degrees(float rad) {
    return (180.f*(float)INV_PI) * rad;
}

inline CUDA_DEVICE Vec Reflect(const Vec &wo, const Vec &n) {
	return wo - 2.f * wo.Dot(n) * n;
}

inline CUDA_DEVICE Vec Refract(const Vec &wo, const Vec &n, float eta) {
	float cost = wo.Dot(n);
	float cos2t = 1 - eta*eta * (1 - cost * cost);
	return (wo * eta - (eta * cost + sqrtf(cos2t)) * n).Normalize();
}

inline CUDA_DEVICE float reflectance(const Vec &inc, const Vec &nor, const float n1, const float n2) {
    float n = n1/n2;
    float cosI = -(nor.Dot(inc));
    float sinT2 = n*n*(1.0f-cosI*cosI);
    if (sinT2 > 1.0f) return 1.0f; // TIR
    float cosT = sqrt(1.0f-sinT2);
    float rOrth = (n1*cosI-n2*cosT) / (n1*cosI+n2*cosT);
    float rPar = (n2*cosI-n1*cosT) / (n2*cosI+n1*cosT);
    return (rOrth*rOrth+rPar*rPar) / 2.0f;
}

struct Intersection {
	CUDA_DEVICE Intersection() : prim(NULL), mat(NULL), p(Point()), n(Vec()), t(0.f) {}

	Primitive *prim;
	Material *mat;
	Point p;
	Vec n;
	float t;
};

#endif /* __GLOBALS_H__ */