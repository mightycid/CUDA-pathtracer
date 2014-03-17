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

#ifndef __MONTECARLO_H__
#define __MONTECARLO_H__

#include "globals.h"

/**
 * From the book "Phisically Based Rendering" (pbrt.org)
 */
//struct Distribution1D {
//    Distribution1D(const float *f, int n) {
//		count = n;
//		float *hostFunc = new float[n];
//		memcpy(func, f, n*sizeof(float));
//		float *hostCdf = new float[n+1];
//		cdf[0] = 0.f;
//		for(int i=1; i<count+1; ++i)
//			hostCdf[i] = hostCdf[i-1] + hostFunc[i-1] / n;
//
//		funcInt = hostCdf[count];
//		if(funcInt == 0.f) {
//			for(int i=1; i<n+1; ++i)
//				hostCdf[i] = float(i) / float(n);
//		} else {
//			for(int i=1; i<n+1; ++i)
//				hostCdf[i] /= funcInt;
//		}
//		CudaSafeCall(CudaMalloc((void**)&(func), &hostFunc, n*sizeof(float)));
//		CudaSafeCall(CudaMalloc((void**)&(cdf), &hostCdf, (n+1)*sizeof(float)));
//		delete[] hostCdf;
//		delete[] hostFunc;
//	}
//    inline CUDA_DEVICE float SampleContinuous(float u, float *pdf, int *off) const {
//		float *ptr = std::lower_bound(cdf, cdf+count+1, u);
//		int offset = max(0, int(ptr-cdf-1));
//		if(off) *off = offset;
//		float du = (u - cdf[offset]) / (cdf[offset+1] - cdf[offset]);
//		if(pdf) *pdf = func[offset] / funcInt;
//		return (offset + du) / count;
//	}
//    inline CUDA_DEVICE int SampleDiscrete(float u, float *pdf) const {
//		float *ptr = std::lower_bound(cdf, cdf+count+1, u);
//		int offset = max(0, int(ptr-cdf-1));
//		if(pdf) *pdf = func[offset] / (funcInt * count);
//		return offset;
//	}
//    ~Distribution1D() {
//        CudaFree(func);
//        CudaFree(cdf);
//    }
//
//private:
//    float *func, *cdf;
//    float funcInt;
//    int count;
//};

inline CUDA_DEVICE void ConcentricSampleDisk(float u1, float u2, float *dx, float *dy) {
	float r, theta;
	// Map uniform random numbers to $[-1,1]^2$
	float sx = 2 * u1 - 1;
	float sy = 2 * u2 - 1;

	// Map square to $(r,\theta)$

	// Handle degeneracy at the origin
	if (sx == 0.f && sy == 0.f) {
		*dx = 0.f;
		*dy = 0.f;
		return;
	}
	if (sx >= -sy) {
		if (sx > sy) {
			// Handle first region of disk
			r = sx;
			if (sy > 0.f) theta = sy / r;
			else          theta = 8.f + sy / r;
		}
		else {
			// Handle second region of disk
			r = sy;
			theta = 2.f - sx / r;
		}
	}
	else {
		if (sx <= sy) {
			// Handle third region of disk
			r = -sx;
			theta = 4.f - sy / r;
		}
		else {
			// Handle fourth region of disk
			r = -sy;
			theta = 6.f + sx / r;
		}
	}
	theta *= M_PI / 4.f;
	*dx = r * cosf(theta);
	*dy = r * sinf(theta);
}

inline CUDA_DEVICE Vec RotateByNormal(const Vec &wo, const Vec &n) {
	Vec w(n);
	Vec u = fabs(n.x)>fabs(n.z) ? Vec(-n.y, n.x, 0.f) : Vec(0.f, -n.z, n.y);
	Vec v = w.Cross(u);
	return (u*wo.x + v*wo.y + w*wo.z).Normalize();
}

inline CUDA_DEVICE Vec CosineSampleHemisphere(float u1, float u2, const Vec &n) {
	Vec ret;
	ConcentricSampleDisk(u1, u2, &ret.x, &ret.y);
	ret.z = sqrtf(max(0.f, 1.f - ret.x*ret.x - ret.y*ret.y));

	return RotateByNormal(ret, n);
}

inline Vec CUDA_DEVICE UniformSampleSphere(float u1, float u2) {
    float z = 1.f - 2.f * u1;
    float r = sqrtf(max(0.f, 1.f - z*z));
    float phi = 2.f * M_PI * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    return Vec(x, y, z);
}

inline CUDA_DEVICE Vec UniformSampleCone(float u1, float u2, float costhetamax,
        const Vec &x, const Vec &y, const Vec &z) {
    float costheta = Lerp(u1, costhetamax, 1.f);
    float sintheta = sqrtf(1.f - costheta*costheta);
    float phi = u2 * 2.f * M_PI;
    return x * cosf(phi) * sintheta + y * sinf(phi) * sintheta + z * costheta;
}

inline CUDA_DEVICE float UniformConePdf(float cosThetaMax) {
    return 1.f / (2.f * M_PI * (1.f - cosThetaMax));
}

inline CUDA_DEVICE float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
	float f = nf * fPdf, g = ng * gPdf;
	return (f*f) / (f*f + g*g);
}

#endif /* __MONTECARLO_H__ */