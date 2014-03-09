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

/**
 * From the book "Phisically Based Rendering" (pbrt.org)
 */
inline CUDA_DEVICE void ConcentricSampleDisk(float u1, float u2, float *dx, float *dy) {
	float r, theta;
	// Map uniform random numbers to $[-1,1]^2$
	float sx = 2 * u1 - 1;
	float sy = 2 * u2 - 1;

	// Map square to $(r,\theta)$

	// Handle degeneracy at the origin
	if (sx == 0.0 && sy == 0.0) {
		*dx = 0.0;
		*dy = 0.0;
		return;
	}
	if (sx >= -sy) {
		if (sx > sy) {
			// Handle first region of disk
			r = sx;
			if (sy > 0.0) theta = sy / r;
			else          theta = 8.0f + sy / r;
		}
		else {
			// Handle second region of disk
			r = sy;
			theta = 2.0f - sx / r;
		}
	}
	else {
		if (sx <= sy) {
			// Handle third region of disk
			r = -sx;
			theta = 4.0f - sy / r;
		}
		else {
			// Handle fourth region of disk
			r = -sy;
			theta = 6.0f + sx / r;
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

#endif /* __MONTECARLO_H__ */