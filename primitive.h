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

#ifndef __PRIMITIVE_H__
#define __PRIMITIVE_H__

#include <stdlib.h>

/**
 * Represents a sphere - will be changed to triangle later
 */
class Primitive {
public:
	Primitive(const Point &p, float rad, uint32_t matId) : pos(p),  materialId(matId), radius(rad), lightId(-1) {}
	Primitive(const Point &p, float rad, uint32_t matId, int lightId) : pos(p),  materialId(matId), radius(rad), lightId(lightId) {}

	CUDA_DEVICE float Intersect(const Ray &ray, float tmin=EPSILON, float tmax=INF) const {
		Vec op = (pos-ray.o);
		float t, b=op.Dot(ray.d), det=b*b-op.Dot(op)+radius*radius;
		if(det<0.f) return -1.f; else det=sqrtf(det);
		return (t=b-det)>tmin ? (t<tmax ? t : -1.f) : ((t=b+det)>tmin ? (t<tmax ? t : -1.f) : -1.f);
	}
	CUDA_DEVICE bool IntersectP(const Ray &ray, float tmin, float tmax) const {
		//TODO maybe there is a faster hit test for spheres :(
		return Intersect(ray, tmin, tmax) > 0.f;
	}
	CUDA_DEVICE Point Sample(const UVSample &sample, Vec *ns) const {
		Point p = pos + UniformSampleSphere(sample.u, sample.v) * radius;
		*ns = GetNormal(p);
		return p;
	}
	CUDA_DEVICE Point Sample(const Point &p, const UVSample &sample, Vec *ns) const {
		Vec wc = (pos - p).Normalize();
		Vec wcX, wcY;
		CoordinateSystem(wc, &wcX, &wcY);
		float dist2 = p.DistanceSquared(pos);
		if(dist2 - radius*radius < 1e-4f)
			return Sample(sample, ns);
		float sinThetaMax2 = radius*radius / dist2;
		float cosThetaMax = sqrtf(max(0.f, 1.f - sinThetaMax2));
		float thit;
		Point ps;
		Ray r(p, UniformSampleCone(sample.u, sample.v, cosThetaMax, wcX, wcY, wc));
		if(thit = Intersect(r) > 0.f)
			thit = (pos-p).Dot(r.d.Normalize());
		ps = r(thit);
		*ns = (ps-pos).Normalize();
		return ps;
	}
	CUDA_DEVICE Vec GetNormal(const Point &p) const { return (p-pos)/radius; }
	CUDA_DEVICE bool IsLight() const { return (lightId > -1); }

	Point pos;
	uint32_t materialId;
	float radius;
	int lightId;
};

/**
 * Provides a list of primitives on the device memory
 */
struct PrimitiveList {
	PrimitiveList(Primitive* p, uint32_t s) : primitives(p), size(s) {}

	CUDA_DEVICE Primitive* operator[] (uint32_t index) const {
		return &primitives[index];
	}

	Primitive* primitives;
	uint32_t size;
};

#endif /* __PRIMITIVE_H__ */