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

/**
 * Represents a sphere - will be changed to triangle later
 */
class Primitive {
public:
	Primitive(const Point &p, float rad, uint32_t matId) : pos(p),  materialId(matId), radius(rad) {}

	CUDA_DEVICE float Intersect(const Ray &ray, float tmin, float tmax) const {
		Vec op = (pos-ray.o);
		float t, b=op.Dot(ray.d), det=b*b-op.Dot(op)+radius*radius;
		if(det<0.f) return -1.f; else det=sqrtf(det);
		//t = (t=b-det)>tmin ? t : ((t=b+det)>tmin ? t : -1.f);
		return (t=b-det)>tmin ? (t<tmax ? t : -1.f) : ((t=b+det)>tmin ? (t<tmax ? t : -1.f) : -1.f);
	}
	CUDA_DEVICE bool IntersectP(const Ray &ray, float tmin, float tmax) const {
		//TODO maybe there is a faster hit test for spheres :(
		return Intersect(ray, tmin, tmax) > 0.f;
	}
	CUDA_DEVICE Vec GetNormal(const Point &p) const { return (p-pos)/radius; }
	CUDA_DEVICE uint32_t GetMaterialId() const { return materialId; }

private:
	Point pos;
	uint32_t materialId;
	float radius;
};

/**
 * Provides a list of primitives on the device memory
 */
struct PrimitiveList {
	PrimitiveList() : prims(NULL), size(0) {}
	PrimitiveList(Primitive* p, uint32_t s) : prims(p), size(s) {}
	CUDA_DEVICE Primitive& operator[] (uint32_t index) const { return prims[index]; }

	Primitive* prims;
	uint32_t size;
};

#endif /* __PRIMITIVE_H__ */