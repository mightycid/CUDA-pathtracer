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

#ifndef __LIGHT_H__
#define __LIGHT_H__

struct VisibilityTester  {
	CUDA_DEVICE void SetSegment(const Point &p1, float eps1, const Point &p2, float eps2) {
		float dist = p1.Distance(p2);
		mint = eps1;
		maxt = dist * (1.f - eps2);
		r = Ray(p1, (p2 - p1) / dist);
	}
	Ray r;
	float mint, maxt;
};

enum LightType { POINT_LIGHT, AREA_LIGHT };

class Light {
public:
	Light(const Point &p, const Color &l)
		: type(POINT_LIGHT), pos(p), intensity(l) {}
	Light(uint32_t pId, const Color &l)
		: type(AREA_LIGHT), primId(pId), intensity(l) {}

	CUDA_DEVICE Color L(const Point &p, const Vec &w, const Vec &n) const {
		return n.Dot(-w) > 0.f ? intensity : Color();
	}

	LightType type;
	union {
		struct { Point pos; };
		struct { uint32_t primId; };
	};
	Color intensity;
};

/**
 * Provides a list of lights on the device memory
 */
struct LightList {
	LightList(Light* l, uint32_t s) : lights(l), size(s) {}

	CUDA_DEVICE Light* operator[] (uint32_t index) const { return &lights[index]; }

	Light* lights;
	uint32_t size;
};

#endif /* __LIGHT_H__ */