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

#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "globals.h"

class Camera {
public:
	Camera() {}
	Camera(const Point &eye, const Point &lookat, const Vec &up, uint32_t w, uint32_t h, float lr, float fd, float fov)
			: width(w), height(h), pos(eye), lensRadius(lr), focalDistance(fd), fov(fov) {
		
		float dist;
		Vec dir = (lookat - eye).Normalize(&dist);
		u = (dir.Cross(up)).Normalize();
		v = u.Cross(dir);

		float aspect = (float)height / (float)width;
		float hfov = fov;
		float vfov = hfov*aspect;
		
		dist *= 2.f;
		float magnitude = dist * tanf(Radians(hfov*0.5f)) / width;
		u *= magnitude;

		magnitude = dist * tanf(Radians(vfov*0.5f)) / height;
		v *= magnitude;

		Vec offset = v * (float)(height * 0.5f) - u * (float)(width * 0.5f);
		firstRayDir = (lookat - eye) + offset;
		updated = true;
	}
	~Camera() {}

	CUDA_DEVICE Ray GenerateRay(int x, int y, const UVSample &sample = UVSample()) const {
		//TODO depth of field
		return Ray(pos, (firstRayDir - v * ((float)y + sample.v) + u * ((float)x + sample.u)).Normalize());
	}

	CUDA_HOST_DEVICE uint32_t Width() const { return width; }
	CUDA_HOST_DEVICE uint32_t Height() const { return height; }

	void Translate(const Vec &delta) {
		pos += delta;
		Updated();
	}

	void Updated() { updated = true; }
	bool IsUpdated() { const bool b = updated; updated = false; return b; }


private:
	uint32_t width;
	uint32_t height;

	Matrix mat;
	Point pos;
	Vec u, v;
	Vec firstRayDir;
	float focalDistance;
	float lensRadius;
	float fov;
	bool updated;
};

#endif /* __CAMERA_H__ */