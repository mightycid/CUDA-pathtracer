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

/**
 * Perspective camera class with the possibility to controll the camera view
 */
class Camera {
public:
	Camera() {}
	Camera(const Point &eye, const Point &lookAtP, const Vec &up, uint32_t iw, 
			uint32_t ih, float lr, float fd, float fov)
			: width(iw), height(ih), pos(eye), lensRadius(lr), focalDistance(fd) {
		const float aspect = (float)height / (float)width;
		hfov = fov;
		vfov = hfov*aspect;

		const Vec lookAt = (lookAtP - eye);
		dist = lookAt.Length();

		// calculate orthogonal values for camera view matrix
		const Vec w = lookAt/dist;
		const Vec v = (up - up.Dot(w) * w).Normalize();
		const Vec u = (w.Cross(v)).Normalize();

		// create view matrix
		// without translation parameters because all values are relative
		// to the position of the camera
		view = Matrix(
			u.x, u.y, u.z, 0.f,
			v.x, v.y, v.z, 0.f,
			w.x, w.y, w.z, 0.f,
			0.f, 0.f, 0.f, 1.f
		);

		UpdateParameter();
	}
	~Camera() {}

	/**
	 * Generate a ray through image plane at position px,py and a possible
	 * sub shift given through the sample
	 * if no sample is given a sample with the shift of zero is generated
	 * which leads to no shift of the original ray
	 */
	CUDA_DEVICE Ray GenerateRay(int px, int py, 
		const UVSample &sample = UVSample()) const {
		//TODO depth of field
		float sx = ((float)px + sample.u);
		float sy = ((float)py + sample.v);
		return Ray(pos, (firstRayDir - pxY*sy + pxX*sx).Normalize());
	}

	/**
	 * Translates the camera in given delta values respresented
	 * through the vector delta
	 * The translation is relative to the current view direction
	 */
	void Translate(const Vec &delta) {
		const Vec u = view.GetRow(0);
		const Vec v = view.GetRow(1);
		const Vec w = view.GetRow(2);

		// translate in respect to each relative axis
		pos += u*delta.x + v*delta.y + w*delta.z;

		// the image plane values don't change so we don't need
		// to call the UpdateParameter() method here
		Updated();
	}

	/**
	 * Rotates the camera in given angels represented through
	 * the vector theta
	 * We allow only rotation in x and y direction for now
	 */
	void Rotate(const Vec &theta) {
		// we ignore the z rotation for now
		const float ctx = theta.x == 0.f ? 1.f : cosf(theta.x);
		const float cty = theta.y == 0.f ? 1.f : cosf(theta.y);
		const float stx = theta.x == 0.f ? 0.f : sinf(theta.x);
		const float sty = theta.y == 0.f ? 0.f : sinf(theta.y);
		
		// x-axis rotation matrix
		Matrix rx = Matrix(
			1.f, 0.f, 0.f,  0.f,
			0.f, cty, -sty, 0.f,
			0.f, sty, cty,  0.f,
			0.f, 0.f, 0.f,  1.f
		);

		// y-axis rotation matrix
		Matrix ry = Matrix(
			ctx,  0.f, stx, 0.f,
			0.f,  1.f, 0.f, 0.f,
			-stx, 0.f, ctx, 0.f,
			0.f,  0.f, 0.f, 1.f
		);

		// multiply rotation matrices with the current view
		if(theta.x == 0.f)
			view = rx * view;
		else if(theta.y == 0.f)
			view = ry * view;
		else
			view = rx * ry * view;

		UpdateParameter();
	}

	/**
	 * Checks if the camera has changed and set the indicator back to false
	 */
	bool IsUpdated() { const bool b = updated; updated = false; return b; }

	CUDA_HOST_DEVICE uint32_t Width() const { return width; }
	CUDA_HOST_DEVICE uint32_t Height() const { return height; }
	CUDA_HOST_DEVICE Point GetPos() const { return pos; }

private:
	/**
	 * Updates the image plane parameters
	 */
	void UpdateParameter() {
		const Vec u = view.GetRow(0);
		const Vec v = view.GetRow(1);
		const Vec w = view.GetRow(2);

		float magnitude = dist * 2.f * tanf(Radians(hfov*0.5f)) / width;
		pxX = u * magnitude;
		magnitude = dist * 2.f * tanf(Radians(vfov*0.5f)) / height;
		pxY = v * magnitude;

		Vec offset = pxY * (float)(height * 0.5f) - pxX * (float)(width * 0.5f);
		firstRayDir = w*dist + offset;

		Updated();
	}

	/**
	 * Sets the changed indicator to true
	 */
	void Updated() { updated = true; }

	uint32_t width;			// pixel width of rendered image
	uint32_t height;		// pixel height of renderer image

	Matrix view;			// view matrix
	Point pos;				// current position
	Vec pxX, pxY;			// ray shift for one pixel in image coordinates
	Vec firstRayDir;		// position of first on image plane
	float dist;				// distance to image plane
	float focalDistance;	// focal distance for DOF
	float lensRadius;		// lens radius for DOF
	float hfov, vfov;		// horizontal and vertical field of view
	bool updated;			// indicator if camera parameters did change
};

#endif /* __CAMERA_H__ */