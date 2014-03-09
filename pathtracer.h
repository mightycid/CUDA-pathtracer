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

#if defined(_WIN32) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <vector>

class Pathtracer {
public:
	Pathtracer(uint32_t w, uint32_t h, int nBounces) : width(w), height(h), iteration(0), maxBounces(nBounces) { }
	bool Init(const Camera &camera, const std::vector<Material> &mv, const std::vector<Primitive> &pv, const std::vector<Light> &lv);
	void Run(float *devBuffer);
	void Release();
	
	uint32_t GetIteration() const { return iteration; }

private:
	Scene *scene;			//pointer to scene object on GPU
	float *devRand;			//pointer to random numbers on GPU
	uint32_t iteration;		//current sample iteration on GPU

	uint32_t width;			//width of rendered image
	uint32_t height;		//height of rendered image
	uint32_t bufferSize;	//buffersize in pixels
	uint32_t sampleSize;	//number of random numbers

	int maxBounces;			//maximum number of bounces per array
};