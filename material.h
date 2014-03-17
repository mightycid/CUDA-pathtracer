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

#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "montecarlo.h"


enum MaterialType { DIFFUSE, SPECULAR, TRANSMISSIVE };

/**
 * simple material abstraction
 * TODO more fancy stuff
 */
class Material {
public:
	/**
	 * Sample color contribution vom incoming direction wo to outgoing 
	 * direction wi with respect to normal n
	 */
	CUDA_DEVICE Color F(const Vec &wo, const Vec &wi, const Vec &n) const {
		Color f;
		if(type == DIFFUSE) {
			f += color * INV_PI;
		}
		return f;
	}

	Color color;
	float coef;
	MaterialType type;
};


/**
 * Material creation methods
 */

inline Material CreateDiffuseMaterial(const Color &color, float coef) {
	Material mat;
	mat.color = color;
	mat.coef = coef;
	mat.type = DIFFUSE;
	return mat;
}

inline Material CreateSpecularMaterial(const Color &color, float coef) {
	Material mat;
	mat.color = color;
	mat.coef = coef;
	mat.type = SPECULAR;
	return mat;
}

inline Material CreateTransmissiveMaterial(const Color &color, float coef) {
	Material mat;
	mat.color = color;
	mat.coef = coef;
	mat.type = TRANSMISSIVE;
	return mat;
}

inline Material CreateEmittingMaterial(const Color &color) {
	Material mat;
	mat.color = color;
	mat.coef = 0.f;
	mat.type = TRANSMISSIVE;
	return mat;
}


/**
 * Provides a list of materials on the device memory
 */
struct MaterialList {
	MaterialList(Material* m, uint32_t s) : materials(m), size(s) {}

	CUDA_DEVICE Material* operator[](uint32_t index) const {
		return &materials[index];
	}

	Material* materials;
	uint32_t size;
};

#endif /* __MATERIAL_H__ */