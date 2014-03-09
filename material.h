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

/**
 * saves u and v components for a random BDSF sample
 */
struct BSDFSample {
	CUDA_DEVICE BSDFSample() : u(0.f), v(0.f) {}
	CUDA_DEVICE BSDFSample(float u_, float v_) : u(u_), v(v_) {}

	const float u, v;
};

enum MaterialType { DIFFUSE, SPECULAR, TRANSMISSIVE };

/**
 * simple material abstraction
 * TODO more fancy stuff
 */
class Material {
public:
	/**
	 * Generates a sample reflection from direction wo in direction wo
	 * If the material is perfect specular or transmitting then the scatter ray are deterministic
	 * Otherwise do simple cosine hemisphere sampling for diffuse materials
	 */
	CUDA_DEVICE Color SampleF(const Vec &wo, Vec *wi, float *pdf, const Intersection &isect, const BSDFSample &sample) const {
		const Material *mat = isect.mat;
		const Vec &n = isect.n;
		Color f;

		// diffuse sampling
		if(mat->type == DIFFUSE) {
			*wi = CosineSampleHemisphere(sample.u, sample.v, n);
			*pdf = wo.Dot(*wi) < 0.f ? fabs(wi->Dot(n)) * INV_PI : 0.f;
			f = mat->F(wo, *wi, n);

		// perfect reflection
		} else if(mat->type == SPECULAR) {
			*wi = Reflect(wo, n);
			*pdf = 1.f;
			f = mat->coef * mat->color;

		// refraction
		} else if(mat->type == TRANSMISSIVE) {
			float n1, n2;
			Vec nnor;

			// ray from the outside
			if (wo.Dot(n) < 0.0f) {
				n1 = 1.0f;
				n2 = mat->coef;
				nnor = n;
			}
			// ray from the inside
			else {
				n1 = mat->coef;
				n2 = 1.0f;
				nnor = -n;
			}

			float refl = reflectance(wo, nnor, n1, n2);
			if (sample.u < refl) {
				*wi = Reflect(wo, nnor);
				f = refl*mat->color;
			} else {
				*wi = Refract(wo, nnor, n1/n2);
				f = (1.f-refl)*mat->color; 
			}
			*pdf = 1.f;
		}
		return f;
	}

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


/**
 * Provides a list of materials on the device memory
 */
struct MaterialList {
	MaterialList() : materials(NULL), size(0) {}
	MaterialList(Material* m, uint32_t s) : materials(m), size(s) {}
	CUDA_DEVICE Material& operator[](uint32_t index) const { return materials[index]; }

	Material* materials;
	uint32_t size;
};

#endif /* __MATERIAL_H__ */