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

#ifndef __SCENE_H__
#define __SCENE_H__

#include <cuda_runtime.h>
#include "camera.h"
#include "material.h"
#include "primitive.h"
#include "light.h"


/**
 * Abstract access to scene objects like primitives, lights and materials
 * Also includes methods for scene intersection and direct lighting estimation
 * TODO kd-tree acceleration structure
 */
class Scene {
public:
	Scene(const MaterialList &ml, const PrimitiveList &pl, const LightList &ll)
		: materials(ml), primitives(pl), lights(ll) {}

	const PrimitiveList primitives;
	const LightList lights;
	const MaterialList materials;
};

struct Intersection {
	CUDA_DEVICE Intersection() : prim(NULL), mat(NULL), p(Point()), n(Vec()), t(0.f) {}

	/**
	 * Get emitted Light of intersection
	 */
	CUDA_DEVICE Color Le(const Vec &w, const Scene *scene) {
		if(prim->IsLight()) {
			const Light *light = scene->lights[prim->lightId];
			return light->L(p, w, n);
		} else
			return Color();
	}

	Primitive *prim;
	Material *mat;
	Point p;
	Vec n;
	float t;
};

/**
* Ray - scene intersection test
*
* @return true if ray hit an object
*/
inline CUDA_DEVICE bool Intersect(const Scene *scene, const Ray &ray, Intersection *isect, 
		float tmin=EPSILON, float tmax=INF) {
	Intersection is;
	bool found = false;
	for(uint32_t i=0; i<scene->primitives.size; ++i) {
		float thit;
		Primitive *prim = scene->primitives[i];
		if((thit = prim->Intersect(ray, tmin, tmax)) > 0.f) {
			uint32_t matId = prim->materialId;
			tmax = thit;
			found = true;

			is.prim = prim;
			is.mat = scene->materials[matId];
			is.p = ray(thit);
			is.n = prim->GetNormal(is.p);
			is.t = thit;
		}
	}
	*isect = is;
	return found;
}

/**
* Ray - scene hit test
*
* @return true if ray hit an object
*/
inline CUDA_DEVICE bool IntersectP(const Scene *scene, const Ray &ray, float tmin=EPSILON, 
		float tmax=INF) {
	for(uint32_t i=0; i<scene->primitives.size; ++i) {
		if(scene->primitives[i]->IntersectP(ray, tmin, tmax))
			return true;
	}
	return false;
}

inline CUDA_DEVICE float Pdf(const Primitive *prim, const Point &p, const Vec &wi) {
	float rad = prim->radius;
	// Return uniform weight if point inside sphere
	//if (p.DistanceSquared(prim->pos) - rad*rad < 1e-4f)
	//	return prim->Pdf(p, wi);

	// Compute general sphere weight
	float sinThetaMax2 = rad*rad / p.DistanceSquared(prim->pos);
	float cosThetaMax = sqrtf(max(0.f, 1.f - sinThetaMax2));
	return UniformConePdf(cosThetaMax);
}

inline CUDA_DEVICE float Pdf(const Material *mat, const Vec &wo, const Vec &wi,
		const Vec &n) {
	if(mat->type == DIFFUSE) {
		return -wo.Dot(wi) > 0.f ? wi.Dot(n) * INV_PI : 0.f;

	} else {
		return 0.f;
	}
}

inline CUDA_DEVICE Color SampleLight(const Light *light, const Point &p, Vec *wi,
		float *pdf, VisibilityTester *vt, const UVSample &sample, 
		const Scene *scene) {
	if(light->type == POINT_LIGHT) {
		const Point &pos = light->pos;
		*wi = (pos-p).Normalize();
		*pdf = 1.f;
		vt->SetSegment(p, EPSILON, pos, 0);
		return light->intensity/pos.DistanceSquared(p);

	} else if(light->type == AREA_LIGHT) {
		Vec n;
		const Primitive *prim = scene->primitives[light->primId];
		Point sp = prim->Sample(p, sample, &n);
		//printf("%f, %f, %f\n", sp.x, sp.y, sp.z);
		*wi = (sp-p).Normalize();
		*pdf = Pdf(prim, p, *wi);
		vt->SetSegment(p, EPSILON, sp, EPSILON);
		return light->L(p, -*wi, n);
	}
	return Color();
}

/**
* Generates a sample reflection from direction wo in direction wo
* If the material is perfect specular or transmitting then the scatter ray are deterministic
* Otherwise do simple cosine hemisphere sampling for diffuse materials
*/
inline CUDA_DEVICE Color SampleMaterial(const Material *mat, const Vec &wo, Vec *wi, 
		float *pdf,	const Vec &n, const UVSample &sample) {
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
		if (wo.Dot(n) < 0.f) {
			n1 = 1.f;
			n2 = mat->coef;
			nnor = n;
		}
		// ray from the inside
		else {
			n1 = mat->coef;
			n2 = 1.f;
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
* Estimate direct lighting
*
* @return Color contribution
*/
inline CUDA_DEVICE Color EstimateDirect(const Light *light, const Vec &wo, 
		const Intersection &isect, const UVSample &sample, const Scene *scene) {
	Color Ld;
	Vec wi;
	float lightPdf, bsdfPdf;
	VisibilityTester vt;
	const Point &p = isect.p;

	// get light sample
	const Color Li = SampleLight(light, isect.p, &wi, &lightPdf, &vt, sample, scene);

	const Material *mat = isect.mat;
	const Vec &n = isect.n;
	if (lightPdf > 0.f && !Li.IsBlack()) {
		Color f = mat->F(wo, wi, n);
		Ray &lightRay = vt.r;
		if (!f.IsBlack() && !IntersectP(scene, lightRay, vt.mint, vt.maxt)) {
			if (light->type == POINT_LIGHT)
				Ld += f * Li * (fabsf(wi.Dot(n)) / lightPdf);
			else {
				bsdfPdf = Pdf(mat, wo, wi, n);
				float weight = PowerHeuristic(1.f, lightPdf, 1.f, bsdfPdf);
				Ld += f * Li * (fabsf(wi.Dot(n)) * weight / lightPdf);
			}
		}
	}

	if(light->type == AREA_LIGHT) {
		Color f = SampleMaterial(mat, wo, &wi, &bsdfPdf, n, sample);
		if(!f.IsBlack() && bsdfPdf > 0.f) {
			float weight = 1.f;
			if(mat->type == DIFFUSE) {
				lightPdf = Pdf(scene->primitives[light->primId], p, wi);
				if(lightPdf == 0.f)
					return Ld;
				weight = PowerHeuristic(1, bsdfPdf, 1, lightPdf);
			}

			Intersection lightIsect;
			Color Li;
			Ray ray(p, wi);
			if(Intersect(scene, ray, &lightIsect) > 0.f) {
				if(lightIsect.prim == scene->primitives[light->primId]) {
					Li = lightIsect.Le(-wi, scene);
					Ld += f * Li * fabsf(wi.Dot(n)) * weight / bsdfPdf;
				}
			}
		}
	}
	return Ld;
}

#endif /* __SCENE_H__ */