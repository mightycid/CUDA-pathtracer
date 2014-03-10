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

	/**
	 * Ray - scene intersection test
	 *
	 * @return true if ray hit an object
	 */
	CUDA_DEVICE bool Intersect(const Ray &ray, Intersection *isect, float tmin=EPSILON, float tmax=INF) const {
		Intersection is;
		bool found = false;
		for(uint32_t i=0; i<primitives.size; ++i) {
			float thit;
			Primitive &prim = primitives[i];
			if((thit = prim.Intersect(ray, tmin, tmax)) > 0.f) {
				uint32_t matId = prim.GetMaterialId();
				tmax = thit;
				found = true;

				is.prim = &prim;
				is.mat = &materials[matId];
				is.p = ray(thit);
				is.n = prim.GetNormal(is.p);
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
	CUDA_DEVICE bool IntersectP(const Ray &ray, float tmin=EPSILON, float tmax=INF) const {
		Intersection is;
		for(uint32_t i=0; i<primitives.size; ++i) {
			if(primitives[i].IntersectP(ray, tmin, tmax))
				return true;
		}
		return false;
	}

	/**
	 * Estimate direct lighting
	 *
	 * @return Color contribution
	 */
	CUDA_DEVICE Color EstimateDirect(const Light &light, const Vec &wo, const Intersection &isect) const {
		Color Ld;
		Vec wi;
		float lightPdf, bsdfPdf;
		VisibilityTester vt;
		// get light sample
		const Color Li = light.SampleL(isect.p, &wi, &lightPdf, &vt);

		const Material *mat = isect.mat;
		const Vec &n = isect.n;
		if (lightPdf > 0.f && !Li.IsBlack()) {
			Color f = mat->F(wo, wi, n);
			Ray &lightRay = vt.r;
			if (!f.IsBlack() && !IntersectP(lightRay, vt.mint, vt.maxt)) {
				//if (light->IsDeltaLight())
					Ld += f * Li * (fabsf(wi.Dot(n)) / lightPdf);
				/*else {
					bsdfPdf = m->Pdf(wo, wi, n, flags);
					float weight = PowerHeuristic(1.f, lightPdf, 1.f, bsdfPdf);
					Ld += f * Li * (fabsf(wi.Dot(n)) * weight / lightPdf);
				}*/
			}
		}
		return Ld;
	}

	const PrimitiveList primitives;
	const LightList lights;
	const MaterialList materials;
};

#endif /* __SCENE_H__ */