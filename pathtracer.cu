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

#include "scene.h"
#include "pathtracer.h"
#include "cutil.h"

#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <vector>
#include <time.h>

// gamme correction macro
#define GAMMA_FLOAT(c, gamma) powf(Clamp(c, 0.f, 1.f), 1.f/gamma)

// saves the random numbers for subsampling for a pixel
// each thread produces one to abstract the uniform grid
struct CameraSamples {
	CUDA_DEVICE CameraSamples(float *s) : samples(s) {
		//TODO elegant way to do this
		// upper left
		samples[0] *= -0.25f;
		samples[1] *= -0.25f;
		// upper right
		samples[2] *= 0.25f;
		samples[3] *= -0.25f;
		// lower left
		samples[4] *= -0.25f;
		samples[5] *= 0.25f;
		// lower right
		samples[6] *= 0.25f;
		samples[7] *= 0.25f;
	}
	CUDA_DEVICE CameraSample operator[](uint32_t index) const {
		return CameraSample(samples[index*2], samples[index*2+1]);
	}

	float *samples;
};


__device__ Color Trace(const Ray &ray, const Scene *scene, float* rng, int maxBounces=10);

__global__ void GenerateRayPool(Camera camera, Ray *rayBuffer, float *devRand) {
	const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	const uint32_t index = (y*camera.Width()+x);

	if(x >= camera.Width() || y >= camera.Height()) {
		return;
	}

	CameraSamples samples(&devRand[index*4*2]);
	for(int i=0; i<4; ++i)
		rayBuffer[index*4+i] = camera.GenerateRay(x, y, samples[i]);
}

__global__ void RenderKernel(const Scene *scene, float *buffer, float *rng, Ray *rayPool, uint32_t width, uint32_t height, uint32_t iteration, int maxBounces) {
	const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	const uint32_t index = (y*width+x);
	const uint32_t fbindex = index*3;

	if(x >= width || y >= height) {
		return;
	}

	int rngOffset = index*maxBounces*3;

	Color c;
	// 2x2 Subsampling
	for(int i=0; i<4; ++i) {
		Ray ray = rayPool[index*4+i];
		c += Trace(ray, scene, &rng[rngOffset], maxBounces);
	}
	c /= 4.f;

	float gamma = 2.2f;
	float invIteration = 1.f/(float)iteration;
	
	buffer[fbindex]   = ((buffer[fbindex]   * (iteration-1) + GAMMA_FLOAT(c.r, gamma)) * invIteration);
	buffer[fbindex+1] = ((buffer[fbindex+1] * (iteration-1) + GAMMA_FLOAT(c.g, gamma)) * invIteration);
	buffer[fbindex+2] = ((buffer[fbindex+2] * (iteration-1) + GAMMA_FLOAT(c.b, gamma)) * invIteration);
}

__device__ Color Trace(const Ray &r, const Scene *scene, float* rng, int maxBounces) {
	Color pathThroughput(1.f,1.f,1.f), L;

	Ray ray = r;
	bool specularBounce = false;
	Intersection isect;

	if(!scene->Intersect(ray, &isect))
		return Color();

	Intersection isectp = isect;

	for (int bounces = 0;; ++bounces) {
		const Primitive *prim = isectp.prim;
		const Material &mat = scene->materials[prim->GetMaterialId()];
		const Point &p = isectp.p;
		const Vec &n = isectp.n;
		const Vec wo = ray.d;
		const int rngIndex = bounces*3;

		//if (bounces == 0 || specularBounce)
		//	L += pathThroughput * mat.color * (mat.emitting ? 1.f : 0.f);

		float lightPdf = 1.f;
		// chose light depend on power heuristic
		//uint32_t lightNum = lightDistribution->SampleDiscrete(rndFloat(),
		//	&lightPdf);
		//const Light *light = scene->lights[lightNum];
		const Light &light = scene->lights[0];

		BSDFSample bsdfSample(rng[rngIndex], rng[rngIndex+1]);

		// get direct lighting
		L += pathThroughput * scene->EstimateDirect(light, wo, isectp) / lightPdf;

		Vec wi;
		float pdf;
		// get reflected sampling
		const Color f = mat.SampleF(wo, &wi, &pdf, isectp, bsdfSample);

		// leave if current sample gives no contribution
		if (f.IsBlack() || pdf == 0.f) break;

		specularBounce = (mat.type == SPECULAR || mat.type == TRANSMISSIVE);
		pathThroughput *= f * fabs(wi.Dot(n)) / pdf;
		ray = Ray(p, wi);

		// russian roulette
		if (bounces > 3) {
			float continueProbability = min(.5f, pathThroughput.Y());
			float rnd = rng[rngIndex+2];
			if (rnd > continueProbability)
				break;
			pathThroughput /= continueProbability;
		}
		if (bounces == maxBounces)
			break;

		// if ray leaves scene we can stop here
		if (!scene->Intersect(ray, &isect))
			break;

		isectp = isect;
	}
	return L;
}

bool Pathtracer::Init(const std::vector<Material> &mv, const std::vector<Primitive> &pv, const std::vector<Light> &lv) {

	//copy primitive list to device
	Primitive* devPrimitives = NULL;
	size_t numPrims = pv.size();
	CudaSafeCall(cudaMalloc((void**)&(devPrimitives), numPrims*sizeof(Primitive)));
	CudaSafeCall(cudaMemcpy(devPrimitives, &pv[0], numPrims*sizeof(Primitive), cudaMemcpyHostToDevice));
	PrimitiveList primList = PrimitiveList(devPrimitives, numPrims);

	//copy light list to device
	Light* devLights = NULL;
	size_t numLights = lv.size();
	CudaSafeCall(cudaMalloc((void**)&(devLights), numLights*sizeof(Light)));
	CudaSafeCall(cudaMemcpy(devLights, &lv[0], numLights*sizeof(Light), cudaMemcpyHostToDevice));
	LightList lightList = LightList(devLights, numLights);

	//copy material list to device
	Material* devMats = NULL;
	size_t numMats = pv.size();
	CudaSafeCall(cudaMalloc((void**)&(devMats), numMats*sizeof(Material)));
	CudaSafeCall(cudaMemcpy(devMats, &mv[0], numMats*sizeof(Material), cudaMemcpyHostToDevice));
	MaterialList matList = MaterialList(devMats, numMats);

	//copy scene to device
	Scene hostScene (matList, primList, lightList);
	CudaSafeCall(cudaMalloc((void**)&(scene), sizeof(Scene)));
	CudaSafeCall(cudaMemcpy(scene, &hostScene, sizeof(Scene), cudaMemcpyHostToDevice));

	//allocate memory for random numbers
	sampleSize = width*height*SAMPLES_PER_PIXEL*maxBounces*3;
	CudaSafeCall(cudaMalloc((void**)&(devRand), sampleSize*sizeof(float)));

	//allocate memory for ray pool
	rayPool = NULL;
	size_t numRays = width*height*4;
	CudaSafeCall(cudaMalloc((void**)&(rayPool), numRays*sizeof(Ray)));

	return true;
}

void Pathtracer::Run(float* devBuffer) {
	// create random numbers
	uint32_t sampleSize = width*height*SAMPLES_PER_PIXEL*maxBounces*3;
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
	curandGenerateUniform(gen, devRand, sampleSize);
	curandDestroyGenerator(gen);
	CudaCheckError();

	int dimx = 16;
	int dimy = 16;
	dim3 dimGrid(width/dimx, height/dimy);
	dim3 dimBlock(dimx, dimy);

	GenerateRayPool<<<dimGrid, dimBlock>>>(*camera, rayPool, devRand);
	cudaThreadSynchronize();
	CudaCheckError();

	//launch kernel
	uint32_t randOffset = width*height*SAMPLES_PER_PIXEL;
	RenderKernel<<<dimGrid, dimBlock>>>(scene, devBuffer, &devRand[randOffset], rayPool, width, height, ++iteration, maxBounces);

	cudaThreadSynchronize();
	CudaCheckError();
}

void Pathtracer::Reset() {
	iteration = 0;
}

void Pathtracer::Release() {
	//release buffers
	cudaFree(devRand);
}
