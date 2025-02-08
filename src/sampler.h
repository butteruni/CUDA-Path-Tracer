#pragma once
#include <glm/glm.hpp>
#include <thrust/random.h>
#include "macro.h"
__host__ __device__ inline glm::vec3 calculateRandomDirectionInHemisphere(
	glm::vec3 normal,
	thrust::default_random_engine& rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(u01(rng)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD)
	{
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
	{
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else
	{
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}

CPUGPU static glm::vec2 squareToDiskConcentric(const glm::vec2& sample)
{
	float r = glm::sqrt(sample.x);
	float theta = TWO_PI * sample.y;
	return glm::vec2(r * glm::cos(theta), r * glm::sin(theta));
}


CPUGPU static glm::vec3 squareToHemiSphereUniform(const glm::vec2& sample)
{
	float z = sample.x;
	float r = sqrtf(fmax(0.0f, 1.0f - z * z));
	float phi = TWO_PI * sample.y;
	return glm::vec3(r * cos(phi), r * sin(phi), z);
}
CPUGPU static glm::mat3 localRefMatrix(glm::vec3 n) {
	glm::vec3 t = (glm::abs(n.y) > 0.9999f) ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);
	glm::vec3 b = glm::normalize(glm::cross(n, t));
	t = glm::cross(b, n);
	return glm::mat3(t, b, n);
}

CPUGPU static glm::vec3 localToWorld(glm::vec3 n, glm::vec3 v) {
	return glm::normalize(localRefMatrix(n) * v);
}
CPUGPU static glm::vec3 squareToHemiSphereCos(const glm::vec3 n, const glm::vec2& sample) {
	glm::vec2 d = squareToDiskConcentric(sample);
	float z = sqrtf(fmax(0.f, 1.f - glm::dot(d, d)));
	
	return localToWorld(n, glm::vec3(d, z));

}
CPUGPU static glm::vec3 GGX_sampleNormal(const glm::vec3& n, const glm::vec3 &wo, const glm::vec2& r, float alpha) {
	glm::mat3 refMat = localRefMatrix(n);
	glm::mat3 refMatInv = glm::inverse(refMat);
	glm::vec3 wh = wo * glm::vec3(alpha, alpha, 1.f);
	wh = glm::normalize(refMatInv * wh);

	glm::vec3 t1 = (wh.z < 0.9999f) ? 
		glm::normalize(glm::cross(glm::vec3(0.f, 0.f, 1.f), wh)) : glm::vec3(1.f, 0.f, 0.f);
	glm::vec3 t2 = glm::cross(wh, t1);
	glm::vec2 p = squareToDiskConcentric(r);
	float h = sqrtf(fmax(0.f, 1.f - p.x * p.x));
	p.y = (1.f - h) * glm::sqrt(1.f - p.x * p.x) + h * p.y;
	float pz = glm::sqrt(glm::max(0.f, 1.f - p.length()));
	glm::vec3 nh = p.x * t1 + p.y * t2 + pz * wh;
	nh = glm::vec3(nh.x * alpha, nh.y * alpha, glm::max(0.f, nh.z));
	return glm::normalize(refMat * nh);
}
CPUGPU static float squareToSphereUniformPdf(const glm::vec3& sample)
{
	return INV_FOUR_PI;
}
CPUGPU static float squareToHemiSphereCosPdf(const glm::vec3& sample)
{
	return INV_TWO_PI * sample.z;
}
GPU inline float sample1D(thrust::default_random_engine& rng) {
	return thrust::uniform_real_distribution<float>(0.f, 1.f)(rng);
}
GPU inline glm::vec2 sample2D(thrust::default_random_engine& rng) {
	return glm::vec2(sample1D(rng), sample1D(rng));
}
GPU inline glm::vec3 sample3D(thrust::default_random_engine& rng) {
	return glm::vec3(sample1D(rng), sample1D(rng), sample1D(rng));
}

template <typename T>
struct DF
{
	T prob;
	int failId;
};
template <typename T>
class Sampler1D {

};

template<typename T>
class GPUSampler1D {

};