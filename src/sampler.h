#pragma once
#include <glm/glm.hpp>
#include <thrust/random.h>
#include <stack>
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

CPUGPU static glm::mat3 localRefMatrix(glm::vec3 n) {
	glm::vec3 t = (glm::abs(n.y) > 0.9999f) ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);
	glm::vec3 b = glm::normalize(glm::cross(n, t));
	t = glm::cross(b, n);
	return glm::mat3(t, b, n);
}

CPUGPU static glm::vec3 localToWorld(glm::vec3 n, glm::vec3 v) {
	return glm::normalize(localRefMatrix(n) * v);
}

CPUGPU static glm::vec3 squareToHemiSphereUniform(const glm::vec3 n, const glm::vec2& sample)
{
	float z = sample.x;
	float r = sqrtf(fmax(0.0f, 1.0f - z * z));
	float phi = TWO_PI * sample.y;
	return localToWorld(n, glm::vec3(r * cos(phi), r * sin(phi), z));
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

	glm::vec3 t1 = (wh.z < 0.999999f) ? 
		glm::normalize(glm::cross(glm::vec3(0.f, 0.f, 1.f), wh)) : glm::vec3(1.f, 0.f, 0.f);
	glm::vec3 t2 = glm::cross(wh, t1);
	glm::vec2 p = squareToDiskConcentric(r);
	float h = sqrtf(fmax(0.f, 1.f - p.x * p.x));
	p.y = (1.f - h) * glm::sqrt(1.f - p.x * p.x) + h * p.y;
	float pz = glm::sqrt(glm::max(0.f, 1.f - glm::length(p)));
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
GPU inline glm::vec4 sample4D(thrust::default_random_engine& rng) {
	return glm::vec4(sample3D(rng), sample1D(rng));
}
CPUGPU inline glm::vec3 sampleBary(const glm::vec2& sample) {
	glm::vec2 r = glm::sqrt(sample);
	float u = 1.f - r.y;
	float v = sample.x * r.y;
	return glm::vec3(1.f - u - v, u, v);
}
// use for multiple importance sampling
template <typename T>
struct DF
{
	T prob;
	int failId;
};
template <typename T>
class DiscreteSampler1D {
public:
	using DistributionT = DF<T>;
	std::vector<DistributionT> binomDistribution;
	float sum = 0;
	DiscreteSampler1D() = default;
	DiscreteSampler1D(std::vector<T> distribution) {
		sum = 0;
		binomDistribution.resize(distribution.size());
		for (int i = 0; i < distribution.size(); i++) {
			sum += distribution[i];
		}
		T sum_inv = static_cast<T>(distribution.size()) / sum;
		for (auto& x : distribution) {
			x *= sum_inv;
		}
		std::stack<DistributionT> greaterThanOne;
		std::stack<DistributionT> lessThanOne;
		for (int i = 0; i < distribution.size(); i++) {
			if (distribution[i] > 1.f) {
				greaterThanOne.push({ distribution[i], i });
			}
			else {
				lessThanOne.push({ distribution[i], i });
			}
		}
		while (!greaterThanOne.empty() && !lessThanOne.empty()) {
			auto& g = greaterThanOne.top();
			auto& l = lessThanOne.top();
			greaterThanOne.pop();
			lessThanOne.pop();
			binomDistribution[l.failId] = { l.prob, g.failId };
			g.prob -= 1.f - l.prob;
			if (g.prob > 1.f) {
				greaterThanOne.push(g);
			}
			else {
				lessThanOne.push(g);
			}
		}
		while (!greaterThanOne.empty()) {
			auto& g = greaterThanOne.top();
			greaterThanOne.pop();
			binomDistribution[g.failId] = g;
		}
		while (!lessThanOne.empty()) {
			auto& l = lessThanOne.top();
			lessThanOne.pop();
			binomDistribution[l.failId] = l;
		}
		std::cout << "sum: " << sum << '\n';
	}
	int sample(float u, float v) {
		int idx = int(float(binomDistribution.size()) * u);
		return v < binomDistribution[idx].prob ? idx : binomDistribution[idx].failId;
	}
	void clear() {
		binomDistribution.clear();
		sum = 0;
	}
};

template <typename T>
class GPUDiscreteSampler1D {
public:
	using DistributionT = DF<T>;
	DistributionT* binomDistribution;
	int size = 0;
	void loadFromHost(const DiscreteSampler1D<T>& sampler) {
		size = sampler.binomDistribution.size();
		cudaMalloc(&binomDistribution, size * sizeof(DistributionT));
		cudaMemcpy(binomDistribution, sampler.binomDistribution.data(), size * sizeof(DistributionT), cudaMemcpyHostToDevice);
	}
	void clear() {
		cudaFree(binomDistribution);
		size = 0;
	}
	GPU int sample(float u, float v) {
		int idx = min(int(float(size) * u), size - 1);
		DistributionT d = binomDistribution[idx];
		return v < d.prob ? idx : d.failId;
	}
};
