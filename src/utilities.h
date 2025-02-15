#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <thrust/random.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include "macro.h"
#define ERRORCHECK 1
#define RESUFFLE_BY_MATERIAL 0
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}
void inline checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__ inline
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
template <typename T>
inline size_t getVectorByteSize(const std::vector<T>& vec)
{
    return vec.size() * sizeof(T);
}

template <typename T>
void inline safeFree(T*& ptr)
{
    if (ptr != nullptr)
    {
        free(ptr);
        ptr = nullptr;
    }
}

template <typename T>
void inline safeCudaFree(T*& ptr)
{
    if (ptr != nullptr)
    {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

CPUGPU inline glm::vec3 ACES(glm::vec3& color) {
	const float a = 2.51f;
	const float b = 0.03f;
	const float c = 2.43f;
	const float d = 0.59f;
	const float e = 0.14f;
	return glm::clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.f, 1.f);
}
CPUGPU inline glm::vec3 calcfilmic(glm::vec3& color) {
	const float A = 0.22f;
	const float B = 0.3f;
	const float C = 0.1f;
	const float D = 0.2f;
	const float E = 0.01f;
	const float F = 0.3f;
	return (color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F);
}
CPUGPU inline glm::vec3 uncharted2filmic(glm::vec3& color) {
	float exposureBias = 1.6f;
	glm::vec3 curr = exposureBias * color;
	glm::vec3 W = glm::vec3(11.2f);
	glm::vec3 whiteScale = glm::vec3(1.0f) / uncharted2filmic(W);
	return uncharted2filmic(curr) * whiteScale;
}

CPUGPU inline glm::vec3 gammaCorrect(glm::vec3& color) {
	return glm::pow(color, glm::vec3(1.f / 2.2f));
}

CPUGPU inline float rgbTolumin(const glm::vec3 &c) {
	return glm::dot(c, glm::vec3(0.2126f, 0.7152f, 0.0722f));
}

CPUGPU inline std::string vec3ToString(const glm::vec3 &p) {
	return std::to_string(p.x) + ", " + std::to_string(p.y) + ", " + std::to_string(p.z);
}
CPUGPU inline float triangleArea(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
	return 0.5f * glm::length(glm::cross(v1 - v0, v2 - v0));
}
CPUGPU inline float luminance(const glm::vec3& color) {
	return glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
}
CPUGPU inline float computeSolidAngle(const glm::vec3& x, const glm::vec3& y, const glm::vec3& normalY) {
    glm::vec3 yTox = x - y;
	glm::vec3 dir = glm::normalize(yTox);
	return glm::dot(yTox, yTox) / glm::abs(glm::dot(dir, normalY));
}
CPUGPU inline float powerHeuristic(float f, float g) {
	float f2 = f * f;
	return f2 / (f2 + g * g);
}
