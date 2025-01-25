#pragma once
#include <glm/glm.hpp>
#include "macro.h"

CPUGPU static glm::vec2 squareToDiskConcentric(const glm::vec2& sample)
{
	glm::vec2 sampleOffset = 2.0f * sample - glm::vec2(1.0f);
	if (sampleOffset.x == 0 && sampleOffset.y == 0)
	{
		return glm::vec2(0.0f);
	}

	float theta, r;
	if (abs(sampleOffset.x) > abs(sampleOffset.y))
	{
		r = sampleOffset.x;
		theta = PI_OVER_TWO * (sampleOffset.y / sampleOffset.x);
	}
	else
	{
		r = sampleOffset.y;
		theta = PI_OVER_TWO - PI_OVER_TWO * (sampleOffset.x / sampleOffset.y);
	}

	return r * glm::vec2(cos(theta), sin(theta));
}

CPUGPU static glm::vec3 squareToSphereUniform(const glm::vec2& sample)
{
	float z = 1.0f - 2.0f * sample.x;
	float r = sqrtf(fmax(0.0f, 1.0f - z * z));
	float phi = TWO_PI * sample.y;
	return glm::vec3(r * cos(phi), r * sin(phi), z);
}
CPUGPU static glm::vec3 squareToHemiSphereUniform(const glm::vec2& sample)
{
	float z = sample.x;
	float r = sqrtf(fmax(0.0f, 1.0f - z * z));
	float phi = TWO_PI * sample.y;
	return glm::vec3(r * cos(phi), r * sin(phi), z);
}
CPUGPU static float squareToSphereUniformPdf(const glm::vec3& sample)
{
	return INV_FOUR_PI;
}
CPUGPU static float squareToHemiSphereCosPdf(const glm::vec3& sample)
{
	return INV_TWO_PI * sample.z;
}