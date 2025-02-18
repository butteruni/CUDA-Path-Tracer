#pragma once
#include <glm/glm.hpp>
#include "sampler.h"
// ref: https://pbr-book.org/4ed/Reflection_Models
enum BxDFFlags {
    Unset = 0,
    Reflection = 1 << 0,
    Transmission = 1 << 1,
    Diffuse = 1 << 2,
    Glossy = 1 << 3,
    Specular = 1 << 4,
    DiffuseReflection = Diffuse | Reflection,
    DiffuseTransmission = Diffuse | Transmission,
    GlossyReflection = Glossy | Reflection,
    GlossyTransmission = Glossy | Transmission,
    SpecularReflection = Specular | Reflection,
    SpecularTransmission = Specular | Transmission,
    All = Diffuse | Glossy | Specular | Reflection | Transmission
};
struct BSDFSample {
    glm::vec3 wi;
    float pdf;
    BxDFFlags flags;
    glm::vec3 bsdf;
	BSDFSample() = default;
    BSDFSample(glm::vec3 wi, float pdf, BxDFFlags flags, glm::vec3 bsdf) :
        wi(wi), pdf(pdf), flags(flags), bsdf(bsdf) {}
};
CPUGPU inline float fresnelSchlick(float cosTheta, float f0) {
	f0 = (1.f - f0) / (1.f + f0);
	return f0 + (1.f - f0) * glm::pow(1.f - cosTheta, 5.f);
}
CPUGPU inline glm::vec3 fresnelSchlick(float cosTheta, glm::vec3 f0) {
	return f0 + (1.f - f0) * glm::pow(1.f - cosTheta, 5.f);
}
CPUGPU inline float fresnelDielectric(float cosThetaI, float eta) {
	cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);
	bool entering = cosThetaI > 0.f;
	if (!entering) {
		eta = 1.f / eta;
		cosThetaI = -cosThetaI;
	}
	float sinThetaI = glm::sqrt(glm::max(0.f, 1.f - cosThetaI * cosThetaI));
	float sinThetaT = sinThetaI / eta;
	if (sinThetaT >= 1.f) {
		return 1.f;
	}
	float cosThetaT = glm::sqrt(glm::max(0.f, 1.f - sinThetaT * sinThetaT));
	float Rparl = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
	float Rperp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
	return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
}
CPUGPU inline bool refract(const glm::vec3& wi, const glm::vec3& n, float eta, glm::vec3& wt) {
	float cosThetaI = glm::dot(n, wi);
	if (cosThetaI < 0.f) {
		eta = 1.f / eta;
		cosThetaI = -cosThetaI;
	}
	float sin2ThetaI = glm::max(0.f, 1.f - cosThetaI * cosThetaI);
	float sin2ThetaT = sin2ThetaI / (eta * eta);
	if (sin2ThetaT >= 1.f) {
		return false;
	}
	float cosThetaT = glm::sqrt(1.f - sin2ThetaT);
	if (cosThetaI < 0) {
		cosThetaT *= -1;
	}
	wt = glm::normalize(-wi / eta + (cosThetaI / eta - cosThetaT) * n);
	return true;
}
// Geometry Function
CPUGPU inline float Schlick_GGX(float cosTheta, float k) {
	return cosTheta / (cosTheta * (1.f - k) + k);
}
CPUGPU inline float Smith_GGX(float cosThetaI, float cosThetaO, float alphaG) {
	return Schlick_GGX(glm::abs(cosThetaI), alphaG) * Schlick_GGX(glm::abs(cosThetaO), alphaG);
}
// Normal Distribution Function
CPUGPU inline float GGX_D(float cosThetaH, float alphaG) {
	float alphaG2 = alphaG * alphaG;
	float denom = (cosThetaH * cosThetaH * (alphaG2 - 1.f) + 1.f);
	return alphaG2 / (PI * denom * denom);
}
CPUGPU inline float GGX_Pdf(glm::vec3 n, glm::vec3 m, glm::vec3 wo, float alpha) {
	return GGX_D(glm::dot(n, m), alpha) * Schlick_GGX(glm::dot(n, wo), alpha) 
		* fabs(glm::dot(wo, m) / glm::dot(n, wo));
}
enum MaterialType {
	Lambertian,
	Conductor,
	Dielectric,
	Light
};
struct Material
{
	MaterialType type;
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
	int colorTextureId;
    float roughness;
    float indexOfRefraction;
    float emittance;
	float metallic;
	CPUGPU glm::vec3 DiffuseBSDF() {
		return color * INV_PI;
	}
	CPUGPU float DiffusePdf(const glm::vec3& n, const glm::vec3& wi) {
		return glm::max(0.f, glm::dot(n, wi)) * INV_PI;
	}
	CPUGPU void DiffuseSampleBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 r, BSDFSample& sample) {
		sample.wi = squareToHemiSphereCos(n, glm::vec2(r.x, r.y));
		sample.bsdf = DiffuseBSDF();
		sample.pdf = DiffusePdf(n, sample.wi);
		sample.flags = BxDFFlags::DiffuseReflection;
	}
	CPUGPU glm::vec3 ConductorBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
		float alpha = roughness * roughness;
		glm::vec3 h = glm::normalize(wo + wi);
		float cosThetaO = glm::dot(n, wo);
		float cosThetaI = glm::dot(n, wi);
		if (cosThetaI * cosThetaO < 1e-7f) {
			return glm::vec3(0.f);
		}
		glm::vec3 F = fresnelSchlick(glm::dot(h, wo), glm::mix(glm::vec3(.08f), color, metallic));
		float G = Smith_GGX(cosThetaI, cosThetaO, alpha);
		float D = GGX_D(glm::dot(n, h), alpha);

		return glm::mix(DiffuseBSDF() * (1.f - metallic),
						glm::vec3(G * D / (4.f * cosThetaI * cosThetaO)),
						F);
	}
	CPUGPU float ConductorPdf(const glm::vec3 &n, const glm::vec3 &wo, const glm::vec3& wi) {
		glm::vec3 h = glm::normalize(wi + wo);
		return glm::mix(DiffusePdf(n, wi),
						GGX_Pdf(n, h, wo, roughness * roughness) / glm::max(EPSILON, (4.f * glm::dot(h, wo))),
						1.f / (2.f - metallic));


	}
	CPUGPU void ConductorSampleBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 r, BSDFSample& sample) {
		float alpha = roughness * roughness;
		if (r.z > 1.f / (2.f - metallic)) {
			sample.wi = squareToHemiSphereCos(n, glm::vec2(r.x, r.y));
		}
		else {
			glm::vec3 h = GGX_sampleNormal(n, wo, glm::vec2(r.x, r.y), alpha);
			sample.wi = -glm::reflect(wo, h);
		}
		if (glm::dot(n, sample.wi) < 0.f) {
			sample.flags = BxDFFlags::Unset;
		}
		else {
			sample.bsdf = ConductorBSDF(n, wo, sample.wi);
			sample.pdf = ConductorPdf(n, wo, sample.wi);
			sample.flags = BxDFFlags::GlossyReflection;
		}
	}
	CPUGPU glm::vec3 DielectricBSDF() {
		return glm::vec3(0.f);
	}
	CPUGPU float DielectricPdf(const glm::vec3& n, const glm::vec3& wi) {
		return 0.f;
	}
	CPUGPU void DielectricSampleBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 r, BSDFSample& sample) {
		float reflection_prob = fresnelDielectric(glm::dot(n, wo), indexOfRefraction);

		if (r.z < reflection_prob) {
			sample.wi = glm::reflect(-wo, n);
			sample.flags = BxDFFlags::SpecularReflection;
			sample.bsdf = color;
			sample.pdf = 1.f;
		}
		else {
			if (!refract(wo, n, indexOfRefraction, sample.wi)) {
				sample.flags = BxDFFlags::Unset;
				return;
			}
			if (glm::dot(n, wo) < 0) {
				indexOfRefraction = 1.f / indexOfRefraction;
			}
			sample.bsdf = color / (indexOfRefraction * indexOfRefraction);
			sample.flags = BxDFFlags::SpecularTransmission;
			sample.pdf = 1.f;
		}
	}
	CPUGPU glm::vec3 BSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
		switch (type)
		{
		case Lambertian:
			return DiffuseBSDF();
			break;
		case Conductor:
			return ConductorBSDF(n, wo, wi);
			break;
		case Dielectric:
			return DielectricBSDF();
			break;
		}
		return glm::vec3(0.f);
	}
	CPUGPU float pdf(const glm::vec3& n, const glm::vec3& wo, const glm::vec3 wi) {
		switch (type) {
		case MaterialType::Lambertian:
			return DiffusePdf(n, wi);
		case MaterialType::Conductor:
			return ConductorPdf(n, wo, wi);
		case MaterialType::Dielectric:
			return DielectricPdf(n, wi);
		default:
			return 0.f;
		}
		
	}
	CPUGPU void SampleBSDF(const glm::vec3& n, const glm::vec3& wo, const glm::vec3& r, BSDFSample& sample) {
		switch (type) {
		case MaterialType::Lambertian:
			DiffuseSampleBSDF(n, wo, r, sample);
			break;
		case MaterialType::Conductor:
			ConductorSampleBSDF(n, wo, r, sample);
			break;
		case MaterialType::Dielectric:
			DielectricSampleBSDF(n, wo, r, sample);
			break;
		}
	}
	CPUGPU void operator =(const Material &rhs) {
		type = rhs.type;
		color= rhs.color;
		specular = rhs.specular;
		roughness = rhs.roughness;
		indexOfRefraction = rhs.indexOfRefraction;
		metallic = rhs.metallic;
	}
};
