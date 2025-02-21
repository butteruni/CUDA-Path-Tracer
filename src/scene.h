#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <map>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "material.h"
#include "bvh.h"
#include "image.h"
#include "intersections.h"
using namespace std;

class Scene;
class GPUScene {
public:
	glm::vec3* vertices = nullptr;
	glm::vec3* normals = nullptr;
	glm::vec2* uvs = nullptr;
	Material* materials = nullptr;
	int* materialIDs = nullptr;
	int verticesSize = 0;
	GPUImage* devEnvTexture = nullptr;
	glm::vec3* texturePixels = nullptr;
	GPUImage* textures = nullptr;
	AABB* deviceBounds = nullptr;
	LinearBVHNode* devlinearNodes = nullptr;
	int devNumNodes = 0;
	int* devLightPrimIds = nullptr;
	glm::vec3* devLightUnitRadiance = nullptr;
	GPUDiscreteSampler1D<float> devlightSampler;
	GPUDiscreteSampler1D<float> devEnvSampler;
	int devNumLightPrim = 0;
	float devSumLightPower = 0;
	float devSumLightPowerInv = 0;
	void loadFromScene(const Scene& scene);
	void clear();

	GPU Material getIntersectionMaterial(ShadeableIntersection &isect) {
		Material m = materials[isect.materialId];
		if (m.colorTextureId != -1) {
			m.color = textures[m.colorTextureId].linearSample(isect.uv.x, isect.uv.y);
		}
		if (m.metallicTextureId != -1) {
			m.metallic = textures[m.metallicTextureId].linearSample(isect.uv.x, isect.uv.y).r;
		}
		if (m.roughnessTextureId != -1) {
			m.roughness = textures[m.roughnessTextureId].linearSample(isect.uv.x, isect.uv.y).r;
		}
		if (m.normalTextureId != -1) {
			glm::vec3 mapped = textures[m.normalTextureId].linearSample(isect.uv.x, isect.uv.y);
			glm::vec3 localNormal = glm::normalize(mapped * 2.f - 1.f);
			isect.surfaceNormal = glm::normalize(localToWorld(isect.surfaceNormal, localNormal));
		}
		return m;
	}

	GPU float getPrimitiveArea(int id) {
		glm::vec3 v0 = vertices[id * 3 + 0];
		glm::vec3 v1 = vertices[id * 3 + 1];
		glm::vec3 v2 = vertices[id * 3 + 2];

		return triangleArea(v0, v1, v2);
	}
	GPU float intersectByIndex(const Ray& r, int index, glm::vec3& bray) {
		glm::vec3 v0 = vertices[index * 3];
		glm::vec3 v1 = vertices[index * 3 + 1];
		glm::vec3 v2 = vertices[index * 3 + 2];
		return triangleIntersectionTest(v0, v1, v2, r, bray);
	}
	GPU void updateIntersection(const Ray& r, ShadeableIntersection& isect,const glm::vec3& bary, float min_T) {
		if (isect.primitiveId != -1) {
			int min_index = isect.primitiveId;
			isect.t = min_T;
			isect.materialId = materialIDs[min_index];
			isect.point = r.origin + min_T * r.direction;
			glm::vec3 n0 = normals[min_index * 3 + 0];
			glm::vec3 n1 = normals[min_index * 3 + 1];
			glm::vec3 n2 = normals[min_index * 3 + 2];
			isect.surfaceNormal = glm::normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);
			glm::vec2 uv0 = uvs[min_index * 3 + 0];
			glm::vec2 uv1 = uvs[min_index * 3 + 1];
			glm::vec2 uv2 = uvs[min_index * 3 + 2];
			isect.uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
		}
		else {
			isect.t = -1;
			isect.materialId = -1;
		}
	}

	GPU bool occlusionNaive(glm::vec3 x, glm::vec3 y) {
		glm::vec3 dir = y - x;
		float dist = glm::length(dir) - EPSILON;
		dir = glm::normalize(dir);
		Ray r = { x + dir * 1e-5f, dir };
		for (int i = 0; i * 3 < verticesSize; i++) {
			glm::vec3 bary;
			float t = intersectByIndex(r, i, bary);
			if (t > 0 && t < dist) {
				return true;
			}
		}
		return false;
	}

	GPU bool occlusionAccel(glm::vec3 x, glm::vec3 y) {
		glm::vec3 dir = y - x;
		float dist = glm::length(dir) - EPSILON;
		dir = glm::normalize(dir);
		Ray r = { x + dir * 1e-5f, dir };
		int cur_node = 0;
		float min_T = dist;
		int start = getLinearId(-r.direction);
		while (cur_node != devNumNodes) {
			AABB bound = deviceBounds[devlinearNodes[start + cur_node].aabbIndex];
			float tmp_t = min_T;
			if (bound.intersect(r, tmp_t)) {
				if (devlinearNodes[start + cur_node].primIndex != -1) {
					int primId = devlinearNodes[start + cur_node].primIndex;
					glm::vec3 tmp_bary;
					float t = intersectByIndex(r, primId, tmp_bary);
					if (t > 0 && t < min_T) {
						return true;
					}
				}
				cur_node++;
			}
			else {
				cur_node = devlinearNodes[start + cur_node].secondChild;
			}
		}
		return false;
	}

	GPU bool occlusionTest(glm::vec3 x, glm::vec3 y) {
		if (BVH_ACCELERATION)
			return occlusionAccel(x, y);
		else
			return occlusionNaive(x, y);
	}
	GPU float envLightPdf(const glm::vec3 radiance) {
		return luminance(radiance) * devSumLightPowerInv * devEnvTexture->xSize * devEnvTexture->ySize * 0.5f;
	}
	GPU float sampleEnvLight(glm::vec3 pos, glm::vec2 random, glm::vec3& radiance, glm::vec3& wi) {
		int pixelId = devEnvSampler.sample(random.x, random.y);
		int y = pixelId / devEnvTexture->xSize;
		int x = pixelId % devEnvTexture->xSize;
		radiance = devEnvTexture->pixels[pixelId];
		wi = UVtoDir(glm::vec2((0.5f + x) / devEnvTexture->xSize, (0.5 + y) / devEnvTexture->ySize));
		bool visible = !occlusionTest(pos, pos + wi * 1e8f);
		if (!visible) {
			return -1;
		}
		return envLightPdf(radiance) * INV_PI * INV_PI;
	}
	GPU float sampleDirectLight(glm::vec3 pos, glm::vec4 random, glm::vec3 &radiance, glm::vec3 &wi) {
		
		int lightId = devlightSampler.sample(random.x, random.y);
		if (lightId == devlightSampler.size - 1 && devEnvSampler.size != 0) {
			return sampleEnvLight(pos, glm::vec2(random.z, random.w), radiance, wi);
		}
		int lightPrimId = devLightPrimIds[lightId];
		glm::vec3 v0 = vertices[lightPrimId * 3 + 0];
		glm::vec3 v1 = vertices[lightPrimId * 3 + 1];
		glm::vec3 v2 = vertices[lightPrimId * 3 + 2];
		float r = glm::sqrt(random.z);
		float u = 1.f - r;
		float v = random.w * r;
		glm::vec3 samplePoint = (1 - u - v) * v0 + u * v1 + v * v2;
		bool visible = !occlusionTest(pos, samplePoint);
		if (!visible) {
			return -1;
		}
		glm::vec3 lightNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
		glm::vec3 lightdir = samplePoint - pos;
		
		radiance = devLightUnitRadiance[lightId];
		wi = glm::normalize(lightdir);
		return luminance(radiance) * devSumLightPowerInv * computeSolidAngle(pos, samplePoint, lightNormal);
	}

	GPU void intersectNaive(const Ray& r, ShadeableIntersection& isect) {
		float min_T = FLT_MAX;
		int min_index = -1;
		glm::vec3 bary;
		for (int i = 0; i * 3 < verticesSize; i++) {
			glm::vec3 tmp_bary;
			float t = intersectByIndex(r, i, tmp_bary);
			if (t > 0 && t < min_T) {
				min_T = t;
				min_index = i;
				bary = tmp_bary;
			}
		}
		isect.primitiveId = min_index;
		updateIntersection(r, isect, bary, min_T);
	}
	GPU int getLinearId(const glm::vec3 &dir) {
		int dim = 0;
		glm::vec3 absDir = glm::abs(dir);
		if (absDir.x > absDir.y) {
			if (absDir.x > absDir.z) {
				dim = (dir.x > 0 ? 0 : 1);
			}
			else {
				dim = (dir.z > 0 ? 4 : 5);
			}
		}
		else {
			if (absDir.y > absDir.z) {
				dim = (dir.y > 0 ? 2 : 3);
			}
			else {
				dim = (dir.z > 0 ? 4 : 5);
			}
		}
		return dim * devNumNodes;
	}
	GPU void intersectAccel(const Ray& r, ShadeableIntersection& isect) {
		float min_T = FLT_MAX;
		int min_index = -1;
		glm::vec3 bary;
		int cur_node = 0;
		int start = getLinearId(-r.direction);
		while (cur_node != devNumNodes) {
			AABB bound = deviceBounds[devlinearNodes[start + cur_node].aabbIndex];
			float t = min_T;
			if (bound.intersect(r, t)) {
				int primId = devlinearNodes[start + cur_node].primIndex;
				if (primId != -1) {
					glm::vec3 tmp_bary;
					float t = intersectByIndex(r, primId, tmp_bary);
					if (t > 0 && t < min_T) {
						min_T = t;
						min_index = primId;
						bary = tmp_bary;
					}
				}
				cur_node++;
			}
			else {
				cur_node = devlinearNodes[start + cur_node].secondChild;
			}
		}
		isect.primitiveId = min_index;
		updateIntersection(r, isect, bary, min_T);
	}
	GPU void intersectTest(const Ray& r, ShadeableIntersection& isect) {
		if (BVH_ACCELERATION) 
			intersectAccel(r, isect);
		else
			intersectNaive(r, isect);
	}
};
class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
	void loadMeshFromObj(const std::string& objName, MeshData *dst_data);
	void loadMesh(const std::string& meshName, Geom &dst_data);
	Image* loadTexture(const std::string& textureName);
	int getTextureIndex(const std::string& textureName);
	void buildSampler();
public:
    Scene(string filename);
    ~Scene();

	std::vector<Geom> geoms;
	std::vector<Image*> textures;
	std::vector<Material> materials;
	std::vector<int> materialIDs;
	std::vector<AABB> bounds;
	std::vector<std::vector<LinearBVHNode>> linearNodes;

	std::vector<int> lightPrimIds;
	std::vector<float> lightPower;
	std::vector<glm::vec3> lightUnitRadiance;
	float sumLightPower = 0;
	int numLightPrim = 0;
	DiscreteSampler1D<float> lightSampler;
	
	std::map<std::string, MeshData*> meshMap;
	std::map<std::string, Image*> textureMap;
	std::map<std::string, int> textureIds;
	int envTextureId = -1;
	DiscreteSampler1D<float> envSampler;
	
	MeshData meshData;
	RenderState state;
	GPUScene hstScene;
	GPUScene* devScene = nullptr;
	void toDevice();
	void clearScene();
};