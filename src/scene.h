#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "material.h"
#include "bvh.h"
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
	
	AABB* deviceBounds = nullptr;
	LinearBVHNode* devlinearNodes = nullptr;
	int devNumNodes = 0;
	int* devLightPrimIds = nullptr;
	glm::vec3* devLightUnitRadiance = nullptr;
	DF<float>* devLightDistribution;
	int devNumLightPrim = 0;
	float devSumLightPower = 0;
	float devSumLightPowerInv = 0;
	void loadFromScene(const Scene& scene);
	void clear();

	GPU Material getMaterialByIndex(int index) {
		return materials[index];
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
			glm::vec3 n0 = normals[min_index * 3];
			glm::vec3 n1 = normals[min_index * 3 + 1];
			glm::vec3 n2 = normals[min_index * 3 + 2];
			isect.surfaceNormal = n0 * bary.x + n1 * bary.y + n2 * bary.z;
			glm::vec2 uv0 = uvs[min_index * 3];
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
		float dist = glm::length(dir);
		dir = glm::normalize(dir);
		Ray r = { x, dir };
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
		Ray r = { x, dir };
		int cur_node = 0;
		float min_T = dist;
		while (cur_node != devNumNodes) {
			AABB& bound = deviceBounds[devlinearNodes[cur_node].aabbIndex];
			if (bound.intersect(r, min_T)) {
				if (devlinearNodes[cur_node].primIndex != -1) {
					int primId = devlinearNodes[cur_node].primIndex;
					glm::vec3 tmp_bary;
					float t = intersectByIndex(r, primId, tmp_bary);
					if (t > 0 && t < min_T) {
						return true;
					}
				}
				cur_node++;
			}
			else {
				cur_node = devlinearNodes[cur_node].secondChild;
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

	GPU float sampleDirectLight(glm::vec3 pos, glm::vec4 random, glm::vec3 &radiance, glm::vec3 &wi) {
		int passId = int(float(devNumLightPrim) * random.x);
		DF<float> light = devLightDistribution[passId];
		int lightId = (random.y < light.prob) ? light.failId : passId;
		int lightPrimId = devLightPrimIds[lightId];
		glm::vec3 v0 = vertices[lightPrimId * 3];
		glm::vec3 v1 = vertices[lightPrimId * 3 + 1];
		glm::vec3 v2 = vertices[lightPrimId * 3 + 2];
		glm::vec3 samplePoint = sampleWithinTriangle(v0, v1, v2, glm::vec2(random.z, random.w));

		bool visible = !occlusionTest(pos, samplePoint);
		if (!visible) {
			return -1;
		}
		glm::vec3 lightNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
		glm::vec3 lightdir = samplePoint - pos;
		
		radiance = devLightUnitRadiance[lightId];
		wi = glm::normalize(lightdir);
		return luminance(radiance) / devSumLightPower * computeSolidAngle(pos, samplePoint, lightNormal);
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
	GPU void intersectAccel(const Ray& r, ShadeableIntersection& isect) {
		float min_T = FLT_MAX;
		int min_index = -1;
		glm::vec3 bary;
		int cur_node = 0;
		while (cur_node != devNumNodes) {

			AABB& bound = deviceBounds[devlinearNodes[cur_node].aabbIndex];
			if (bound.intersect(r, min_T)) {
				if (devlinearNodes[cur_node].primIndex != -1) {
					int primId = devlinearNodes[cur_node].primIndex;
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
				cur_node = devlinearNodes[cur_node].secondChild;
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
 
public:
    Scene(string filename);
    ~Scene();

	std::vector<Geom> geoms;
	std::vector<Material> materials;
	std::vector<int> materialIDs;
	std::vector<AABB> bounds;
	std::vector<LinearBVHNode> linearNodes;

	std::vector<int> lightPrimIds;
	std::vector<float> lightPower;
	std::vector<glm::vec3> lightUnitRadiance;
	float sumLightPower = 0;
	int numLightPrim = 0;
	DiscreteSampler1D<float> lightSampler;

	MeshData meshData;
	RenderState state;
	GPUScene hstScene;
	GPUScene* devScene = nullptr;
	void toDevice();
	void clearScene();
};