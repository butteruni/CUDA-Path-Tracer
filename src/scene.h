#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "material.h"
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

	GPU void intersectTest(const Ray& r, ShadeableIntersection& isect) {
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
		if (min_index != -1) {
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
	MeshData meshData;
	RenderState state;
	GPUScene hstScene;
	GPUScene* devScene = nullptr;
	void toDevice();
	void clearScene();
};