#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "material.h"
using namespace std;
struct MeshData {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
};

struct Geom
{
    enum GeomType type;
    int materialid;
	MeshData* meshData;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

class GPUScene {
	glm::vec3* vertices = nullptr;
	glm::vec3* normals = nullptr;
	glm::vec2* uvs = nullptr;
    Material* materials = nullptr;
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
    RenderState state;

};