#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "macro.h"
#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
	CUBE,
    MESH,
};
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
struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
    CPUGPU glm::vec3 at(float t) {
        return origin + t * direction;
    }
};
CPUGPU inline Ray makeSteppedRay(glm::vec3& p, glm::vec3& dir) {
    return Ray{ p + EPSILON * dir, dir };
}
struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius;
    float focalDist;
    float tanFovY;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    glm::vec3 radiance;
    int pixelIndex;
    int remainingBounces;
	float pdf;
    bool deltaSample = false;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 point;
  glm::vec3 dir;
  glm::vec3 surfaceNormal;
  glm::vec2 uv;
  glm::vec3 prev;
  int primitiveId = -1;
  int materialId;
  float deltaSample = false;
  float pdf;
  GPU void operator = (const ShadeableIntersection& rhs) {
	  t = rhs.t;
	  point = rhs.point;
	  dir = rhs.dir;
	  surfaceNormal = rhs.surfaceNormal;
	  uv = rhs.uv;
	  primitiveId = rhs.primitiveId;
	  materialId = rhs.materialId;
	  prev = rhs.prev;
      deltaSample = rhs.deltaSample;
      pdf = rhs.pdf;
  }
};
