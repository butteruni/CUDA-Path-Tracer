#pragma once
// ref: https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
#include <glm/glm.hpp>
#include "sceneStructs.h"
#include "macro.h"
#include "utilities.h"
struct AABB {
	glm::vec3 pmin, pmax;
	AABB() {
		pmin = glm::vec3(FLT_MAX);
		pmax = glm::vec3(-FLT_MAX);
	}
	CPUGPU AABB(const glm::vec3& p) : pmin(p), pmax(p) {}
	CPUGPU AABB(const glm::vec3& pmin, const glm::vec3& pmax) : pmin(pmin), pmax(pmax) {}
	CPUGPU AABB(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) {
		pmin = glm::min(p0, glm::min(p1, p2));
		pmax = glm::max(p0, glm::max(p1, p2));
	}
	CPUGPU void merge(const glm::vec3& p) {
		pmin = glm::min(pmin, p);
		pmax = glm::max(pmax, p);
	}
	CPUGPU void merge(const AABB& aabb) {
		pmin = glm::min(pmin, aabb.pmin);
		pmax = glm::max(pmax, aabb.pmax);
	}
	CPUGPU std::string toString() const {
		return "pmin: " + vec3ToString(pmin) + ", pmax: " + vec3ToString(pmax);
	}
	CPUGPU glm::vec3 extend() {
		return pmax - pmin;
	}
	CPUGPU glm::vec3 center() {
		return 0.5f * (pmin + pmax);
	}
	CPUGPU glm::vec3 offset(const glm::vec3& p) {
		glm::vec3 o = p - pmin;
		o /= extend();
		return o;
	}
	CPUGPU float surfaceArea() {
		glm::vec3 e = extend();
		return 2.f * (e.x * e.y + e.y * e.z + e.z * e.x);
	}
	CPUGPU int maxExtend() {
		glm::vec3 e = extend();
		if (e.x > e.y && e.x > e.z) {
			return 0;
		}
		else if (e.y > e.z) {
			return 1;
		}
		else {
			return 2;
		}
	}
	CPUGPU bool intersect(const Ray& r, float& tmin, float& tmax) {
		float t0 = 0.f, t1 = FLT_MAX;
		for (int i = 0; i < 3; i++) {
			float invD = 1.f / r.direction[i];
			float tNear = (pmin[i] - r.origin[i]) * invD;
			float tFar = (pmax[i] - r.origin[i]) * invD;
			if (tNear > tFar) {
				float tmp = tNear;
				tNear = tFar;
				tFar = tmp;
			}
			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar < t1 ? tFar : t1;
			if (t0 > t1) {
				return false;
			}
		}
		tmin = t0;
		tmax = t1;
		return true;
	}
	AABB Union(const AABB &a, const AABB &b) {
		return AABB(glm::min(a.pmin, b.pmin), glm::max(a.pmax, b.pmax));
	}
};	


struct treeInfo {
	int idx, left, right;
};
struct BVHNodeInfo {
	bool isLeaf;
	union {
		int index;
		int size;
	};
};
struct BucketInfo {
	int count = 0;
	AABB bounds;
};
struct LinearBVHNode {
	int aabbIndex;
	int axis;
	int nPrims;
	union {
		int primIndex;
		int secondChild;
	};
};
struct Prim {
	int primIndex;
	AABB aabb;
	Prim() = default;
	Prim(int primIndex, const AABB& aabb) : primIndex(primIndex), aabb(aabb) {}
};
enum class SplitMethod {
	SAH,
	HLBVH
};

struct BVHBuilder {
	static int build(const std::vector<glm::vec3>& vertices,
		std::vector<AABB>& aabbs, std::vector<LinearBVHNode>& linearNodes,
		SplitMethod method = SplitMethod::SAH);
	static void SAHBVHbuild(std::vector<Prim>& prims, std::vector<BVHNodeInfo>&nodes, std::vector<AABB>& aabbs);
	static void HLBVHbuild();
	static void flattenBVH(const std::vector<BVHNodeInfo> &nodes, std::vector<LinearBVHNode>&linearNodes);
};
