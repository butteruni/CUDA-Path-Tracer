#pragma once
// ref: https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
#include <glm/glm.hpp>
#include <thrust/swap.h>
#include "sceneStructs.h"
#include "macro.h"
#include "utilities.h"
struct AABB {
	glm::vec3 pmin = glm::vec3(FLT_MAX);
	glm::vec3 pmax = glm::vec3(-FLT_MAX);
	AABB() = default;
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
	CPUGPU glm::vec3 extend() const{
		if (pmin.x > pmax.x) {
			return glm::vec3(0.f);
		}
		return pmax - pmin;
	}
	CPUGPU glm::vec3 center() const{
		return 0.5f * (pmin + pmax);
	}
	CPUGPU glm::vec3 offset(const glm::vec3& p) const{
		if (pmin.x > pmax.x) {
			return p;
		}
		glm::vec3 o = p - pmin;
		o /= extend();
		return o;
	}
	CPUGPU float surfaceArea() const{
		glm::vec3 e = extend();
		return 2.f * (e.x * e.y + e.y * e.z + e.z * e.x);
	}
	CPUGPU int maxExtend() const{
		glm::vec3 e = extend();
		if (e.x < e.y) {
			return e.y > e.z ? 1 : 2;
		}
		return e.x > e.z ? 0 : 2;
	}
	CPUGPU bool intersect(const Ray& r, float& tmax) const{
		glm::vec3 o = r.origin;
		const glm::vec3 invDir = 1.0f / r.direction;
		float tEnter = FLT_MIN;
		float tExit = tmax;

		for (int i = 0; i < 3; ++i)
		{
			float tMax;
			float tMin;
			if (r.direction[i] == 0)
			{
				if (o[i] < pmin[i] || o[i] > pmax[i]) return false;
			}
			else
			{
				tMax = (pmax[i] - o[i]) * invDir[i];
				tMin = (pmin[i] - o[i]) * invDir[i];
				if (r.direction[i] < 0) thrust::swap(tMax, tMin);
				tEnter = tMin > tEnter ? tMin : tEnter;
				tExit = tMax < tExit ? tMax : tExit;
			}
		}
		tmax = tEnter;
		return tEnter <= tExit && tExit > 0;
	}
	AABB Union(const AABB &a, const AABB &b) {
		return AABB(glm::min(a.pmin, b.pmin), glm::max(a.pmax, b.pmax));
	}
};	


struct treeInfo {
	int idx;
	int left, right;
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
	int primIndex;
	int secondChild;
};
struct TreeBVHNode {
	int aabbIndex;
	int primIndex;
	int left, right;
};
struct Prim {
	int primIndex;
	AABB aabb;
	Prim() = default;
	Prim(int primIndex, const AABB& aabb) : primIndex(primIndex), aabb(aabb) {}
};
enum class SplitMethod {
	SAH,
	EQUAL,
	HLBVH,
};

struct BVHBuilder {
	static int build(const std::vector<glm::vec3>& vertices, std::vector<AABB>& aabbs,
		std::vector<std::vector<LinearBVHNode>>& linearNodes, SplitMethod method = SplitMethod::EQUAL);
	static int buildTree(const std::vector<glm::vec3> &vertices, std::vector<AABB>& aabbs
		, std::vector<TreeBVHNode>& nodes);
	static void SAHBVHbuild(std::vector<Prim>& prims, std::vector<BVHNodeInfo>&nodes, std::vector<AABB>& aabbs);
	static void EQUALBVHbuild(std::vector<Prim>& prims, std::vector<BVHNodeInfo>& nodes, std::vector<AABB>& aabbs);
	static void HLBVHbuild();
	static void flattenBVH(const std::vector<AABB>& aabbs,const std::vector<BVHNodeInfo> &nodes, std::vector<std::vector<LinearBVHNode>>& linearNodes);
};
