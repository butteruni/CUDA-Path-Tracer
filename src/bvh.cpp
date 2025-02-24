#include <stack>
#include "bvh.h"
int BVHBuilder::build(const std::vector<glm::vec3>& vertices, std::vector<AABB>& aabbs, 
	std::vector<std::vector<LinearBVHNode>>& linearNodes, SplitMethod method) {
	std::cout << "Building BVH\n";
	int faceSize = vertices.size() / 3;
	int maxBVHSize = 2 * faceSize - 1;
	aabbs.resize(maxBVHSize);
	std::vector<Prim> prims(faceSize);
	for (int i = 0; i < faceSize; i++) {
		prims[i] = { i, AABB(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]) };
		//prims[i].aabb.merge(prims[i].aabb.pmin - glm::vec3(EPSILON));
		//prims[i].aabb.merge(prims[i].aabb.pmax + glm::vec3(EPSILON));

	}
	std::vector<BVHNodeInfo> nodeInfos(maxBVHSize);
	switch (method)
	{
	case SplitMethod::HLBVH:
		break;
	case SplitMethod::SAH:
		SAHBVHbuild(prims, nodeInfos, aabbs);
		break;
	case SplitMethod::EQUAL:
		EQUALBVHbuild(prims, nodeInfos, aabbs);
	default:
		break;
	}
	std::cout << "BVHSize: " << nodeInfos.size() << "\n";
	flattenBVH(aabbs, nodeInfos, linearNodes);
	std::cout << "BVH flatten finish\n";
	return maxBVHSize;
}
void BVHBuilder::EQUALBVHbuild(std::vector<Prim>& prims, std::vector<BVHNodeInfo>& nodeInfos, std::vector<AABB>& aabbs) {
	aabbs.clear();
	aabbs.resize(nodeInfos.size());
	std::stack<treeInfo> stk;
	stk.push({ 0, 0, (int)prims.size() - 1 });
	int depth = 0;
	while (stk.size()) {
		depth = std::max(depth, (int)stk.size());
		treeInfo info = stk.top();
		stk.pop();
		int primSize = info.right - info.left + 1;
		int nodeSize = primSize * 2 - 1;
		bool isLeaf = (nodeSize == 1);
		nodeInfos[info.idx] = { isLeaf, isLeaf ? prims[info.left].primIndex : nodeSize };
		AABB nodeAABB;
		AABB centroid;
		for (int i = info.left; i <= info.right; i++) {
			nodeAABB.merge(prims[i].aabb);
			centroid.merge(prims[i].aabb.center());
		}
		aabbs[info.idx] = nodeAABB;
		if (isLeaf) continue;
		int splitAxis = centroid.maxExtend();
		if (nodeSize <= 2) {
			int mid = nodeSize / 2;
			aabbs[info.idx + 1] = prims[info.left].aabb;
			aabbs[info.idx + 1 + mid] = prims[info.right].aabb;
			nodeInfos[info.idx + 1] = { true, prims[info.left].primIndex };
			nodeInfos[info.idx + 1 + mid] = { true, prims[info.right].primIndex };
		}
		else {
			std::sort(prims.begin() + info.left + 1, prims.begin() + info.right + 1, [splitAxis](const Prim& a, const Prim& b) {
				return a.aabb.center()[splitAxis] < b.aabb.center()[splitAxis];
				});
			int div = info.left + (primSize / 2);
			div = glm::clamp(div - 1, info.left, info.right - 1);
			int lsize = 2 * (div - info.left + 1) - 1;
			stk.push({ info.idx + 1 + lsize, div + 1, info.right });
			stk.push({ info.idx + 1, info.left, div });
		}

	}
	std::cout << "BVH depth: " << depth << '\n';
}
void BVHBuilder::SAHBVHbuild(std::vector<Prim>& prims, std::vector<BVHNodeInfo>& nodeInfos, std::vector<AABB>& aabbs) {
	aabbs.clear();
	aabbs.resize(nodeInfos.size());
	std::stack<treeInfo> stk;
	stk.push({ 0, 0, (int)prims.size() - 1 });
	int depth = 0;
	while (stk.size()) {
		depth = std::max(depth, (int)stk.size());
		treeInfo info = stk.top();
		stk.pop();
		int primSize = info.right - info.left + 1;
		int nodeSize = primSize * 2 - 1;
		bool isLeaf = (nodeSize == 1);
		nodeInfos[info.idx] = { isLeaf, isLeaf ? prims[info.left].primIndex : nodeSize};
		AABB nodeAABB;
		AABB centroid;
		for (int i = info.left; i <= info.right; i++) {
			nodeAABB.merge(prims[i].aabb);
			centroid.merge(prims[i].aabb.center());
		}
		aabbs[info.idx] = nodeAABB;
		if (isLeaf) continue;
		int splitAxis = centroid.maxExtend();
		if (nodeSize <= 2) {
			int mid = nodeSize / 2;
			if (prims[info.left].aabb.center()[splitAxis] > prims[info.right].aabb.center()[splitAxis])
				std::swap(prims[info.left], prims[info.right]);
			aabbs[info.idx + 1] = prims[info.left].aabb;
			aabbs[info.idx + 1 + mid] = prims[info.right].aabb;
			nodeInfos[info.idx + 1] = { true, prims[info.left].primIndex };
			nodeInfos[info.idx + 1 + mid] = { true, prims[info.right].primIndex };
		}
		else {

			const int bucket_size = 12;
			BucketInfo buckets[bucket_size];
			for (int i = info.left; i <= info.right; i++) {
				int b = bucket_size * centroid.offset(prims[i].aabb.center())[splitAxis];
				b = glm::clamp(b, 0, bucket_size - 1);
				buckets[b].count++;
				buckets[b].bounds.merge(prims[i].aabb);
			}
			constexpr int nSplits = bucket_size - 1;
			float costs[nSplits] = { 0 };
			int countBelow = 0, countAbove = 0;
			AABB boundBelow, boundAbove;
			for (int i = 0; i < nSplits; ++i) {
				boundBelow.merge(buckets[i].bounds);
				countBelow += buckets[i].count;
				costs[i] += countBelow * boundBelow.surfaceArea();
			}
			for (int i = nSplits - 1; i >= 0; --i) {
				boundAbove.merge(buckets[i + 1].bounds);
				countAbove += buckets[i + 1].count;
				costs[i] += countAbove * boundAbove.surfaceArea();
			}
			int minCostSplitBucket = 0;
			float minCost = FLT_MAX;
			for (int i = 0; i < nSplits; ++i) {
				if (costs[i] < minCost) {
					minCost = costs[i];
					minCostSplitBucket = i;
				}
			}
			std::vector<Prim> subPrim(primSize);
			memcpy(subPrim.data(), prims.data() + info.left, primSize * sizeof(Prim));
			int l = info.left, r = info.right;
			for (int i = 0; i < primSize; ++i) {
				int b = bucket_size * centroid.offset(subPrim[i].aabb.center())[splitAxis];
				b = glm::clamp(b, 0, bucket_size - 1);
				if (b > minCostSplitBucket) {
					prims[r--] = subPrim[i];
				}
				else {
					prims[l++] = subPrim[i];
				}
			}
			l = glm::clamp(l - 1, info.left, info.right - 1);
			int lsize = 2 * (l - info.left + 1) - 1;
			stk.push({ info.idx + 1 + lsize, l + 1, info.right });
			stk.push({ info.idx + 1, info.left, l});
		}
		
	}
	std::cout << "BVH depth: " << depth << '\n';
}

void BVHBuilder::flattenBVH(
	const std::vector<AABB>& aabbs,
	const std::vector<BVHNodeInfo>& nodeInfos, 
	std::vector<std::vector<LinearBVHNode>>& linearNodes) {
	linearNodes.clear();
	linearNodes.resize(nodeInfos.size());
	linearNodes.clear();
	linearNodes.resize(6);
	for (int i = 0; i < 6; ++i) {
		linearNodes[i].resize(aabbs.size());
	}
	std::stack<int>stk;
	for (int i = 0; i < 6; ++i) {
		int offset = 0;
		auto& nodes = linearNodes[i];
		stk.push(0);
		while (stk.size()) {
			int idx = stk.top();
			stk.pop();
			BVHNodeInfo node = nodeInfos[idx];
			bool isLeaf = node.isLeaf;
		
			int nodeSize = isLeaf ? 1 : node.size;
			nodes[offset].aabbIndex = idx;
			nodes[offset].primIndex = isLeaf ? node.index : -1;
			nodes[offset].secondChild = offset + nodeSize;
			offset++;
			if (!isLeaf) {
				int left = idx + 1;
				bool leftLeaf = nodeInfos[left].isLeaf;
				int leftSize = leftLeaf ? 1 : nodeInfos[left].size;
				int right = left + leftSize;
				int dim = i / 2;
				bool less = i & 1;
				if ((aabbs[left].center()[dim] < aabbs[right].center()[dim])
					^ less) {
					std::swap(left, right);
				}
				stk.push(right);
				stk.push(left);
			}
		}
	}
}