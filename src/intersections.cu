#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

CPUGPU float triangleIntersectionTest(
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    Ray r,
    glm::vec3& bary
) {
	glm::vec3 e1 = v1 - v0;
	glm::vec3 e2 = v2 - v0;
	glm::vec3 s = r.origin - v0;
	glm::vec3 dir = r.direction;
	float denom = glm::dot(glm::cross(e1, dir), e2);
    if (abs(denom) < EPSILON) return -1;
	float u = -glm::dot(glm::cross(s, e2), dir) / denom;
	if (u < 0 || u > 1) return -1;
    float v = glm::dot(glm::cross(e1, s), dir) / denom;
	if (v < 0 || u + v > 1) return -1;
	float t = -glm::dot(glm::cross(s, e2), e1) / denom;
	bary = glm::vec3(1 - u - v, u, v);
    return t;
}

CPUGPU float meshIntersectionTest(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside) {
	float min_t = FLT_MAX;
    int triIndex = -1;
    glm::vec3 min_bary;

	//int num_tri = mesh.meshData->vertexCount;
 //   for (int i = 0; i < num_tri; ++i) {
	//	glm::vec3 v0 = mesh.meshData->vertices[i * 3];
	//	glm::vec3 v1 = mesh.meshData->vertices[i * 3 + 1];
	//	glm::vec3 v2 = mesh.meshData->vertices[i * 3 + 2];
	//	glm::vec3 bary;
	//	float t = triangleIntersectionTest(v0, v1, v2, r, bary);
 //       if (t > 0 && t < min_t) {
	//		min_t = t;
	//		triIndex = i;
	//		min_bary = bary;
 //       }
 //   }
 //   if (triIndex == -1) {
	//	outside = false;
	//	return -1;
 //   }
	//glm::vec3 n0 = mesh.meshData->vertices[triIndex * 3];
	//glm::vec3 n1 = mesh.meshData->vertices[triIndex * 3 + 1];
	//glm::vec3 n2 = mesh.meshData->vertices[triIndex * 3 + 2];
	//glm::vec2 uv0 = mesh.meshData->uvs[triIndex * 3];
	//glm::vec2 uv1 = mesh.meshData->uvs[triIndex * 3 + 1];
	//glm::vec2 uv2 = mesh.meshData->uvs[triIndex * 3 + 2];
	//intersectionPoint = getPointOnRay(r, min_t);
	//normal = glm::normalize(n0 * min_bary.x + n1 * min_bary.y + n2 * min_bary.z);
	//uv = uv0 * min_bary.x + uv1 * min_bary.y + uv2 * min_bary.z;
	//outside = true;
	return min_t;
}