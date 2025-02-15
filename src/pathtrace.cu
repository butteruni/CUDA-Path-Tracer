#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "material.h"
#include "intersections.h"
#include "interactions.h"
#include "sampler.h"
#include "macro.h"


//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::vec3 color;
		color = image[index] / (float)iter;
		//color = uncharted2filmic(color);
        color = ACES(color);
        glm::ivec3 icolor = glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));
        // Each thread writes one pixel location in the texture (textel)

        pbo[index].w = 0;
        pbo[index].x = icolor.x;
        pbo[index].y = icolor.y;
        pbo[index].z = icolor.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
int* dev_material_ids = NULL;
thrust::device_ptr<int> dev_thrust_material_ids = NULL;
thrust::device_ptr<PathSegment> dev_thrust_paths;
thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections;
void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_material_ids, pixelcount * sizeof(int));
	dev_thrust_material_ids = thrust::device_pointer_cast(dev_material_ids);
	dev_thrust_paths = thrust::device_pointer_cast(dev_paths);
	dev_thrust_intersections = thrust::device_pointer_cast(dev_intersections);
    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_material_ids);
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
GPU Ray physical_light(const Camera& cam, int x, int y, thrust::default_random_engine &rng) {
    Ray ray;
    glm::vec4 r = sample4D(rng);
    float aspect = float(cam.resolution.x) / cam.resolution.y;
    float tanFovY = glm::tan(glm::radians(cam.fov.y));
    glm::vec2 pixelSize = 1.f / glm::vec2(cam.resolution);
    glm::vec2 scr = glm::vec2(x, y) * pixelSize;
    glm::vec2 ruv = scr + pixelSize * glm::vec2(r.x, r.y);
    ruv = 1.f - ruv * 2.f;

    glm::vec3 pLens = glm::vec3(squareToDiskConcentric(glm::vec2(r.z, r.w)) * cam.lensRadius, 0.f);
    glm::vec3 pFocusPlane = glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * cam.focalDist;
    glm::vec3 dir = pFocusPlane - pLens;
    ray.direction = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);
    ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;
    return ray;
}
GPU  glm::vec3 random_light_dir(const Camera& cam, int x, int y, thrust::default_random_engine & rng) {
    thrust::uniform_real_distribution<float> u01(-0.5f, 0.5f);
    float jitterX = u01(rng);
    float jitterY = u01(rng);
    return glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
    );
}
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];


        // TODO: implement antialiasing by jittering the ray
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        
        //segment.ray.direction = random_light_dir(cam, x, y, rng);
        segment.ray = physical_light(cam, x, y, rng);
        //segment.ray.origin = cam.position;
        segment.radiance = glm::vec3(0.f);
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersectionsScene(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    GPUScene* dev_scene,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    
   
    if (path_index >= num_paths) {
        return;
    }
    PathSegment pathSegment = pathSegments[path_index];
    ShadeableIntersection isect;
    dev_scene->intersectTest(pathSegment.ray, isect);
    
    if (isect.primitiveId != -1) {
        if (dev_scene->materials[isect.materialId].type == MaterialType::Light) {
            if (depth != 0) {
                isect.prev = pathSegment.ray.origin;
            }
        }
        else {
            isect.dir = -pathSegment.ray.direction;
        }
    }
    intersections[path_index] = isect;
}
// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.

__global__ void pathIntegrator(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    GPUScene *dev_scene) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }
    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment& segment = pathSegments[idx];
    if (intersection.t > 0.f) {
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
	    Material material = dev_scene->materials[intersection.materialId];
        segment.ray.origin = intersection.point;
        if (material.type == MaterialType::Light) {
            segment.color *= (material.color * material.emittance);
		    segment.remainingBounces = 0;
	    }
        else {
            BSDFSample sampler;
            material.SampleBSDF(intersection.surfaceNormal, intersection.dir, sample3D(rng), sampler);
            if (sampler.pdf < 0 || sampler.flags == Unset) {
                segment.remainingBounces = 0;
            }
            else {
                bool isDelta = sampler.flags & BxDFFlags::Specular;
                segment.color *= sampler.bsdf / sampler.pdf;
                if(!isDelta)
				    segment.color *= glm::dot(sampler.wi, intersection.surfaceNormal);
                segment.ray.direction = glm::normalize(sampler.wi);
			    segment.remainingBounces--;
            }
        }
    }
    else {
		segment.color = glm::vec3(0.0f);
		segment.remainingBounces = 0;
    }
	segment.ray.origin += EPSILON * segment.ray.direction;
}

__global__ void misPathIntegrator(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    GPUScene* dev_scene) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }
	ShadeableIntersection intersection = shadeableIntersections[idx];
	PathSegment& segment = pathSegments[idx];
    if (intersection.primitiveId == -1) {
        return;
    }
	thrust::default_random_engine rng(makeSeededRandomEngine(iter, idx, depth));
	Material material = dev_scene->materials[intersection.materialId];
    glm::vec3 sumRadiance(0.f);
    if (material.type == MaterialType::Light) {
		glm::vec3 radiance = material.color * material.emittance;
        if (depth == 0) {
			sumRadiance += radiance;
        }
        else if (segment.deltaSample) {
			sumRadiance += radiance * segment.color;
        }
        else {
            float lightPdf = luminance(radiance) * dev_scene->devSumLightPowerInv *
                computeSolidAngle(intersection.prev, intersection.point, intersection.surfaceNormal);
            float bsdfPdf = segment.pdf;
			float weight = powerHeuristic(bsdfPdf, lightPdf);
			sumRadiance += radiance * segment.color * weight;
        }
		segment.remainingBounces = 0;
    }
    else {
		bool deltaBSDF = (material.type == MaterialType::Dielectric);
        if (!deltaBSDF) {
			glm::vec3 radiance(0.f);
            glm::vec3 wi;
			float lightPdf = dev_scene->sampleDirectLight(intersection.prev, sample4D(rng), radiance, wi);
            if (lightPdf > 0) {
				glm::vec3 bsdf = material.BSDF(intersection.surfaceNormal, intersection.prev, wi);
				float bsdfPdf = material.pdf(intersection.surfaceNormal, intersection.prev, wi);
				float weight = powerHeuristic(lightPdf, bsdfPdf);
                sumRadiance += segment.color * bsdf * radiance * weight * max(0.f, glm::dot(intersection.surfaceNormal, wi)) / lightPdf;
            }
        }
		BSDFSample sample;
		material.SampleBSDF(intersection.surfaceNormal, intersection.prev, sample3D(rng), sample);
        if (sample.flags != Unset) {
			bool deltaSample = sample.flags & BxDFFlags::Specular;
			segment.color *= sample.bsdf / sample.pdf;
			if (!deltaSample)
                segment.color *= glm::abs(glm::dot(sample.wi, intersection.surfaceNormal));
			segment.ray.direction = glm::normalize(sample.wi);
			segment.ray.origin = intersection.point + EPSILON * segment.ray.direction;
			segment.deltaSample = deltaSample;
			segment.pdf = sample.pdf;
			segment.remainingBounces--;
        }
        else {
			segment.remainingBounces = 0;
        }
    }
    segment.radiance += sumRadiance;
}
// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}
struct IsPathRunning {
    CPUGPU bool operator()(const PathSegment& path) const {
        return path.remainingBounces != 0;
    }
};
struct sortByMaterial {
    CPUGPU bool operator()(
        const ShadeableIntersection& a, 
        const ShadeableIntersection& b) {
        return a.materialId < b.materialId;
    }
};
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        /*computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );*/
        computeIntersectionsScene << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
			hst_scene->devScene,
			dev_intersections
			);
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        // reshuffle material
        if (RESUFFLE_BY_MATERIAL) {
            thrust::sort_by_key(
                thrust::device_pointer_cast(dev_intersections),
                thrust::device_pointer_cast(dev_intersections + num_paths),
                thrust::make_zip_iterator(
                    thrust::make_tuple(
                        thrust::device_pointer_cast(dev_paths),
                        thrust::device_pointer_cast(dev_intersections)
                    )
                ),
                sortByMaterial()
            
            );
        }
		pathIntegrator << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
            depth,
			num_paths,
			dev_intersections,
			dev_paths,
			hst_scene->devScene
			);
		//misPathIntegrator << <numblocksPathSegmentTracing, blockSize1d >> > (
		//	iter,
		//	depth,
		//	num_paths,
		//	dev_intersections,
		//	dev_paths,
		//	hst_scene->devScene
		//	);
        checkCUDAError("shading");
        cudaDeviceSynchronize();
        PathSegment* new_end = thrust::stable_partition(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            IsPathRunning()
        );
        cudaDeviceSynchronize();
        // iterationComplete = true; // TODO: should be based off stream compaction results.
        int existing_paths = new_end - dev_paths;
        num_paths = existing_paths;
        iterationComplete = (num_paths == 0) || (depth >= traceDepth);
        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
