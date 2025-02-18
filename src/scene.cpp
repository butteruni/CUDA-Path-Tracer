#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "tiny_obj_loader.h"
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
			newMaterial.type = Lambertian;
        }
        else if (p["TYPE"] == "Emitting")
        {
            newMaterial.emittance = p["EMITTANCE"];
			newMaterial.type = Light;
        }
        else if (p["TYPE"] == "Conductor")
        {
			newMaterial.roughness = p["ROUGHNESS"];
			newMaterial.metallic = p["METALLIC"];
			newMaterial.type = Conductor;
		}
		else if (p["TYPE"] == "Dielectric")
		{
			newMaterial.indexOfRefraction = p["IOR"];
			newMaterial.type = Dielectric;
		}
        const auto& col = p["RGB"];
        if (col.size() > 2) {
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.colorTextureId = -1;
		}
		else {
            if (col.is_string()) {
                newMaterial.colorTextureId = getTextureIndex(col);
            }
            else {
				std::cout << "illegal color format" << col << std::endl;
                exit(-1);
            }
		}
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
		else if (type == "sphere")
        {
            newGeom.type = SPHERE;
		}
		else if (type == "mesh")
		{
			newGeom.type = MESH;
			loadMesh(p["MESH"], newGeom);
		}
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    camera.focalDist = cameraData["FOCALDIST"];
    camera.lensRadius = cameraData["LENSRADIUS"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);
    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);
    camera.tanFovY = glm::tan(glm::radians(fovy * 0.5f));
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadMesh(const std::string& meshName, Geom& dst_data)
{
    if (meshMap.find(meshName) != meshMap.end()) {
		dst_data.meshData = meshMap[meshName];
        std::cout << "Loading mesh from mesh Map\n";
		return;
    }
    dst_data.meshData = new MeshData();
	std::cerr << "Loading mesh from " << meshName << std::endl;
	if (meshName.substr(meshName.find_last_of('.')) == ".obj")
	{
        loadMeshFromObj(meshName, dst_data.meshData);
    }
    else {
		cout << "Couldn't read mesh from " << meshName << endl;
		exit(-1);
    }
	meshMap[meshName] = dst_data.meshData;
}

Image* Scene::loadTexture(const std::string& textureName) {
	if (textureMap.find(textureName) != textureMap.end()) {
		return textureMap[textureName];
	}
	Image* texture = new Image(textureName);
	textureMap[textureName] = texture;
	textureIds[textureName] = textures.size();
	textures.push_back(texture);
	return texture;
}

int Scene::getTextureIndex(const std::string& textureName) {
    if (textureIds.find(textureName) != textureIds.end()) {
        return textureIds[textureName];
    }
    int id = textureIds.size();
    Image* texture = loadTexture(textureName);
	textureIds[textureName] = id;
	return id;
}

void Scene::loadMeshFromObj(const std::string& meshName, MeshData* dst_data) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::string warn, err;
	if (!tinyobj::LoadObj(&attrib, &shapes, nullptr, &warn, &err, meshName.c_str()))
	{
		std::cerr << warn << err << std::endl;
		exit(-1);
	}
	bool hasNormals = attrib.normals.size() > 0;
	bool hasUVs = attrib.texcoords.size() > 0;
	for (const auto& shape : shapes)
	{
		for (const auto& index : shape.mesh.indices)
		{
			glm::vec3 vertex = glm::vec3(attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]);
			glm::vec3 normal = glm::vec3(0);
            if (hasNormals) {
			     normal = glm::vec3(attrib.normals[3 * index.normal_index + 0],
				    attrib.normals[3 * index.normal_index + 1],
				    attrib.normals[3 * index.normal_index + 2]);
            }
            else {
				normal = glm::vec3(0, 0, 0);
            }
			glm::vec2 uv = glm::vec2(0);
			if (hasUVs) {
				uv = glm::vec2(attrib.texcoords[2 * index.texcoord_index + 0],
					attrib.texcoords[2 * index.texcoord_index + 1]);
            }
            else {
				uv = glm::vec2(0, 0);
            }
			dst_data->vertices.push_back(vertex);
			dst_data->normals.push_back(normal);
			dst_data->uvs.push_back(uv);
		}
	}
	std::cout << "Loaded mesh from " << meshName << std::endl;
    return;
}

void Scene::toDevice()
{
    int primId = 0;
    for (auto& geom : geoms) {
        if (geom.type != MESH)
            continue;
        glm::vec3 radianceUnit = materials[geom.materialid].color * materials[geom.materialid].emittance;
        float powerUnit = luminance(radianceUnit);
        for (size_t i = 0; i < geom.meshData->vertices.size(); i++) {
            meshData.vertices.push_back(glm::vec3(geom.transform * glm::vec4(geom.meshData->vertices[i], 1.0f)));
            meshData.normals.push_back(glm::vec3(geom.invTranspose * glm::vec4(geom.meshData->normals[i], 0.0f)));
            meshData.uvs.push_back(geom.meshData->uvs[i]);
            if (i % 3 == 0) {
                materialIDs.push_back(geom.materialid);
			}
			else if (i % 3 == 2 && materials[geom.materialid].type == Light) {
				glm::vec3 v0 = meshData.vertices[i - 2];
				glm::vec3 v1 = meshData.vertices[i - 1];
				glm::vec3 v2 = meshData.vertices[i];
				float area = triangleArea(v0, v1, v2);
				float power = powerUnit * area;
				lightPrimIds.push_back(primId);
				lightUnitRadiance.push_back(radianceUnit);
				lightPower.push_back(power);
				sumLightPower += power;
				numLightPrim++;
                primId++;
			}
        }
    }
	std::cout << meshData.vertices.size() / 3 << " triangles" << std::endl;
	std::cout << lightPrimIds.size() << " light primitives" << std::endl;
	int bvhsize = BVHBuilder::build(meshData.vertices, bounds, linearNodes, SplitMethod::SAH);

	lightSampler = DiscreteSampler1D<float>(lightPower);
    hstScene.loadFromScene(*this);
    cudaMalloc(&devScene, sizeof(GPUScene));
    cudaMemcpy(devScene, &hstScene, sizeof(GPUScene), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    checkCUDAError("loadscene");
	meshData.normals.clear();
	meshData.uvs.clear();
	meshData.vertices.clear();
    lightPrimIds.clear();
	lightUnitRadiance.clear();
	lightPower.clear();
    lightSampler.binomDistribution.clear();
	sumLightPower = 0;
	numLightPrim = 0;
    cudaDeviceSynchronize();
}


void GPUScene::loadFromScene(const Scene& scene) {
    cudaDeviceSynchronize();
    verticesSize = scene.meshData.vertices.size();



    cudaMalloc(&vertices, getVectorByteSize(scene.meshData.vertices));
    cudaMemcpy(vertices, scene.meshData.vertices.data(), getVectorByteSize(scene.meshData.vertices), cudaMemcpyHostToDevice);

    cudaMalloc(&normals, getVectorByteSize(scene.meshData.normals));
    cudaMemcpy(normals, scene.meshData.normals.data(), getVectorByteSize(scene.meshData.normals), cudaMemcpyHostToDevice);

    cudaMalloc(&uvs, getVectorByteSize(scene.meshData.uvs));
    cudaMemcpy(uvs, scene.meshData.uvs.data(), getVectorByteSize(scene.meshData.uvs), cudaMemcpyHostToDevice);
    checkCUDAError("load vertices");


    cudaMalloc(&materials, getVectorByteSize(scene.materials));
    cudaMemcpy(materials, scene.materials.data(), getVectorByteSize(scene.materials), cudaMemcpyHostToDevice);

    cudaMalloc(&materialIDs, getVectorByteSize(scene.materialIDs));
    cudaMemcpy(materialIDs, scene.materialIDs.data(), getVectorByteSize(scene.materialIDs), cudaMemcpyHostToDevice);
    checkCUDAError("load material");

    int textureSize = 0;
	std::vector<GPUImage> devTextures;
	for (auto& texture : scene.textures) {
		textureSize += texture->getSize();
	}
	cudaMalloc(&texturePixels, textureSize);
    int offset = 0;
	for (auto& texture : scene.textures) {
		cudaMemcpy(texturePixels, texture->pixels, texture->getSize(), cudaMemcpyHostToDevice);
		devTextures.push_back(GPUImage(texture, texturePixels + offset));
        offset += texture->getSize() / sizeof(glm::vec3);
	}
	cudaMalloc(&textures, getVectorByteSize(devTextures));
	cudaMemcpy(textures, devTextures.data(), getVectorByteSize(devTextures), cudaMemcpyHostToDevice);
	checkCUDAError("load texture");


	cudaMalloc(&devLightPrimIds, getVectorByteSize(scene.lightPrimIds));
	cudaMemcpy(devLightPrimIds, scene.lightPrimIds.data(), getVectorByteSize(scene.lightPrimIds), cudaMemcpyHostToDevice);
	cudaMalloc(&devLightUnitRadiance, getVectorByteSize(scene.lightUnitRadiance));
	cudaMemcpy(devLightUnitRadiance, scene.lightUnitRadiance.data(), getVectorByteSize(scene.lightUnitRadiance), cudaMemcpyHostToDevice);
	cudaMalloc(&devLightDistribution, getVectorByteSize(scene.lightSampler.binomDistribution));
	cudaMemcpy(devLightDistribution, scene.lightSampler.binomDistribution.data(), getVectorByteSize(scene.lightSampler.binomDistribution), cudaMemcpyHostToDevice);
    devNumLightPrim = scene.numLightPrim;
	devSumLightPower = scene.sumLightPower;
    devSumLightPowerInv = 1.f / devSumLightPower;
    printf("lightSumInv: %f\n", devSumLightPowerInv);
    checkCUDAError("load Light sample");

	devNumNodes = scene.linearNodes[0].size();
    cudaMalloc(&deviceBounds, getVectorByteSize(scene.bounds));
	cudaMemcpy(deviceBounds, scene.bounds.data(), getVectorByteSize(scene.bounds), cudaMemcpyHostToDevice);
    checkCUDAError("load bounds");
    std::vector<LinearBVHNode> flattened;
    flattened.clear();
    for (auto& nodes : scene.linearNodes) {
        for (auto& node : nodes) {
            flattened.push_back(node);
        }
    }
    cudaMalloc(&devlinearNodes, getVectorByteSize(flattened));
    checkCUDAError("malloc linear");

    cudaMemcpy(devlinearNodes, flattened.data(), getVectorByteSize(flattened), cudaMemcpyHostToDevice);
	checkCUDAError("load linear");
}

void GPUScene::clear() {
    safeCudaFree(vertices);
    safeCudaFree(normals);
    safeCudaFree(uvs);
    safeCudaFree(materials);
    safeCudaFree(materialIDs);
	safeCudaFree(deviceBounds);
	safeCudaFree(devlinearNodes);
	safeCudaFree(devLightPrimIds);
	safeCudaFree(devLightUnitRadiance);
	safeCudaFree(devLightDistribution);
}
void Scene::clearScene() {
    hstScene.clear();
    safeCudaFree(devScene);
}