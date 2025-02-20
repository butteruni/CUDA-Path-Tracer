#pragma once

#include <glm/glm.hpp>
#include <string>
#include <stb_image.h>
#include <iostream>
#include "macro.h"
using namespace std;

class Image
{
public:
    int xSize = 0;
    int ySize = 0;
    glm::vec3 *pixels;

    Image(int x, int y);
    ~Image();
    int getSize() const {
		return xSize * ySize * sizeof(glm::vec3);
    }
    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);
    Image(const std::string& filename) {
		float* data = stbi_loadf(filename.c_str(), &xSize, &ySize, nullptr, 3);
        if (!data) {
			std::cout << "Failed to load image: " << filename << std::endl;
            exit(-1);
        }
		pixels = new glm::vec3[xSize * ySize];
		memcpy(pixels, data, xSize * ySize * sizeof(glm::vec3));
        if (data) {
            stbi_image_free(data);
        }
    }
    glm::vec3 getPixel(int x, int y) const {
		return pixels[y * xSize + x];
    }
};
class GPUImage {
public:
    int xSize, ySize;
	glm::vec3* pixels;
	GPUImage() = default;
    GPUImage(Image* img, glm::vec3* devPixels) {
		xSize = img->xSize;
		ySize = img->ySize;
		pixels = devPixels;
    }
    GPU glm::vec3 getPixel(int x, int y) const {
		return pixels[y * xSize + x];
    }
    GPU glm::vec3 linearSample(float x, float y) const{

        x = glm::fract(x), y = glm::fract(y);
		float fx = x * xSize;
		float fy = y * ySize;

		int x0 = glm::fract(fx) > 0.5 ? glm::ceil(fx) : glm::floor(fx);
        int y0 = glm::fract(fy) > 0.5 ? glm::ceil(fy) : glm::floor(fy);
		int x1 = x0 + 1;
		int y1 = y0 + 1;
        float dx = glm::fract(fx);
        float dy = glm::fract(fy);

        glm::vec3 p00 = getPixel(x0, y0);
        glm::vec3 p01 = getPixel(x0, y1);
        glm::vec3 p10 = getPixel(x1, y0);
        glm::vec3 p11 = getPixel(x1, y1);


        glm::vec3 interpolatedX0 = glm::mix(p00, p10, dx);  
        glm::vec3 interpolatedX1 = glm::mix(p01, p11, dx);  
        glm::vec3 result = glm::mix(interpolatedX0, interpolatedX1, dy);  

        return result;
    }
    GPU glm::vec3 linearSample(const glm::vec2 sample) const {

        float x = glm::fract(sample.x), y = glm::fract(sample.y);
        float fx = x * xSize;
        float fy = y * ySize;

        int x0 = glm::fract(fx) > 0.5 ? glm::ceil(fx) : glm::floor(fx);
        int y0 = glm::fract(fy) > 0.5 ? glm::ceil(fy) : glm::floor(fy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float dx = glm::fract(fx);
        float dy = glm::fract(fy);

        glm::vec3 p00 = getPixel(x0, y0);
        glm::vec3 p01 = getPixel(x0, y1);
        glm::vec3 p10 = getPixel(x1, y0);
        glm::vec3 p11 = getPixel(x1, y1);


        glm::vec3 interpolatedX0 = glm::mix(p00, p10, dx);
        glm::vec3 interpolatedX1 = glm::mix(p01, p11, dx);
        glm::vec3 result = glm::mix(interpolatedX0, interpolatedX1, dy);

        return result;
    }
};