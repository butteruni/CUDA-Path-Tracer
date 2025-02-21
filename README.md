CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* MingLi Gu
* Tested on: Windows 11 23H2
  * AMD Ryzen 7 8845HS w 3.80 GHz
  * 32 RAM
  * NVIDIA GeForce RTX 4070 Laptop 8GB

## Example：

![](https://github.com/butteruni/CUDA-Path-Tracer/blob/main/img/camera.png?raw=true)


## Basic:

- Physical based Material

  

|                           Diffuse                            |
| :----------------------------------------------------------: |
| ![](https://github.com/butteruni/CUDA-Path-Tracer/blob/main/img/diffuse.png?raw=true) |
|                        **Conductor**                         |
| ![](https://github.com/butteruni/CUDA-Path-Tracer/blob/main/img/conductor.png?raw=true) |
|                        **Dieletric**                         |
| ![](https://github.com/butteruni/CUDA-Path-Tracer/blob/main/img/dielectric.png?raw=true) |

- Environment Map

  ​	![](https://github.com/butteruni/CUDA-Path-Tracer/blob/main/img/envmap.png?raw=true)

- Image based Texture

  ​	![](https://github.com/butteruni/CUDA-Path-Tracer/blob/main/img/image_based_texture.png?raw=true)

## Optimization:

- MTBVH acceleration

- Reshuffle by Material

  ![](https://github.com/butteruni/CUDA-Path-Tracer/blob/main/img/conductor_bunny.png?raw=true)

​		

|        Resolution        | 800 x 800               |
| :----------------------: | ----------------------- |
|          Speed           | 15.6 frames per seconds |
| Million Rays Per Seconds | 9.98                    |
|     Triangle Number      | 69642                   |
|           SPP            | 2000                    |



- MIS(multi-importance sampling)

  

