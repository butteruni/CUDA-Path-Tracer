CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* MingLi Gu
* Tested on: Windows 11 23H2
  * AMD Ryzen 7 8845HS w 3.80 GHz
  * 32 RAM
  * NVIDIA GeForce RTX 4070 Laptop 8GB


## Basic:

- Physical based Material

  

|                           Diffuse                            |
| :----------------------------------------------------------: |
| ![](https://github.com/butteruni/CUDA-Path-Tracer/blob/main/img/diffuse.png?raw=true) |
|                        **Conductor**                         |
| ![](https://github.com/butteruni/CUDA-Path-Tracer/blob/main/img/conductor.png?raw=true) |
|                        **Dieletric**                         |
| ![](https://github.com/butteruni/CUDA-Path-Tracer/blob/main/img/dielectric.png?raw=true) |



## Optimization:

- MTBVH acceleration

- Reshuffle by Material

  ![](https://github.com/butteruni/CUDA-Path-Tracer/blob/main/img/conductor_bunny.png?raw=true)

​	69642 triangles rendered in 2.4fps(about 400ms per frame)

- MIS(multi-importance sampling)

  ![](https://github.com/butteruni/CUDA-Path-Tracer/blob/main/img/mis_diffuse.png?raw=true)

## Todo Lists:

- environment map
- image based texture
- multi-importance sampling
