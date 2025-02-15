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

  

|          Diffuse          |
| :-----------------------: |
|  ![](.\img\diffuse.png)   |
|       **Conductor**       |
| ![](.\img\conductor.png)  |
|       **Dieletric**       |
| ![](.\img\dielectric.png) |



## Optimization:

- MTBVH acceleration

- Reshuffle by Material

  ![](D:\Github\self_repos\CUDA-Path-Tracer\img\counductor_bunny.png)

​	69642 triangles rendered in 2.4fps(about 400ms per frame)
