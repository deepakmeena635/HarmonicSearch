# [**Harmonics Search**](https://doi.org/10.1177%2F003754970107600201)

This repo contains a barly working CUDA implementation of [Harmonics Search Algorithm](https://doi.org/10.1177%2F003754970107600201).

## What's Needed: 
- nvcc 
- CUDA 6+/ Tested on 10 
- Compute_35 capable device
- essentially you should be able to run the following line 
	- $ nvcc - lcurand - lcudadevrt  - rdc=true - arch=compute_35 
	
## Compilation And Execution: 
- To run simply compile: nvcc - lcurand - lcudadevrt  - rdc=true - arch=compute_35 - o exec two.cu
- To run : ./exec

**Check out bugs section to find out known bugs or to correct them**

