#include<cuda.h>
#include<iostream>
#include<curand.h>
#include<malloc.h>
#include<stdlib.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void prnt(float * arr, int n){

	for(int i=0; i<n;i++){
		printf("%f ",arr[i] );
	}
	printf("\n");
}
void prnt(float ** arr, int m, int n ){
	for(int j=0; j<m; j++){
		for(int i=0 ;i< n;i++){
			printf("%f ",arr[j][i] );
		}
		printf("\n");
	}
}
float **alloc2d(int m,int n, float *prev =NULL){
	float * arr = (float*) malloc(sizeof(float )*m*n);
	if (prev != NULL){
		memcpy(arr, prev, sizeof(float)*m*n);
	}
	float **brr = (float** ) malloc(sizeof(float*)*m);
	for(int i=0 ; i<m; i++){
		brr[i] = &arr[n*i];
	}
	return brr ;
}
__global__ void objectiveFn(float *res, float *harmonics, int dim, int pitch, int population){

//	cudaMalloc(&res, population);
	int index = blockIdx.x*blockDim.x*pitch+ threadIdx.x*pitch;
	int rindex = (int) index/pitch;
	if ( rindex>=population ){
		return ;
	}

	res[rindex] = 0 ;
	for(int i=0; i<dim;i++ ){
		res[rindex] += harmonics[index+i]*harmonics[index+i];
	}
}
__global__ void scale_vector(float *vec, int findex,int low, int high , int dim){

	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid >= dim){
			return;
	}
	vec[findex+tid] = low+vec[findex+tid]*(high-low);
}
__global__ void inrange(float *pop, size_t pitch, int low, int high, int population, int dim){
	/**
			pop: population array len:(population x dimensions)
			pitch: pitch for the memeory
			low bound of search space
			high er bound of search space
			**/
	int index = threadIdx.x*pitch+ (blockIdx.x*blockDim.x*pitch);
	if(index/pitch>= population){
		return ;
	}
	if (dim>1000){
		scale_vector<<<(int )dim/200+1,200>>>( pop, index, low, high, dim);
	}
	else{
		scale_vector<<<1,dim>>>( pop, index, low, high, dim);
	}
}
float* gen_random(curandGenerator_t gen, int row, int col, size_t *pitch,int low, int high){

	float *arr;
	cudaMallocPitch(&arr, pitch, sizeof(float)*col,sizeof(float)*row);
	curandGenerateUniform(gen, arr, *pitch*row);
	if(col < 1000){

		inrange<<<1, row>>>(arr, *pitch, low, high, row, col);
	}
	else{
		inrange<<<(int )row/200+1, 200>>>(arr, *pitch, low, high, row, col);
	}
	return arr;
}
__global__ void swap(float*a , float*b, int srcIndex, int destIndex, int dim){

	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if(id>=dim ){
		return ;
	}
	float temp ;
	temp = a[id+srcIndex];
	a[id+srcIndex] = b[id+destIndex];
	b[id+destIndex] = temp;
}
__global__ void sorted(float * res, float *harmonics, float* obj, int population , int dim, int pitch){

	//get an array for objectives values
	int myPos= blockIdx.x+blockDim.x+threadIdx.x;
	if(myPos>= population){
		return ;
	}

	//find resultant positions
	int countSmaller=0;
	float myobjective =  obj[myPos];

	for (int i=0 ; i< population; i++ ){
		if(myobjective>= obj[i] and i!=myPos){
			if(i< myPos && myobjective== obj[i]){
				continue;
			}
			countSmaller++;
		}
	}
	//for duplicate entries

	int srcIndex = myPos*pitch;
	int destIndex = countSmaller*pitch;
	if (dim>1000){
		swap<<<(int)dim/100,100>>>(res, harmonics, srcIndex, destIndex, dim);
	}
	else{
		swap<<<1, dim>>>(res, harmonics, srcIndex, destIndex, dim);
	}
}
__global__ void harmonic_update(float *harmonics, float *bests,float* noise, int hindex, int bindex, float brange, int dim , curandGenerator_t gen){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>= dim){
		return ;
	}
	harmonics[hindex+id] = bests[bindex+id]+ noise[hindex+id]*brange;
}

__global__ void update_harmonics( float *harmonics,float *bests,float *noise, int *randomRecs, float brange, int population,
		int nbgood, int dim, int pitch, curandGenerator_t gen ){

	//generate N(dim) random row and col indexes scale them
	int index = threadIdx.x+blockIdx.x*blockDim.x;
	if (index>=population){
		return;
	}
	int recIndex = randomRecs[index];
	int hindex = pitch*index;
	int bindex = recIndex*pitch;

	if(dim>1000){
		int blocks = dim%100 ==0 ? (int)dim/100 : ((int)dim/100)+1;
		harmonic_update<<<blocks, 100>>>(harmonics, bests, noise, hindex, bindex, brange,dim,gen);
	}
	else{
		harmonic_update<<<1, dim>>>( harmonics, bests, noise, hindex, bindex, brange, dim, gen);
	}

}

