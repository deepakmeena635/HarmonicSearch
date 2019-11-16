#include<cuda.h>
#include<iostream>
#include<curand.h>
#include<malloc.h>
#include<stdlib.h>
#include"hsutils.h"

using namespace std;


void test_gen_random(int pop, int dim, float low, float high){

	printf("--------Random gen and vector_scale and inrange tested together -------\n");
		
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	size_t pitch; 
	
	float* arr = gen_random(gen, pop, dim, &pitch, low, high);
	
	float brr[pop][dim];
	cudaMemcpy2D(brr, dim*sizeof(float), arr, pitch*sizeof(float), dim*sizeof(float),pop, cudaMemcpyDeviceToHost);
	
	for(int i=0; i< pop; i++){
		for(int j=0 ; j< dim; j++){
			cout<<" "<<brr[i][j];
		}
		cout<<endl; 
	}
	printf("----------------DONE////////////////////////::::::::-__-__-__-\n");
}

void test_alloc2d(){

	printf("---------------------alloc2d test-----------------\n");
	int n= 7, m=5;
	float *arr =(float *) malloc(sizeof(float)*m*n);
	for(int i=0; i<m*n;i++){
		arr[i]=i;
	}
	float **brr = alloc2d(m,n, arr);
	printf("assigninng 0, m*n to arr\n");	
	prnt(brr, m,n); 
	printf("-----------------alloc2dWorks------------------\n");
}


void  test_swap(){
	
	float  *a, *b ;
	float inpa[] = {0.9, 1.0, 1.2, 1.3, 1.4, 1.5, 0.9, 1.0, 1.2, 1.3, 1.4, 1.5};
	float inpb[] = {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4 };
	cudaMalloc(&a,sizeof(float)*12 );
	cudaMalloc(&b,sizeof(float)*12 );
	
	cudaMemcpy(a, inpa, sizeof(float)*12, cudaMemcpyHostToDevice);
	cudaMemcpy(b, inpb, sizeof(float)*12, cudaMemcpyHostToDevice);
	
	swap<<<3,3>>>(a,b,1,3,7);
	
	cudaMemcpy(inpa, a ,sizeof(float)*12, cudaMemcpyDeviceToHost);
	cudaMemcpy(inpb, b, sizeof(float)*12, cudaMemcpyDeviceToHost);
	
	prnt(inpa, 12 );
	prnt(inpb, 12 );
	printf("-----------------swap end------------------\n");
	
}



void test_objectiveFn(int pop, int col, int grd, int blk, curandGenerator_t gen){
	printf("-----------------testing objective function------------------\n");

	//float *res, float *harmonics, int dim, int pitch, int population
	/**
		float *pop, size_t pitch, int low, int high, int population
	**/


	float *res, *harmonics;
	size_t pitch; 
	
	// generate random harmioncs in range and display
	cudaMallocPitch(&harmonics, &pitch, sizeof(float)*col, sizeof(float)*pop);
	curandGenerateUniform(gen, harmonics, pitch*pop);
	inrange<<<1,pop>>>(harmonics, pitch, -100.0, 100.0, pop, col);
	
	float** hostHarm = alloc2d(pop, col);	
	cudaMemcpy2D(&hostHarm[0][0], col*sizeof(float), harmonics, 
				pitch*sizeof(float), col*sizeof(float), pop, cudaMemcpyDeviceToHost);
	
	// get objective values
	cudaMalloc(&res, sizeof(float)*pop);
	objectiveFn<<<1,pop>>>(res, harmonics, col, pitch, pop);
	
	float *disp =(float*) malloc(sizeof(float)*pop);
	cudaMemcpy(disp, res , sizeof(float)*pop, cudaMemcpyDeviceToHost);

	cout<<"------out"<<endl;
	//free(disp);
	prnt(hostHarm, pop, col);
	
	prnt(disp, pop);
	printf("-----------------objective function ends------------------\n");
}


void test_sorted(){

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	size_t pitch; 
	
	int pop = 10;
	int dim = 5;
	int col = dim;
	float *obj ;
	
	//generate 
	float *harmonics = gen_random(gen, pop, dim, &pitch, -10, 10);
	float *resHarms = gen_random(gen, pop, dim, &pitch, 0.0,0.0);
	
	cudaMalloc(&obj, sizeof(float)*pop);
	objectiveFn<<<1,pop>>>(obj, harmonics, col, pitch, pop);
	
	//print harmonics
	
	float brr[pop][dim];
	cudaMemcpy2D( brr, dim*sizeof(float), harmonics, pitch*sizeof(float), dim*sizeof(float),pop, cudaMemcpyDeviceToHost);
	for(int i=0; i< pop; i++){
		for(int j=0 ; j< dim; j++){
			cout<<" "<<brr[i][j];
		}
		cout<<endl; 
	}
		
	sorted<<<1, pop>>>( resHarms , harmonics, obj, pop, dim, pitch);

	//display objective function
	float *disp =(float*) malloc(sizeof(float)*pop);
	cudaMemcpy(disp, obj, sizeof(float)*pop, cudaMemcpyDeviceToHost);
	prnt(disp, pop);
					
	//float brr[pop][dim];
	cudaMemcpy2D( brr, dim*sizeof(float), resHarms, pitch*sizeof(float), dim*sizeof(float),pop, cudaMemcpyDeviceToHost);
	for(int i=0; i< pop; i++){
		for(int j=0 ; j< dim; j++){
			cout<<" "<<brr[i][j];
		}
		cout<<endl; 
	}
	
	
	
}

void test_harmonic_update(){}

void test_update_harmonics(){}


int main(){

	//int pop = 7, col = 5;
	//test_alloc2d();
		
	//////////////////////GEN RANDOM ///////////////////////
	//test_gen_random(pop, col, -100.0, 100.0);
	
	

	///////////////////////OBJFN////////////////////////////
	//curandGenerator_t gen;
	//curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	//curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	//size_t pitch; 
	//test_objectiveFn(pop, col, 4, 3, gen);	
	
	//////////////////////SWAP//////////////////////////////
	//test_swap();
	test_sorted();
	
	return 0;
}
