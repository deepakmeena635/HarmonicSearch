#include<cuda.h>
#include<iostream>
#include<curand.h>
#include<malloc.h>
#include<stdlib.h>
#include"hsutils.h"

using namespace std;


__global__ void idek(float *resHarms, float* harmonics, 
					float* obj, int pop,int  dim,size_t  pitch){

	int id = threadIdx.x; 
	if(id > pop){
		return ;
	}
	obj[id]= id;
}

void test_gen_random(int pop, int dim, float low, float high){

	printf("--------Random gen and vector_scale and inrange tested together -------\n");
		
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,551234ULL);
	size_t pitch; 
	
	
	for (int i=0 ; i< 4 ; i++){
	
		float* arr = gen_random(gen, pop, dim, &pitch, low, high);
		float brr[pop][dim];
		cudaMemcpy2D(brr, dim*sizeof(float), arr, pitch*sizeof(float), dim*sizeof(float),pop, cudaMemcpyDeviceToHost);
		
		for(int i=0; i< 10; i++){
			for(int j=0 ; j< dim; j++){
				cout<<" "<<brr[i][j];
			}
			cout<<endl; 
		}
			
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
	objectiveFn<<<1,pop>>>(res, harmonics, col, pitch, pop, 67);
	
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
	
	int pop = 1001;
	int dim = 1001;
	int col = dim;
	float *obj, *sobj ;
	
	//generate 
	float *harmonics = gen_random(gen, pop, dim, &pitch, -10, 10);
	float *resHarms = gen_random(gen, pop, dim, &pitch, 0.0,0.0);
	
	cudaMalloc(&obj, sizeof(float)*pop);
	cudaMalloc(&sobj, sizeof(float)*pop);
	objectiveFn<<<1,pop>>>(obj, harmonics, col, pitch, pop, 69);
	
	//print harmonics
	float brr[pop][dim];
	cudaMemcpy2D( brr, dim*sizeof(float), harmonics, pitch*sizeof(float), dim*sizeof(float),pop, cudaMemcpyDeviceToHost);
	
	
	
	
	//idek<<<1,pop>>>(resHarms, harmonics, obj,sobj,  pop, dim, pitch);
	
	sorted<<<(int)pop/3+1,3>>>( resHarms, harmonics, obj,sobj,  pop, dim, pitch);
	cudaDeviceSynchronize();
	 
	//display objective values
	float *disp =(float*) malloc(sizeof(float)*pop);
	
	CUDA_CALL(cudaMemcpy(disp, obj, sizeof(float)*pop, cudaMemcpyDeviceToHost));
	//prnt(disp, pop);
	CUDA_CALL(cudaMemcpy(disp, sobj, sizeof(float)*pop, cudaMemcpyDeviceToHost));
	prnt(disp, pop);
	
	gpuErrchk(cudaMemcpy2D( brr, dim*sizeof(float), resHarms, pitch*sizeof(float), 
				dim*sizeof(float), pop, cudaMemcpyDeviceToHost));
	
		
}

void test_update_harmonics(){
	
	cout<<"-------------------testing update harmonics------------"<<endl;
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	size_t pitch; 
	
	int pop = 10;
	int dim = 3;
	int col = dim;
	
	float brange = 2.0;
	int nbgood = 3; 
	
	
	//generate 
	float *harmonics= gen_random(gen, pop, dim, &pitch, 2, 10);
	float *best = gen_random(gen, nbgood, dim, &pitch,0,2 );
	float *noise =gen_random(gen, pop, dim, &pitch, -1,1 ); 

	float** hostHarm = alloc2d(pop, col);
	cudaMemcpy2D(&hostHarm[0][0], col*sizeof(float), harmonics, 
			pitch*sizeof(float), col*sizeof(float), pop, cudaMemcpyDeviceToHost);
	
	cout<<"-----------harmonics intially-----------"<<endl;
	prnt(hostHarm, pop, dim);
	
	cout<<"-----------best hamrmonics-----------"<<endl;
	float** bestHrms= alloc2d(nbgood, col);
	cudaMemcpy2D(&bestHrms[0][0], col*sizeof(float), best, 
			pitch*sizeof(float), col*sizeof(float), nbgood, cudaMemcpyDeviceToHost);
	
	prnt(bestHrms, nbgood, dim);	
	
	int recs[] = {0,1,2,0,1,2,0,1,2,0};
	int *rndRecs;
	cudaMalloc(&rndRecs, sizeof(int)*pop );
	cudaMemcpy(rndRecs, recs, sizeof(int)*pop, cudaMemcpyHostToDevice);
	update_harmonics<<<1, pop >>>(harmonics, best, noise, rndRecs, brange, pop, dim, pitch);
	
	//int col = dim ;
		
	cudaMemcpy2D(&hostHarm[0][0], col*sizeof(float), harmonics, 
				pitch*sizeof(float), col*sizeof(float), pop, cudaMemcpyDeviceToHost);
	cout<<"-----------after update-----------"<<endl;
	prnt(hostHarm, pop, dim);
	
	cout<<"-------------------update harmonics testing done------------"<<endl;
	
}

void test_random_indexes(curandGenerator_t gen  ){
	//curandGenerator_t gen, int row, int maxIndex
	int rows = 100, maxIndex=5;
	
	int *indexes = gen_random_indexes(gen,rows,maxIndex);
	
	int* res = (int*) malloc(sizeof(int)*rows);
		
	cudaMemcpy(res, indexes, sizeof(int )*rows, cudaMemcpyDeviceToHost);
	for(int i=0; i<rows;i++){
		cout<<" "<<res[i];
	}
		
}

void test_accept_better( curandGenerator_t gen){
	///		accept_better(float*obj, float*sobj, float*bests, 
	// 		float* newBests, int nbgood, int dim, size_t pitch
	
	int pop= 6, col=3;
	int dim =col, nbgood = 4; 
	float *obj, *sobj;
	size_t pitch; 
	int population =pop;

	cudaMalloc(&obj, sizeof(float)*pop); 
	cudaMalloc(&sobj, sizeof(float)*pop);

	float * sample1, *sample2;

	sample1 = gen_random(gen, pop, dim, &pitch, -20, 20);
	sample2 = gen_random(gen, pop, dim, &pitch, -10, 10);

	objectiveFn<<<1,pop>>>(obj, sample1, col, pitch, pop, 67);
	objectiveFn<<<1,pop>>>(sobj, sample2, col, pitch, pop, 67);

	float *res1, *res2;
	res1 = (float*) malloc(sizeof(float)*pop);
	res2 = (float*) malloc(sizeof(float)*pop);

	cudaMemcpy(res1, obj, sizeof(float)*pop, cudaMemcpyDeviceToHost );
	cudaMemcpy(res2, sobj, sizeof(float)*pop, cudaMemcpyDeviceToHost);

	prnt(res1, pop);
	prnt(res2, pop);

	float * nsample1, *nsample2 ; 

	nsample1 = gen_random(gen, pop, dim, &pitch, -20, 15);
	nsample2 = gen_random(gen, pop, dim, &pitch, -10, 10);
	
	float *nobj, *nsobj; 
	
	cudaMalloc(&nobj, sizeof(float)*pop); 
	cudaMalloc(&nsobj, sizeof(float)*pop);
	cout<<"---------------------------\n";
	
	sorted<<<1,pop>>>(	nsample1, sample1, obj, nobj, pop, dim, pitch);
	sorted<<<1,pop>>>(	nsample2, sample2, sobj, nsobj, pop, dim, pitch);
	
	cudaMemcpy(res1, nobj, sizeof(float)*pop, cudaMemcpyDeviceToHost );
	cudaMemcpy(res2, nsobj, sizeof(float)*pop, cudaMemcpyDeviceToHost);

	prnt(res1, pop);
	prnt(res2, pop);
	
	
	///////////////////////////////////////////////////////////////////////////////////
	
	float  **idek = alloc2d(population, dim);
	
    cudaMemcpy2D( &idek[0][0], dim*sizeof(float), nsample1, 
    					pitch*sizeof(float), dim*sizeof(float), 
    					population, cudaMemcpyDeviceToHost);
    cout<<"---------------------------\n";
    prnt(idek, population, dim);
	idek = alloc2d(pop, dim);
	
    cudaMemcpy2D( &idek[0][0], dim*sizeof(float), nsample2, 
    					pitch*sizeof(float), dim*sizeof(float), 
    					population, cudaMemcpyDeviceToHost);
    cout<<"---------------------------\n";
    prnt(idek, population, dim);
    cout<<"------------pitch---"<<pitch<<endl;
	float *ress =  accept_better(nobj, nsobj, nsample1, nsample2, nbgood, dim, pitch, gen); //<---------------
	
	
	free(idek);
	idek = alloc2d(pop, dim);

    cudaMemcpy2D( &idek[0][0], dim*sizeof(float), ress, 
  					pitch*sizeof(float), dim*sizeof(float), 
					nbgood, cudaMemcpyDeviceToHost);
	cout<<"---------------------------\n";
    prnt(idek, nbgood, dim);

	///////////////////////////////////////////////////////////////////////
	cudaMemcpy(res1, nobj, sizeof(float)*pop, cudaMemcpyDeviceToHost);
	cudaMemcpy(res2, nsobj, sizeof(float)*pop, cudaMemcpyDeviceToHost);

	cout<<"---------------------------\n";
	prnt(res1, pop);
	prnt(res2, pop);	
}



int main(){

	int pop = 2048, col = 5;
	//test_alloc2d();
		
	//////////////////////GEN RANDOM ///////////////////////
	test_gen_random(pop, col, -1000.0, 1000.0);
	
	///////////////////////OBJFN////////////////////////////
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	//size_t pitch; 
	//test_objectiveFn(pop, col, 4, 3, gen);	
	
	//////////////////////SWAP//////////////////////////////
	//test_swap();
	//test_sorted();
	//test_update_harmonics();
	//test_random_indexes(gen);
	//test_accept_better(gen);
	return 0;
}
