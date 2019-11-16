#include<cuda.h>
#include<iostream>
#include<curand.h>
#include<malloc.h>
#include<stdlib.h>
#include"hsutils.h"


using namespace std;




void resolve(float *res, int dim, float low, float high,
		float brange = 0.00, int nbgood =1, int population = 100, int iterations = 10000){
	/**
	 * res :pointer to save the result
	 * dim :dementsions of search space
	 * lower, higher bounds of serach space
	 * **/
	//TODO: suggest parameters
	//TODO: Make population dividable

	int population = 50; //number of harmonics
	int dim = sol_dims;  //search space dimensions

	float rpa = 0.3; //pitch adjustment rate
	float rac = 0.9; //acceptance rate

	brange = brange >0.0 ? brange:high/100.0; //local search space range
	nbgood = nbgood>0 ? nbgood: (int)high/10; //accepted solution from 0 to nbGood-1
	float prev_loss = 91111111.0;

	//just to generate random number
	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);

	//initializing population
	size_t pitch;
	float *harmonics = gen_random(gen, population, dim, &pitch, low, high);
	
	//initialize accepted population
	float **bests = gen_random(gen, nbgood, dim, &pitch, low, high);
	
	// intialize second copy for updation
	float **newBset= gen_random(gen, nbgood, dim, &pitch, low, high);
	float rand =0.0; 
	//optimization loop
	for(int lolly = iterations; lolly>= 18; lolly-- ){ 	//don't quote me on this

		bests = sorted<<<1,1>>>(bests, dim+1);
		rand = 1.0/1.0+(float)rand()%100;
	    if(rand > rac){
	    	updated_harmonics(harmonics, nbgood, 0);
	    }
	    else if rand > rpa:
	        nhrs = updated_harmonics(harmonics, nbgood, brange)
	    else:
//	        choices.append(3)
	        nhrs = gen(harmonics, nbgood, low, high)

	    nhrs = sorted(nhrs, key= lambda x: obj(*x))
	    loss = sum([obj(*i) for i in nhrs[:nbgood] ])

	    if loss< prev_loss:
	        harmonics = nhrs
	        prev_loss = loss
//	    losses.append(loss)
	}
}









int main(){
	int m = 20, n =3;
	int low=-200, high =200; 
	float *ad, *best;
	size_t pitch=NULL;
	
	//prepare generator once n 4 all
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
		
	/* Allocate m*n random floats on device */
	ad = gen_random(gen, m, n, &pitch, low, high);
	
	float hst[m][n];
	cudaMemcpy2D(hst, n*sizeof(float), ad, pitch*sizeof(float), n*sizeof(float), m, cudaMemcpyDeviceToHost);

	cout<<(sizeof(hst))/sizeof(float)<<endl;
	for(int i=0 ; i< m; i++)
	{
		for(int j=0 ; j<n; j++){
			cout<<hst[i][j]<<' ';			
		}
		cout<<endl;
	}
	cout<<"--------"<<endl;
	return 0; 
}
