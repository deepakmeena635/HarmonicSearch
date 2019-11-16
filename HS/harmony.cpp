#include<cuda>
#include<iostream>
#include<curand.h>
#include<math.h>
#include<malloc.h>
#include<stdlib.h>

using namespace std;

__global__ void update_harmonics(float **pot, float **sols, int nb_pot, int nbgood, float brange, curandGenerator_t gen){

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;

	// genrate random index b/w 0,nbGood
	// hars,nbg,  rp
	float rand;
	curandGenerateUniform(gen, &rand, 1);
	rand-=0.5;
	rand *=2;
	pot[idx][idy] += brange*rand;
}

__global__ float **alloc2dint( int m, int n, bool oncuda=false, int *PrevArr=NULL){

	auto allocator = oncuda == true ? cudamalloc : malloc;
	float * arr= (float ** )allocator(sizeof(float)*(m*n));
	if (arr != NULL){
		for (int i=0 ; i<m*n; i++){
			arr[i]= PrevArr[i];
		}
	}

	float **brr = (float **) allocator(sizeof(*float)*m);
	for (int i=0 ; i< m , i++){
		brr[i] = &arr[i*n];
	}
	return brr ;
}

__global__ void sorted(double ** x, int len, int elements ){

	float t1, t2, t3;
	for(int j =0; j<len; j++  ){
		for(int i =1; i<len-j; i++  ){
			if(x[i-1][elements-1]> x[i][elements -1 ]){
				swap<<<1,elements>>>(&x[i-1][0], &x[i][0]);
			}
		}
	}
}



__global__ void resolve(float *res, int dim, float low, float high,
		float brange = 0.00, int nbgood =0, int population = 100, int iterations = 10000){
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
	float **harmonics ;
	cudaMallocPitch((void**) &harmonics, &pitch, population*sizeof(float), dim+1);
	gen_random<<< 1,population  >>>(harmonics, dim+1, low, high, gen);

	//initialize accepted population
	float **bests = NULL;
	cudaMallocPitch((void**) &bests, &pitch, nbgood*sizeof(float), dim+1);
	gen_random<<<1, nbgood>>>(bests, dim+1, low, high, gen);

	// intialize second copy for updation
	float **newBet= NULL;
	cudaMallocPitch((void**) &newBet, &pitch, nbgood*sizeof(float), dim+1);
	gen_random<<<1, nbgood>>>(newBet, dim+1, low, high, gen);

	float rand = NULL;
	//optimization loop
	for(int lolly = iterations; lolly>= 18; lolly-- ){ 	//don't quote me on this

		sorted<<<1,1>>>(bests, nbgood, dim+1);
	    rand = curandGenerateUniform(gen, float(*) &rand, 1);
	    if(rand > rac){
	    	updated_harmonics(harmonics, nbgood, 0)
	    }
	    else if rand > rpa:
//	        choices.append(2)
//			print("2", end = ' ')
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

	int dims = 20, iterations = 2000;
	float* res = malloc(sizeof(float)*dims);


}
