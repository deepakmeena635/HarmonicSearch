#include<iostream>
#include<malloc.h>
#include<stdlib.h>
#include<bits/stdc++.h>
#include<random>
#include<chrono>
#include"sutils.h"

using namespace std;

float *resolve( int dim, float low, float high,
		float brange , int nbgood =0,
		int population = 100, int iterations = 10000){

	auto start = std::chrono::high_resolution_clock::now();

	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<int> dist_int(0, population - 1);
	std::uniform_real_distribution<float> noise_src(-1.0*brange, 1.0*brange);
	std::uniform_real_distribution<float> dist_range(low, high);

	int choise =0;
	float * res ;
	float rpa = 0.3; // adjustment rate
	float rac = 0.8; //acceptance rate

//	brange = brange >0.0 ?  high/100.0: brange; //local search space range
//	nbgood = nbgood>0 ? (int)high/100: nbgood; //accepted solution from 0 to nbGood-1

	float **harmonics = gen_random(gen,dist_range, population, dim);
	float **newHarmonics =gen_random(gen,dist_range, population, dim);
	float **bests = gen_random(gen, dist_range,nbgood, dim);
	float **newBest=  gen_random(gen,dist_range, nbgood, dim);
	float randNo=0.0;

	//configureations
	float loss=0.0;
	float *recObj= (float*)malloc(sizeof(float)*nbgood);
	float **merged ;
	float prevLoss = 3990000.0;

	//optimization loop
	for(int lolly = iterations; lolly>= 18; lolly-- ){ 	//don't quote me on this
		free(newBest);
		newBest = sorted(bests,nbgood, dim);
		free(bests);
		bests = newBest;
		newBest = NULL;

		randNo=rand()%1000/1000.0;

	    if(randNo> rac){
	    	choise =1;
	    	harmonics = update_harmonics( bests,0, population,dim, nbgood, gen, noise_src );
	    }
	    else if (randNo> rpa){
	    	choise =2 ;
	    	harmonics= update_harmonics( bests,brange, population,dim, nbgood,gen, noise_src);
	    }
	    else{
	    	choise =3 ;
	    	free(harmonics);
	    	harmonics = gen_random(gen, dist_range, population, dim);


	    }
	    free(newBest);
	    if (newHarmonics!=NULL){
	    	free(newHarmonics);
	    }
	    newHarmonics  = sorted(harmonics, population, dim);
	    merged = accept_better(bests, newHarmonics, nbgood, dim);
	    bests = merged;

	    recObj = objective(bests, nbgood, dim);
	    loss = avg_loss(recObj, nbgood);
	    if (true or loss<prevLoss){
	    	printf("choise:%d\tAVERAGE LOSS: %f\n", choise,loss);
	    	prevLoss = loss;
	    }
//	    if(lolly%40 ==0){
//	    	system("clear");
//	    }

//		cout<<endl;
//		for(int i=0; i< dim; i++){
//			cout<<newHarmonics[0][i]<<" ";
//		}
//		int ll;
//		cin>>ll;

	}

	auto end = std::chrono::high_resolution_clock::now();
	double time_elapsed = double(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

	std::cout << std::endl << "Elapsed Time(s): " << time_elapsed/1000.0 << std::endl;



	return bests[0];
}


int main(){


	int iterations = 10000;
	int dim = 20;
	int low=-2000, high =2000;
	int nbg = 8;

	float  *result;
	int dm=0;

	for(int pop = 64,i=1; i<2; i++, pop*= 2, nbg = (int) pop/10 ){
		nbg = (int) pop/10;
		result= resolve(dim, low, high, 0.2, nbg, pop, iterations);
		cout<<"iteration:"<<i<<" population: "<<pop<<" nb_g: "<<nbg<<endl;

		cout<<endl;
		cout<<endl;
		for(int i=0; i< dim; i++){
			cout<<result[i]<<" ";
		}
	}

	return 0;

}
