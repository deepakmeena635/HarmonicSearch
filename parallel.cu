#include<cuda.h>
#include<iostream>
#include<curand.h>
#include<malloc.h>
#include<stdlib.h>
#include"hsutils.h"
#include<chrono>


using namespace std;

float *resolve( int dim, float low, float high,
		float brange = 0.00, int nbgood =0, int population = 100, int iterations = 10000, bool debugMode=false){
	/**
	 * res :pointer to save the result
	 * dim :dementsions of search space
	 * lower, higher bounds of serach space
	 * **/
	//TODO: suggest parameters
	//TODO: Make population dividable

	auto start = std::chrono::high_resolution_clock::now();

	float offset= 50.00;

	float * res ;
	float rpa = 0.3; //pitch adjustment rate
	float rac = 0.8; //acceptance rate

	brange = brange >0.0 ? brange: high/100.0; //local search space range
	nbgood = nbgood>0 ? nbgood: (int)high/100; //accepted solution from 0 to nbGood-1
	
	//just to generate random number
	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);

	//initializing population
	size_t pitch;
	float *harmonics = gen_random(gen, population, dim, &pitch, low, high);
	float *newHarmonics = gen_random(gen, population, dim, &pitch, low, high);
	
	//initialize accepted population
	float *bests = gen_random(gen, nbgood, dim, &pitch, low, high);
	
	// intialize second copy for updation
	float *newBest= gen_random(gen, nbgood, dim, &pitch, low,high);
	float randNo=0.0; 
	
	//configureations
	int pgrid = population>1024? 1:int(population/512)+1;
	int pblock = population>1024? population: 512 ;	
	int ggrid = nbgood>1024? 1:int(nbgood/512)+1;
	int gblock = nbgood>1024? nbgood:512;
	
	float  *obj, *sobj ;
	cudaMalloc(&obj, sizeof(float)*population);
	cudaMalloc(&sobj, sizeof(float)*population);
	
	float *noise = gen_random(gen, population, dim, &pitch, -1,1 );
	int *rndRecs = gen_random_indexes(gen, population, nbgood);
	
	float loss=0.0;
	float *recObj= (float*)malloc(sizeof(float)*nbgood);
	float **idek = alloc2d(nbgood, dim);
	float *merged ;
	float prevLoss = 3990000.0;
	int choise =0; 
	//optimization loop
	for(int lolly = iterations; lolly>= 18; lolly-- ){ 	//don't quote me on this
		
		cudaFree(obj);
		cudaFree(sobj);
		cudaFree(newBest);
		
		cudaMalloc(&obj, sizeof(float)*nbgood);
		cudaMalloc(&sobj, sizeof(float)*nbgood);
		newBest= gen_random(gen, nbgood, dim, &pitch, low,high);
		
		objectiveFn<<<pgrid,pblock>>>(obj, bests, dim, pitch, nbgood, offset);
		
//		//////////////////////////////
		if(debugMode ){
					cout<<"before Rand"<<endl;
					cudaMemcpy(recObj, obj, sizeof(float)*nbgood, cudaMemcpyHostToHost);
					prnt(recObj, nbgood);
					
					cudaMemcpy2D( &idek[0][0], dim*sizeof(float), bests, pitch*sizeof(float), 
													dim*sizeof(float), nbgood, cudaMemcpyHostToHost);
					prnt(idek, nbgood, dim);
					
			
		}
		
//		//////////////////// 
		sorted<<<ggrid, gblock>>>(newBest, bests, obj, sobj, nbgood, dim, pitch);
		
		cudaFree(bests);
		bests = newBest;
		newBest = NULL;
		////////////////////////////////////////
		if (debugMode && false){

					cout<<"sorted below"<<endl;
					cudaMemcpy(recObj, sobj, sizeof(float)*nbgood, cudaMemcpyHostToHost);
					prnt(recObj, nbgood);

					cudaMemcpy2D( &idek[0][0], dim*sizeof(float), bests, pitch*sizeof(float), 
															dim*sizeof(float), nbgood, cudaMemcpyHostToHost);
					prnt(idek, nbgood, dim);
					cout<<endl;

		}
		/////////////////////////////
		cudaDeviceSynchronize();

		randNo=rand()%1000/1000.0;
		
		
	    if(randNo> rac){
	    	choise =1;
	    	cudaFree(noise);
	    	cudaFree(rndRecs);
	    	noise =gen_random(gen, population, dim, &pitch, -1, 1 ); 
	    	rndRecs = gen_random_indexes(gen, population, nbgood);
//	    	//////////////////////////////////////////
	    	if(debugMode && false){
		    	cout<<endl;
		    	free(recObj);
		    	recObj= (float*)malloc(sizeof(float)*population);
		    	cudaMemcpy(recObj, rndRecs, sizeof(float)*population, cudaMemcpyHostToHost);
		    	prnt(recObj, nbgood);
	    	}
//	    	//////////////////////////////////////////
	    	
	    	update_harmonics<<<pgrid, pblock>>>(harmonics, bests, noise, 
	    										 rndRecs,0.0, population,dim, pitch);
	    }
	    else if (randNo> rpa){
	    	choise =2;
	    	cudaFree(noise);
	    	cudaFree(rndRecs);
	    	noise =gen_random(gen, population, dim, &pitch, -1, 1 ); 
	    	rndRecs = gen_random_indexes(gen, population, nbgood);
//	    	///////////////////////////////////////////
	    	if(debugMode && false){
				cout<<endl;
				free(recObj);
				recObj= (float*)malloc(sizeof(float)*population);
				cudaMemcpy(recObj, rndRecs, sizeof(float)*population, cudaMemcpyHostToHost);
				prnt(recObj, nbgood);
	    	}
//			//////////////////////////////////////////
	    	cudaDeviceSynchronize();
	    	
	    	update_harmonics<<<pgrid, pblock>>>(harmonics, bests, noise, 
	    										 rndRecs,brange, population,dim, pitch);
	    }
	    else{
	    	choise =3;
	    	cudaFree(harmonics);
	    	harmonics = gen_random(gen, population, dim, &pitch, low, high);
	    }
	    cudaDeviceSynchronize();

	    
	    //////////////////////////////////////////
	    if(debugMode && false){
			cout<<endl;
			free(idek);
			idek = alloc2d(population, dim);
			
		    cudaMemcpy2D( &idek[0][0], dim*sizeof(float), harmonics, 
		    					pitch*sizeof(float), dim*sizeof(float), 
		    					population, cudaMemcpyDeviceToHost);
		    prnt(idek, nbgood, dim );// only first nbgood 
		    cout<<endl;

	    }
	    //////////////////////////////////////////
	    cudaDeviceSynchronize();


	    cudaFree(newBest);
	    if (newHarmonics!=NULL){ 
	    	
	    	cudaFree(newHarmonics);
	    }
		cudaFree(obj);
		cudaFree(sobj);
				
		cudaMalloc(&obj, sizeof(float)*population);
		cudaMalloc(&sobj, sizeof(float)*population);
		newHarmonics= gen_random(gen, population, dim, &pitch, 0.0,0.0);
		newBest= gen_random(gen, nbgood, dim, &pitch, 0.0,0.0);
		
		objectiveFn<<<pgrid,pblock>>>(obj, harmonics,dim, pitch, population, offset);
	    sorted<<<pgrid, pblock>>>(newHarmonics, harmonics, obj, sobj, population, dim, pitch);
	    cudaMemcpy2D( newBest, dim*sizeof(float), newHarmonics, 
	    					pitch*sizeof(float), dim*sizeof(float), 
	    					nbgood, cudaMemcpyHostToHost);
	    cudaFree(newHarmonics);
	    newHarmonics = NULL;
	    
	    cudaFree(obj);
	    cudaMalloc(&obj, sizeof(float)*nbgood);
	    objectiveFn<<<pgrid,pblock>>>(obj, bests,dim, pitch, nbgood, offset);
	    merged = accept_better(obj, sobj, bests, newBest,nbgood, dim, pitch,  gen);
//	    cudaFree(bests);
	    bests = merged;
	    
	    cudaMemcpy(recObj, obj, sizeof(float)*nbgood, cudaMemcpyDeviceToHost);
	    loss = avg_loss(recObj, nbgood);
	    if (true ){
	    	printf("%d choise:%d   AVERAGE LOSS: %f\n", lolly, choise, loss);
	    	prevLoss = loss; 
	    }
	   
	    if(debugMode && false){
	    	cudaMemcpy2D( &idek[0][0], dim*sizeof(float), bests, pitch*sizeof(float), 
								dim*sizeof(float), nbgood, cudaMemcpyDeviceToHost);
	    	prnt(recObj, nbgood);
	    	prnt(idek, nbgood, dim);
	    }
	    
	    
		//if(debugMode){			
	//		system(" sleep 3");
			
		//}
	
		if (loss< nbgood*0.1 ){
				break;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	double time_elapsed = double(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
	std::cout << std::endl << "Elapsed Time(s): " << time_elapsed /1000<< std::endl;
	
	res = (float*) malloc(sizeof(float)*dim);
	cudaMemcpy(res, bests, sizeof(float)*dim,cudaMemcpyDeviceToHost);
	
	return res;
}


int main(){
	
	int iterations = 10000;
	int dim = 128;
	int low=-2000, high =2000; 	
	int nbg = 3;
	int pop = 512; //population
	float  *result;
	int dm=0;
	
	nbg = (int) pop/10 ;
	cout<<" population: "<<pop<<" nb_g: "<<nbg<<endl;

	result = resolve(dim, low, high, 0.2, nbg, pop, iterations, dm);
	for( int i=0; i< dim; i++){
		cout<<result[i]<<" ";
	}	
	
	
	return 0; 
}
