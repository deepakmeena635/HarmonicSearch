#define OFFSET 50.0

float avg_loss(float*obj, int nbgood){
	double sum =0.0;
	for(int i=0; i< nbgood;i++ )
		sum+=obj[i];
	return sum/nbgood;

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
void swap_serial(float * arr , float * brr, int andx, int bendx, int dim){
	for(int i=0 ; i< dim ; i++){
		float temp =arr[i+andx];
		arr[i+andx] = brr[i+bendx];
		brr[i+bendx]= temp;
	}
}
float **gen_random( std::mt19937_64 gen,
					std::uniform_real_distribution<float> noise_src,
					int m, int n){
	float ** arr = alloc2d(m,n);
	float *oned = &arr[0][0];
	for(int i=0; i< m*n;i++){
		oned[i] = noise_src(gen);
	}
	return arr;
}
float* objective(float **arr, int pop, int dim){

	float* res = (float *)malloc(sizeof(float)*pop);
	for(int i=0;i< pop; i++){
		res[i]=0;
		for(int j=0; j< dim; j++){
			res[i]+= (arr[i][j]-OFFSET)*(arr[i][j]-OFFSET);
		}
	}
	return res;
}
float **sorted(float **harmonics,  int population , int dim){

	//find resultant positions
	float** res = alloc2d(population, dim);
	float *obj = objective(harmonics, population, dim);
	float *sortedObj = (float* ) malloc(sizeof(float  )* population);

	for (int j=0 ; j< population; j++){

		float myobjective =  obj[j];
		int countSmaller=0;

		for(int i=0; i< population; i++){
			if(myobjective>= obj[i] and i!=j){
				if(i< j && myobjective== obj[i]){
					continue;
				}
				countSmaller++;
			}
		}
		//for duplicate entries
		int srcIndex = j*dim;
		int destIndex = countSmaller*dim;

		sortedObj[countSmaller] = myobjective;
		swap_serial(&res[0][0], &harmonics[0][0],  destIndex,srcIndex, dim);
	}
	return res;
}
float ** accept_better( float** bests, float** newBests, int nbgood, int dim){

	float *obj  = objective(bests, nbgood, dim);
	float *sobj = objective(newBests, nbgood, dim);

	float *nobj, **merged;
	int nc=0;
	nobj =(float*) malloc( sizeof(float)*nbgood);
	merged = alloc2d(nbgood, dim);
	int oc=0, soc=0;

	while(nc<nbgood){

		if( obj[oc]>sobj[soc] ){
			nobj[nc] = sobj[soc];
			memcpy(merged[nc], newBests[soc], sizeof(float)*dim);
			++soc;
		}
		else {
			nobj[nc] = obj[oc];
			memcpy(merged[nc], bests[oc], sizeof(float)*dim);
			++oc;
		}
		++nc;
	}
	free(obj);
	free(sobj);
	free(bests);

	return merged;
}
float **update_harmonics(  float **bests, float brange,
							int population, int dim,
							int nbgood, std::mt19937_64 gen,
							std::uniform_real_distribution<float> noise_src ){

	float **harmonics = alloc2d(population , dim);
	float ** noise = gen_random(gen, noise_src, population, dim);

	int *recIndex = (int* )malloc(sizeof(int)*population);
	for(int i=0; i<population; i++){
		recIndex[i] = (int ) rand()%nbgood;
	}
	for(int i=0; i<population; i++){
		for (int j=0; j< dim; j++){
			int index = recIndex[i];
			if (brange>0.00){
				harmonics[i][j] = bests[index][j]+noise[i][j];
			}
			else{
				harmonics[i][j] = bests[index][j];
			}
		}
	}
	free(noise);
	return harmonics ;
}
//////////////////////////////////////////////////////////////////////////////////////////
