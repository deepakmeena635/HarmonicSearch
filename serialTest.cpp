#include<iostream>
#include<malloc.h>
#include<bits/stdc++.h>
#include"sutils.h"
using namespace std ;

///////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

void test_sort(){

	float** arr, ** brr ;
	arr = alloc2d(5, 3);
	brr = alloc2d(5, 3);
	for(int i =0; i<5; i++){
		for(int j=0; j< 3;j++){
			arr[i][j] =100-i*20;
		cout<<arr[i][j]<<" ";
		}
	cout<<endl;
	}

	cout<<endl;
	cout<<endl;
	float ** crr;
	crr = sorted(arr, 5,3);

	for(int i =0; i<5; i++){
		for(int j=0; j< 3;j++){
			cout<<crr[i][j]<<" ";
		}
	cout<<endl ;
	}

	cout<<"<------------------------SORT ENDED\n";

}
void test_accept(){

	float **arr, ** brr , **crr;
	arr = alloc2d(5, 3);
	brr = alloc2d(5, 3);
	for(int i =0; i<5; i++){
		for(int j=0; j< 3;j++){
			arr[i][j] =10-i*2;
		cout<<arr[i][j]<<" ";
		}
	cout<<endl;
	}
	cout<<endl;
	cout<<endl;

	for(int i =0; i<5; i++){
		for(int j=0; j< 3;j++){
			brr[i][j] =10-i*2;
			brr[i][j] += (-5+rand()%6);
		cout<<brr[i][j]<<" ";
		}
	cout<<endl;
	}
	cout<<endl;
	cout<<endl;
	cout<<endl;

	crr =accept_better(arr, brr, 5,3 );
	prnt(crr, 5, 3);

	cout<<"<------------------------ACCEPT ENDED\n";

}
void test_harmonics(std::mt19937_64 gen ,uniform_real_distribution<float> dist ){


	float **arr, **brr ;
	arr = alloc2d(5, 3);

	for(int i =0; i<5; i++){
		for(int j=0; j< 3;j++){
			arr[i][j] =10-i*2;
		cout<<arr[i][j]<<" ";
		}
	cout<<endl;
	}
	cout<<endl;

	brr = update_harmonics(arr, 12.0, 10,3, 5, gen, dist);
	prnt(brr, 10, 3);

}

void test_gen(std::mt19937_64 gen ,
				uniform_real_distribution<float> dist ){

	float **noise = gen_random(gen, dist ,10 , 3);
	prnt(noise, 10,3);

}
int main(){
	float low= -p, high =10;

	std::random_device rd;
	std::mt19937_64 gen(rd());
//	std::uniform_int_distribution<int> dist_int(0, population - 1);
//	std::uniform_real_distribution<float> dist_01(0.f, 1.f);
	std::uniform_real_distribution<float> dist_range(low, high);


//
//	test_sort();
//	test_accept();
//	test_gen(gen, dist_range);
	test_harmonics(gen, dist_range);


	return 0;
}
