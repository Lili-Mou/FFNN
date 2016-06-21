//============================================================================
// Name        : FFNN.cpp
// Author      : 
// Version     :
// Copyright   : None
// Description : Hello World in C++, Ansi-style
//============================================================================
//#include"global.h"
#include"FFNN.h"
#include"activation.h"
#include<math.h>
#include <malloc.h>
#include"global.h"
#include<stdlib.h>
int * file_train;
int * file_CV;
int * file_test;

Layer *** X_train;
Layer *** X_CV;
Layer *** X_test;

int * len_X_train;
int * len_X_CV;
int * len_X_test;

int * y_train;
int * y_CV;
int * y_test;
bool isTraining;


//////////////////////////////////////////////////////////
// forward propagation and back propagation

void FeedForward(Layer ** layer, int n){
	//cout<<"net"<<n<<endl;
	for (int i = 0; i < n; i++){
	//	cout<<"layer "<<i<<"  "<<layer[i]->name<<" "<<layer[i]->numUnit<<endl;
		layer[i]->computeY();
		/*if (i>=n-4|| (i>1 && layer[i-1]->name=="pool"))
		{
			cout<<"layer "<<i<<"  "<<n<<layer[i]->name<<layer[i]->numUnit<<endl;
		
			for (int j=0;j<layer[i]->numUnit;j++)
			cout<<layer[i]->y[j]<<" ";
			cout<<endl;
		}
		*/
	}
	
}
void BackPropagation( Layer ** layer, int n){
	for (int i = n - 1; i >= 0; i--){

		layer[i]->updateB();
	}
}
#define FLOAT_SIZE 4
void CleanDerivative( Layer ** layer, int n){
	for (int i = 0; i < n; i++){
		memset( layer[i]->dE_dy, 0 , FLOAT_SIZE * layer[i]->numUnit  );
	}
}

///////////////////////////////////////////////////////////
// layer functions
Layer::Layer(string _name, int _numUnit, int _bidx, int _numDown,
		void(* _f)(float *x, float *y, int n),
		void(* _fprime)(float *, float * dy_dz, int n) ){

	int size = sizeof(float) * _numUnit;

	// neuron properties
	name =_name;
	numUnit = _numUnit;
	numDown = _numDown;
	bidx = _bidx;
    p_drop = 0;

    f = _f;
	fprime = _fprime;


	// initialize states (allocate memory)
//	z = (float *) malloc( size);
//	y = (float *) malloc( size);
//	dE_dz = (float *) malloc( size );
//	dE_dy = (float *) malloc( size );
//	dy_dz = (float *) malloc( size );

	z = new float[ numUnit ];
	y = new float[ numUnit ];
	dE_dz = new float[ numUnit ];
	dE_dy = new float[ numUnit ];
	dy_dz = new float[ numUnit ];
	// connections
	//connectUp = new Connection *[_numUp];
	connectDown = new Connection * [_numDown];

	// activate

}

DropoutLayer::DropoutLayer(string name, int numUnit, int bidx, int numDown,
				void (* f)(float *z, float *y, int n),
				void (* fprime)(float *y, float * dy_dz, int n) )
	: Layer(name, numUnit, bidx, numDown, f, fprime){

	indicator = new int[numUnit];
	p_drop = 0.5;
};

DropoutLayer::~DropoutLayer(){
	delete indicator;
}
void DropoutLayer::computeY(){


	// Embeddings do not have input.
	// Thus, y = biases[bidx]
	int num = this->numUnit;


	// otherwise, y = f(W*x+b)

	// first, compute z = b + sum_i W_i * x_i
	if (bidx >= 0)
		icopy(num, biases + bidx, this->z);
	else
		memset(this->z, 0, sizeof(float) * num);
	for ( int i = 0; i < numDown; i++ ) {
		connectDown[i] -> computeZ();
		// z += connectDown[i].tmpZ
		//iXpY(num, connectDown[i]->tmpZ, this->z);
	}


	// second, apply the activation function
	if ( f!= NULL)
		(* (this->f)) (this->z, this->y, num);

	// training
	// dropout !!!
	if( isTraining ){
		for (int i = 0; i < num; i++){

			int tmp_ind = ( (rand() % 100) >= (this->p_drop * 100) ) ;
			this->indicator[i] = tmp_ind;
			this->y[i] *= tmp_ind;
		}
	}

	return;
}



void DropoutLayer::updateB(){


	int num = this->numUnit;


	if (fprime != NULL){

		// dy_dz = f', evaluated at y
		( * this->fprime)(this->y, this->dy_dz, num);
		// dE_dz = dE_dy .* dy_dz

		pointwise_dot(this->dE_dy, this->dy_dz, this->dE_dz, num);

		for( int i = 0; i < num; ++i){
			this->dE_dy[i] *= this->indicator[i];
		}
	}// else if fprime == NULL{
		// do nothing
	//}


	// dE_db = dE_dz * 1<size_data x 1> / size_data
	//       = dE_dz   if size of data = 1
	alphaXpY( num , 1.0/batch_size, this->dE_dz, gradBiases + bidx );


	for(int i = 0; i < this->numDown; i++){
		connectDown[i]->updateW();
	}


	return;
//		ReLUPrime(float * y, float * dy_dz, int n);
}




void Layer::computeY(){

	//cout << "in base" << endl;
	// Embeddings do not have input.
	// Thus, y = biases[bidx]
	int num = this->numUnit;

	// otherwise, y = f(W*x+b)

	// first, compute z = b + sum_i W_i * x_i
	if (bidx >= 0)
		icopy(num, biases + bidx, this->z);
	else
		memset(this->z, 0, sizeof(float) * num);
	for ( int i = 0; i < numDown; i++ ){
		//cout<<name<<"connect "<<i<<endl;
		connectDown[i] -> computeZ();
		// z += connectDown[i].tmpZ
		//iXpY(num, connectDown[i]->tmpZ, this->z);
	}
	
	// second, apply the activation function
	if( f!= NULL)
		(* (this->f))(this->z, this->y, num);

	return;
}



void Layer::updateB(){

	int num = this->numUnit;


	if (fprime != NULL){
		// dy_dz = f', evaluated at y
		( * this->fprime)(this->y, this->dy_dz, num);
		// dE_dz = dE_dy .* dy_dz
		pointwise_dot(this->dE_dy, this->dy_dz, this->dE_dz, num);
	}// else if fprime == NULL{
		// do nothing, because we assume dE_dz is given by softmax

	//}


	// dE_db = dE_dz * 1<size_data x 1> / size_data
	//       = dE_dz   if size of data = 1
	alphaXpY( num , 1.0/batch_size,  this->dE_dz, gradBiases + bidx );


	for(int i = 0; i < this->numDown; i++){
		connectDown[i]->updateW();
	}


	return;
//		ReLUPrime(float * y, float * dy_dz, int n);
}



PositivePoolLayer::PositivePoolLayer(string _name, int _numUnit, int _numDown)
	:Layer(_name, _numUnit, -1, _numDown, NULL, NULL){
	delete this->dE_dz;
	dE_dz = dE_dy;
	delete this->z;
	z = y;
	indicator = new int[numUnit];
	p_drop = 0;
}

PositivePoolLayer::~PositivePoolLayer(){
	delete indicator;
}

void PositivePoolLayer::computeY(){

	memset(this->z, 0, sizeof(float) * this->numUnit);
	
	for ( int i = 0; i < numDown; i++ ){
		connectDown[i] -> computeZ();
	}
	if (isTraining){
		if ( p_drop > 0){
			for ( int i = 0; i < numUnit; i++){
				int tmp_ind = ( (rand() % 100) >= (this->p_drop * 100));
				this->indicator[i] = tmp_ind;
				this->y[i] *= tmp_ind;
			}
		} else{
			for ( int i = 0; i < numUnit; i++){
				this->indicator[i] = 1;
			}
		}
	}
	return;
}
void PositivePoolLayer::updateB(){
	if ( p_drop > 0){
		for( int i = 0; i < numUnit; i++){
			dE_dy[i] *= indicator[i];
		}
	}
	for(int i = 0; i < this->numDown; i++){
		connectDown[i]->updateW();
	}
	return ;
}
///////////////////////////////////////////////////////
// connection functions
Connection::Connection(){}// do nothing, reserved for derived classed

Connection::Connection(Layer * _x, Layer * _y,
		int _xnum, int _ynum, int _Widx, float _Wcoef){

	xlayer = _x;
	ylayer = _y;
	xnum = _xnum;
	ynum = _ynum;
	Widx = _Widx;
	Wcoef = _Wcoef;

//	tmpZ = (float *) malloc( sizeof(float) * _ynum );

}

void Connection::computeZ(){
	//z += coef * W * x
	//cout<<this->ynum<<" "<< this->xnum<<" "<< this->Wcoef<<" "<<Widx<< endl;
	// not dropout
	if ( isTraining  ){
		if ( Widx >= 0 ){
			selfplus_matrix_dot_vector(this->ynum, this->xnum, this->Wcoef,
		          weights + this->Widx, CblasNoTrans,this->xlayer->y,
		          ylayer->z);
		} else { // Widx <= 0, where we have z += coef * x
			alphaXpY( xnum, Wcoef, xlayer->y, ylayer->z);
		}


	}else{ // testing, dropout
		if ( Widx >= 0 ){
			selfplus_matrix_dot_vector(this->ynum, this->xnum, this->Wcoef * (1-this->xlayer->p_drop),
			    weights + this->Widx, CblasNoTrans,this->xlayer->y,
			    ylayer->z);
		} else{ // Widx <= 0,
			alphaXpY( xnum, Wcoef * ( 1 - xlayer->p_drop ), xlayer->y, ylayer->z);
		}
		
	}
	return;
}
void Connection::updateW(){
	// dE_dW += coef * dE_dz * x.T
	//    note that x.T is exactly x, provided that x is a column vector
	if ( Widx >= 0){
		selfplus_matrix_dot_matrix(this->ynum, this->xnum, 1, this->Wcoef / batch_size,
					  this->ylayer->dE_dz, CblasNoTrans,
					  this->xlayer->y, CblasNoTrans,
					  gradWeights + this->Widx
		);


		//dE_dx += coef * W.T * dE_dz
		selfplus_matrix_dot_vector( this->ynum, this->xnum, this->Wcoef,
						weights + Widx, CblasTrans,
						this->ylayer->dE_dz,
						this->xlayer->dE_dy
		);
	} else{ // Widx < 0, where we have dE_dx += coef * dE_dz
		alphaXpY( xnum, Wcoef, ylayer->dE_dz, xlayer->dE_dy );
	}
//	inline void matrix_dot_vector(int M, int N, float coef, float * A, float * x, float * y){

}

BilinearConnection::BilinearConnection(Layer* _x1, Layer* _x2, Layer* _y, int _xnum, int _Widx, float _Wcoef){
	xlayer = _x1;
	xlayer2 = _x2;
	ylayer  = _y;
	Widx = _Widx;
	xnum = _xnum;
	ynum = _xnum;
	Wcoef = _Wcoef;
	pointwise_prod = new float[_xnum];
}
BilinearConnection::~BilinearConnection(){
	delete pointwise_prod;
}
void BilinearConnection::computeZ(){

	memset( this->pointwise_prod, 0 , FLOAT_SIZE * this->xnum  );

	for (int i = 0; i < xnum; i++){
		pointwise_prod[i] = xlayer->y[i] * xlayer2->y[i];
	}


	if ( isTraining  ){
		if ( Widx >= 0 ){
			selfplus_matrix_dot_vector(this->ynum, this->xnum, 1.0,
			          weights + this->Widx, CblasNoTrans, pointwise_prod,
			          ylayer->z);
		} else { // Widx < 0 
			alphaXpY( xnum, Wcoef, pointwise_prod, ylayer->z);
		}
	}
	else{ // testing, dropout
		if( Widx >= 0 ){
			selfplus_matrix_dot_vector(this->ynum, this->xnum, Wcoef * (1-this->xlayer->p_drop) * (1-this->xlayer2->p_drop),
				weights + this->Widx, CblasNoTrans, pointwise_prod,
				ylayer->z);
		} else {
			alphaXpY( xnum, Wcoef * ( 1-this->xlayer->p_drop) * (1-this->xlayer2->p_drop), pointwise_prod, ylayer->z );
		}
	}

}
void BilinearConnection::updateW(){

	if ( Widx >= 0){
		// dE_dW += coef * dE_dz * x.T

		
		selfplus_matrix_dot_matrix(this->ynum, this->xnum, 1, Wcoef/ batch_size,
					  this->ylayer->dE_dz, CblasNoTrans,
					  this->pointwise_prod, CblasNoTrans,
					  gradWeights + this->Widx
		);

		//  dE_dx += coef * W.T * dE_dz

		//float * tmp = new float[xnum];

		matrix_dot_vector( this->ynum, this->xnum, Wcoef,
						weights + Widx, CblasTrans,
						this->ylayer->dE_dz,
						pointwise_prod
		);
		// dE_dx1 = dE_dz * x2
		for (int i = 0; i < xnum; i++){
			xlayer ->dE_dy[i] += pointwise_prod[i] * xlayer2->y[i] ;
			xlayer2->dE_dy[i] += pointwise_prod[i] * xlayer ->y[i] ;
		}

	} else { // Widx < 0
		for (int i = 0; i < xnum; i++){
			xlayer ->dE_dy[i] += ylayer->dE_dz[i] * xlayer2->y[i] * Wcoef;
			xlayer2->dE_dy[i] += ylayer->dE_dz[i] * xlayer ->y[i] * Wcoef;
		}
			
	}

}

MinusConnection::MinusConnection(Layer* _x1, Layer* _x2, Layer* _y, int _xnum, int _Widx, float _Wcoef){
	xlayer = _x1;
	xlayer2 = _x2;
	ylayer  = _y;
	Widx = _Widx;
	xnum = _xnum;
	ynum = _xnum;
	Wcoef = _Wcoef;
	minus_prod = new float[_xnum];
}
MinusConnection::~MinusConnection(){
	delete minus_prod;
}
void MinusConnection::computeZ(){

	memset( this->minus_prod, 0 , FLOAT_SIZE * this->xnum  );

	for (int i = 0; i < xnum; i++){
		
		minus_prod[i] = xlayer->y[i] - xlayer2->y[i];
		//cout<<xlayer->y[i]<<" "<<xlayer2->y[i]<<" "<<minus_prod[i]<<endl;
	}


	if ( isTraining  ){
		 // Widx < 0 
		// cout<<"here"<<endl;
			//for (int i=0;i<10;i++)
			//cout<<ylayer->z[i]<<" "<<minus_prod[i]<<endl; 
	//	cout<<"what"<<Wcoef<<endl;
			alphaXpY( xnum, Wcoef, minus_prod, ylayer->z);
			//for (int i=0;i<10;i++)
			//cout<<ylayer->z[i]<<" "<<minus_prod[i]<<endl; 
			
	}
	else
	{ // testing, dropout
		
			alphaXpY( xnum, Wcoef * ( 1-this->xlayer->p_drop) * (1-this->xlayer2->p_drop), minus_prod, ylayer->z );
		
	}

}
void MinusConnection::updateW(){

	 // Widx < 0
		for (int i = 0; i < xnum; i++){
			xlayer ->dE_dy[i] += ylayer->dE_dz[i] * 1 * Wcoef;
			xlayer2->dE_dy[i] += ylayer->dE_dz[i] * (-1) * Wcoef;
		}
			

}

MergeConnection::MergeConnection(Layer* _x1, Layer* _x2, Layer* _y, int _xnum, int _Widx, float _Wcoef){
	xlayer = _x1;
	xlayer2 = _x2;
	ylayer  = _y;
	Widx = _Widx;
	xnum = _xnum;
	ynum = _xnum;
	Wcoef = _Wcoef;
	//pointwise_prod = new float[_xnum];
}
MergeConnection::~MergeConnection(){
	
}
void MergeConnection::computeZ(){

//	cout<<"x1"<<xlayer->numUnit<<" x2"<<xlayer2->numUnit<<endl;
	for (int i = 0; i < xlayer->numUnit; i++)	
			ylayer->z[i] = xlayer->y[i];
	
	for (int i=xlayer->numUnit; i<xlayer->numUnit+xlayer2->numUnit; i++)
			ylayer->z[i] = xlayer2->y[i-xlayer->numUnit];
	
	

}
void MergeConnection::updateW(){
	// i don't think there should be any weights  
	// so is there anything  should we do...?
	//cout<<"x1"<<endl;
	for(int i = 0; i < xlayer->numUnit; i++){
		{
		//	cout<<xlayer->dE_dy[i]<<" "<<ylayer->dE_dy[i];
			xlayer->dE_dy[i] += ylayer->dE_dy[i];
			
 		}
	}
		//cout<<endl<<"x2"<<endl;
	for(int i = 0; i < xlayer2->numUnit; i++){
		{
			//cout<<xlayer2->dE_dy[i]<<" "<<ylayer->dE_dy[i+xlayer->numUnit];
			xlayer2->dE_dy[i] += ylayer->dE_dy[i+xlayer->numUnit];
 		}
	}
	//cout<<endl;

}
PositivePoolConnection::PositivePoolConnection(Layer * _x, Layer* _y, int _num, int _type){
	xlayer = _x;
	ylayer = _y;
	xnum = _num;
	ynum = _num;
	type = _type;
}

void PositivePoolConnection::computeZ(){
	// ylayer.y = max( ylayer.y, xlayer.y )
	// In this simplified version, we assume x is always greater than 0,
	// which is true for ReLU outputs

	// max pooling
	
	for (int i = 0; i < this->xnum; i++){
		if ( xlayer->y[i] > ylayer->z[i]){
			ylayer->z[i] = xlayer->y[i];
		}
	}
	return;
}
void PositivePoolConnection::updateW(){
	// Actually, no weight is associated with a pooling layer

	// dE_dx = dE_dy if x = y
	// otherwise, dE_dx = 0
	for(int i = 0; i < this->xnum; i++){
		if( xlayer->y[i] == ylayer->y[i]  ){
			xlayer->dE_dy[i] += ylayer->dE_dy[i];
 		}
	}

}
///////////////////////////////////////////////////////
// blas wrapper
// y <- x
inline void icopy(int n, float * x, float * y){
	cblas_scopy(n, x, 1, y, 1);
	return;
}
// y += x
inline void iXpY(int n, float * x, float * y){
	cblas_saxpy(n, 1.0, x, 1, y, 1);
}
// y += alpha * x
inline void alphaXpY(int n, float alpha, float * x, float * y){
	cblas_saxpy(n, alpha, x, 1, y, 1);
}
// y = coef * A * x
inline void matrix_dot_vector(int M, int N, float coef, float * A, enum CBLAS_TRANSPOSE trans, float * x, float * y){
	cblas_sgemv(CblasRowMajor, trans, M, N, coef, A, N, x, 1, 0, y, 1);
}
inline void selfplus_matrix_dot_vector(int M, int N, float coef, float * A, enum CBLAS_TRANSPOSE trans, float * x, float * y){
	cblas_sgemv(CblasRowMajor, trans, M, N, coef, A, N, x, 1, 1.0, y, 1);
}
// z = x .* y
inline void pointwise_dot(float * x, float * y, float *z, int n){
	// x .* y -> z
	// TODO use blas, multiplying with diagnal matrix
	for(int i = 0; i < n; i++)
		z[i] = x[i] * y[i];
}
// C += A * B, where A: <M x K>
//                  B: <K x N>
//                  C: <M x N>
inline void selfplus_matrix_dot_matrix(int M, int N, int K, float coef, float * A, enum CBLAS_TRANSPOSE transA,
								float * B, enum CBLAS_TRANSPOSE transB, float * C){
	cblas_sgemm(CblasRowMajor, transA, transB,
			M, N, K,
			coef, A, K,
			B, N,
	1.0, C, N
	);
}
void GradDescent(int n, float learn_rate, float * grad, float * param){
	cblas_saxpy(n, -learn_rate, grad, 1, param, 1);

}

