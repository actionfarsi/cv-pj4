#include "SupportVectorMachine.h"

SupportVectorMachine::SupportVectorMachine(): 
_model(NULL), 
_data(NULL)
{
}

SupportVectorMachine::SupportVectorMachine(const char* modelFName):
_model(NULL), 
_data(NULL)
{
	load(modelFName);
}

void
SupportVectorMachine::deinit()
{
	if(_model != NULL) svm_free_and_destroy_model(&_model);
	if(_data != NULL) delete [] _data;
	_model = NULL;
	_data = NULL;	
}

SupportVectorMachine::~SupportVectorMachine()
{
	deinit();
}

void 
SupportVectorMachine::train(const std::vector<float>& labels, const FeatureSet& fset, double C)
{	
	if(labels.size() != fset.size()) throw std::runtime_error("Database size is different from feature set size!");
//	printf("fsize:%d\n",fset.size());
	_fVecShape = fset[0].Shape();

	// Figure out size and number of feature vectors
	int nVecs = labels.size();
	CShape shape = fset[0].Shape();
	int dim = shape.width * shape.height * shape.nBands;
//	printf("dim:%d\n",dim);
	// Parameters for SVM
	svm_parameter parameter;
	parameter.svm_type = C_SVC;
	parameter.kernel_type = LINEAR;
	parameter.degree = 0;
	parameter.gamma = 0;
	parameter.coef0 = 0;
	parameter.nu = 0.5;
	parameter.cache_size = 100;  // In MB
	parameter.C = C;
	parameter.eps = 1e-3;
	parameter.p = 0.1;
	parameter.shrinking = 1;
	parameter.probability = 0;
	parameter.nr_weight = 0; // ?
	parameter.weight_label = NULL;
	parameter.weight = NULL;
	//cross_validation = 0;

	// Allocate memory
	svm_problem problem;
	problem.l = nVecs;
	problem.y = new double[nVecs];
	problem.x = new svm_node*[nVecs];
	if(_data) delete [] _data;

	/******** BEGIN TODO ********/
	// Copy the data used for training the SVM into the libsvm data structures.
	// Put the feature vectors in _data.value and labels in problem.y
	//
	// Hint:
	// * Don't forget to set _data[].index to the corresponding dimension in
	//   the original feature vector. You also need to set _data[].index to -1
	//   right after the last element of each feature vector




	// Vector containing all feature vectors. svm_node is a struct with
	// two fields, index and value. Index entry indicates position 
	// in feature vector while value is the value in the original feature vector, 
	// each feature vector of size k takes up k+1 svm_node's in _data
	// the last one being simply to indicate that the feature has ended by setting the index
	// entry to -1
	_data = new svm_node[nVecs * (dim + 1)]; 
	//int pos=0;
	//int currBand=0;
	//int currRow=0;
	//int currCol=0;
	//int first=0;
	//for(int j=0; j<nVecs*(dim + 1); j++){


	//	//go through each vector by counting pos
	//	if(pos<dim){
	//		_data[j].index=pos;
	//		pos++;
	//	}else{
	//		//reset when position is beyond that of dim
	//		_data[j].index=-1; //this is the actual last one
	//		pos=0;
	//	}
	//	int xval,yval,bval;
	//	if((currBand+1)<shape.nBands){
	//		xval=currCol;
	//		yval=currRow;
	//		if(first){
	//		bval=currBand;
	//		currBand++; //only band value changes, x any y doesnt
	//		first=1;
	//		}else{
	//			currBand++; //only band value changes, x any y doesnt
	//			bval=currBand;
	//		}
	//	}else{
	//		if((currCol+1)<shape.width){
	//		//all bands done, on to next x
	//		currCol++;
	//	    //row remains same
	//		
	//		}else{
	//		//down with all value in row, next y
	//		currCol=0;
	//		currRow++; //increment row
	//		currBand=0;
	//		}
	//		xval=currCol;
	//		yval=currRow;
	//		currBand=0; //reset currBand so we do the other bands as well
	//		bval=currBand;
	//	}
	//	//xval=currRow*shape.width*shape.nBands
	//	////if j is less than width, then its the first row
	//	//if(j<shape.width){
	//	//	xval=_data[j].index;
	//	//	yval=0;
	//	//	bval=0;
	//	//}else{
	//	//	//use mod to get the x position
	//	//	xval=(j) % shape.width;
	//	//	//find y(row)
	//	//	yval=floor((double)j+1 / shape.width);
	//	//	if(yval>shape.height){
	//	//	bval=floor((double)yval/shape.height);
	//	//	//reduce yval
	//	//	yval=yval-bval*shape.height;
	//	//	}else{
	//	//	bval=0;
	//	//	}
	//	//}



	//	_data[j].value=fset[floor((double)j/(dim+1))].Pixel(xval,yval,bval);


	//}
	int d=0;
	int pos=0;

	for(int n=0; n<nVecs; n++){
		for (int y=0; y<shape.height; y++){
			for(int x=0; x<shape.width; x++){
				for(int b=0; b<shape.nBands; b++){
					if(pos<dim){
						_data[d].index=pos;
						pos++;
						_data[d].value=fset[n].Pixel(x,y,b);
						//printf("pos");
					}
					//if this is true, we add the value for index -1
					if(pos==dim){
						//printf("vec");
						d++;
						_data[d].index=-1;
						pos=0;
						//reset, end of vector
						_data[d].value=fset[n].Pixel(x,y,b);
						problem.y[n]=labels[n];
						problem.x[n]=&_data[d-dim];
					
					}
					d++;

				}
			}
		}
	}
	//for(int i=0; i<nVecs; i++){
	//	problem.y[i]=labels[i];
	//	problem.x[i]=&_data[i*(dim+1)];
	//}
	//printf("TODO: SupportVectorMachine.cpp:87\n"); exit(EXIT_FAILURE); 

	/******** END TODO ********/

	// Train the model
	if(_model != NULL) svm_free_and_destroy_model(&_model);
	_model = svm_train(&problem, &parameter);

	// Cleanup
	delete [] problem.y;
	delete [] problem.x;

}

float 
SupportVectorMachine::predict(const Feature& feature) const
{
	CShape shape = feature.Shape();
	int dim = shape.width * shape.height * shape.nBands;

	svm_node* svmNode = new svm_node[dim + 1];

	svm_node* svmNodeIter = svmNode;

	for(int y = 0, k = 0; y < shape.height; y++) {
		float* data = (float*) feature.PixelAddress(0, y, 0);
		for (int x = 0; x < shape.width * shape.nBands; x++, data++, k++, svmNodeIter++) {
			svmNodeIter->index = k;
			svmNodeIter->value = *data;
		}
	}
	svmNodeIter->index = -1;

	double decisionValue;
	float label = svm_predict_values(_model, svmNode, &decisionValue);

	delete [] svmNode;

	return decisionValue;
}

std::vector<float> 
SupportVectorMachine::predict(const FeatureSet& fset) const
{
	std::vector<float> preds(fset.size());
	for(int i = 0; i < fset.size(); i++) {
		preds[i] = predict(fset[i]);
	}

	return preds;
}

double 
SupportVectorMachine::getBiasTerm() const
{
	if(_model == NULL) throw CError("Asking for SVM bias term but there is no model. Either load one from file or train one before.");
	return _model->rho[0];
}

Feature 
SupportVectorMachine::getWeights() const
{
	if(_model == NULL) throw CError("Asking for SVM weights but there is no model. Either load one from file or train one before.");

	Feature weightVec(_fVecShape);
	weightVec.ClearPixels();

	weightVec.origin[0] = _fVecShape.width / 2;
	weightVec.origin[1] = _fVecShape.height / 2;

	int nSVs = _model->l; // number of support vectors
	
	for(int s = 0; s < nSVs; s++) {
		double coeff = _model->sv_coef[0][s];
		svm_node* sv = _model->SV[s];

		for(int y = 0, d = 0; y < _fVecShape.height; y++) {
			float* w = (float*) weightVec.PixelAddress(0,y,0);
			for(int x = 0; x < _fVecShape.width * _fVecShape.nBands; x++, d++, w++, sv++) {
				assert(sv->index == d);
				*w += sv->value * coeff;
			}
		}
	}

	return weightVec;
}

void 
SupportVectorMachine::load(const char *filename)
{
	FILE* f = fopen(filename, "rb");
	if(f == NULL) throw CError("Failed to open file %s for reading", filename);
	this->load(f);
}

void 
SupportVectorMachine::load(FILE* fp)
{
	deinit();
	fscanf(fp, "%d %d %d", &_fVecShape.width, &_fVecShape.height, &_fVecShape.nBands);
	_model = svm_load_model_fp(fp);
	if(_model == NULL) {
		throw CError("Failed to load SVM model");
	}	
}

void 
SupportVectorMachine::save(FILE* fp) const
{
	if(_model == NULL) throw CError("No model to be saved");
	
	fprintf(fp, "%d %d %d\n", _fVecShape.width, _fVecShape.height, _fVecShape.nBands);

 	if(svm_save_model_fp(fp, _model) != 0) {
 		throw CError("Error while trying to write model to file");
 	}	
}

void 
SupportVectorMachine::save(const char* filename) const
{
	FILE *fp = fopen(filename, "wb");
	if(fp == NULL) {
		throw CError("Could not open file %s for writing.", filename);
	}

	save(fp);
	if (ferror(fp) != 0 || fclose(fp) != 0) {
		throw CError("Error while closing file %s", filename);
	}
}

CFloatImage 
SupportVectorMachine::predictSlidingWindow(const Feature& feat) const
{
	CFloatImage score(CShape(feat.Shape().width,feat.Shape().height,1));
	score.ClearPixels();

	/******** BEGIN TODO ********/
	// Sliding window prediction. 
	//
	// In this project we are using a linear SVM. This means that 
	// it's classification function is very simple, consisting of a
	// dot product of the feature vector with a set of weights learned
	// during training, followed by a subtraction of a bias term
	//
	//          pred <- dot(feat, weights) - bias term
	//
	// Now this is very simple to compute when we are dealing with
	// cropped images, our computed features have the same dimensions
	// as the SVM weights. Things get a little more tricky when you
	// want to evaluate this function over all possible subwindows of
	// a larger feature, one that we would get by running our feature
	// extraction on an entire image. 
	//
	// Here you will evaluate the above expression by breaking
	// the dot product into a series of convolutions (remember that
	// a convolution can be though of as a point wise dot product with
	// the convolution kernel), each one with a different band.
 	//
	// Convolve each band of the SVM weights with the corresponding
	// band in feat, and add the resulting score image. The final
	// step is to subtract the SVM bias term given by this->getBiasTerm().
	//
	// Hint: you might need to set the origin for the convolution kernel
	// in order to get the result from convoltion to be correctly centered
	// 
	// Useful functions:
	// Convolve, BandSelect, this->getWeights(), this->getBiasTerm()

	//printf("TODO: SupportVectorMachine.cpp:274\n"); exit(EXIT_FAILURE); 
	//get band size
	int nSize=feat.Shape().nBands;
	//declar features
	Feature weightVec(_fVecShape);
	weightVec.ClearPixels();
	Feature featVec(_fVecShape);
	featVec.ClearPixels();
	Feature resultVec(_fVecShape);
	resultVec.ClearPixels();

	/*
	Feature invert(3,3,1);
	invert.Pixel(0,0,1)=0;
	invert.Pixel(1,0,1)=0;*/

	for(int b=0; b<nSize; b++){
		//iterate through each band
		BandSelect(this->getWeights(),weightVec,0,b);
		BandSelect(feat,featVec,0,b);
	//	printf("BEFORE: w:%d, h:%d, size: \n\r",weightVec.Shape().width,weightVec.Shape().height);
		//sets the origin by diving the width and height by 2
		weightVec.origin[0]=ceil((double)weightVec.Shape().width/2);
		weightVec.origin[1]=ceil((double)weightVec.Shape().height/2);
	
		//printf("x:%d, y:%d, size: \n\r",featVec.origin[0],featVec.origin[1]);
		/convolve
		Convolve(featVec,resultVec,weightVec);

		for(int x=0; x<feat.Shape().width; x++){
			for(int y=0; y<feat.Shape().height; y++){
				//tally up score
				score.Pixel(x,y,b)+=resultVec.Pixel(x,y,b);
			}
		}
	
	}
	//subtract final bias term
	//for(int b=0; b<nSize; b++){
	for(int x=0; x<feat.Shape().width; x++){
			for(int y=0; y<feat.Shape().height; y++){
				score.Pixel(x,y,0)=score.Pixel(x,y,0)-this->getBiasTerm();
			}
	}
	//}
	
	/******** END TODO ********/

	return score;
}

>>>>>>> origin/le
