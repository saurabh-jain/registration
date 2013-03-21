/**
   param_matching.cpp
    Copyright (C) 2010 Laurent Younes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <fstream>
#include <sstream>
#include "param_matching.h"
using namespace std ;

/**
   Main function reading parameter files
*/
param_matching::param_matching() {
  paramMap[(string) "template:"] = TEMPLATE ;
  paramMap[(string) "target:"] = TARGET ;
  paramMap[(string) "tempList:"] = TEMPLIST ;
  paramMap[(string) "targList:"] = TARGLIST ;
  paramMap[(string) "result:"] = RESULT ;
  paramMap[(string) "dataSet:"] = DATASET ;
  paramMap[(string) "dataMomenta:"] = DATAMOM ;
  paramMap[(string) "momentum:"] = MOMENTUM ;
  paramMap[(string) "initialMomentum:"] = INITIALMOMENTUM ;
  paramMap[(string) "miscFiles:"] = AFILES ;
  paramMap[(string) "miscParam:"] = APARAM ;
  paramMap[(string) "outdir:"] = OUTDIR ;
  paramMap[(string) "scale:"] = SCALE ;
  paramMap[(string) "kernel:"] = KERNEL ;
  paramMap[(string) "smoothing_kernel:"] = GAUSS_KERNEL ;
  paramMap[(string) "crop1:"] = CROP1 ;
  paramMap[(string) "crop2:"] = CROP2 ;
  paramMap[(string) "affine:"] = AFFINE ;
  paramMap[(string) "space_resolution:"] = SPACERES ;
  paramMap[(string) "affineReg:"] = AFFINEREG ;
  paramMap[(string) "projectImage:"] = PROJECTIMAGE ;
  paramMap[(string) "applyAffineToTemplate:"] = APPLYAFFINETOTEMPLATE ;
  paramMap[(string) "keepTarget:"] = KEEPTARGET ;
  paramMap[(string) "accuracy:"] = ACCURACY ;
  paramMap[(string) "sigma:"] = SIGMA;
  paramMap[(string) "nb_iter:"] = NB_ITER ;
  paramMap[(string) "nb_semi:"] = NBSEMI ;
  paramMap[(string) "nb_cg_meta:"] = NBCGMETA ;
  paramMap[(string) "epsMax:"] = EPSMAX ;
  paramMap[(string) "time_disc:"] = TIME_DISC;
  paramMap[(string) "parallelTimeDisc:"] = PARALLELTIMEDISC;
  paramMap[(string) "lambda:"] = LAMBDA ;
  paramMap[(string) "gradientThreshold:"] = GRADIENT_THRESHOLD;
  paramMap[(string) "minVarGrad:"] = MINVAREN ;
  paramMap[(string) "expand:"] = EXPAND ;
  paramMap[(string) "maxShootTime:"] = MAXSHOOTTIME ;
  paramMap[(string) "scaleScalars:"] = SCALESCALARS ;
  paramMap[(string) "doNotModifyTemplate:"] = DNMT ;
  paramMap[(string) "doNotModifyImages:"] = DNMI ;
  paramMap[(string) "goldenSearch:"] = GS ;
  paramMap[(string) "binarize:"] = BINARIZE ;
  paramMap[(string) "flipTarget:"] = FLIPTARGET ;
  paramMap[(string) "expandToMaxSize:"] = EXPANDTOMAXSIZE ;
  paramMap[(string) "useVectorMomentum:"] = USEVECTORMOMENTUM ;
  paramMap[(string) "revertIntensities:"] = REVERTINTENSITIES ;
  paramMap[(string) "matchDensities:"] = MATCHDENSITIES ;
  paramMap[(string) "nbThreads:"] = NB_THREADS ;
  paramMap[(string) "continue:"] = CONTINUE ;
  paramMap[(string) "saveMovie:"] = SAVEMOVIE ;
  paramMap[(string) "periodic:"] = PERIODIC ;
  paramMap[(string) "quiet:"] = QUIET ;
  paramMap[(string) "normalizeKernel:"] = NORMALIZEKERNEL ;

  affMap[(string) "none"] = ID ;
  affMap[(string) "translation"] = TRANSLATION ;
  affMap[(string) "rotation"] = ROTATION ;
  affMap[(string) "similitude"] = SIMILITUDE ;
  affMap[(string) "special"] = SPECIAL ;
  affMap[(string) "general"] = GENERAL ;

  kernelMap[(string) "gauss"] = GAUSSKERNEL ;
  kernelMap[(string) "gaussLandmarks"] = GAUSSKERNELLMK ;
  kernelMap[(string) "laplacian"] = LAPLACIANKERNEL ;

  strcpy(fileTemp,"\0")  ;
  strcpy(fileTarg,"\0")  ;
  strcpy(fileMom1,"\0")  ;
  strcpy(fileMom2,"\0")  ;
  strcpy(fileInitialMom,"\0")  ;
  strcpy(outDir,"\0")  ;
  cont = false ;
  saveMovie = false ;
  revertIntensities = false ;
  useVectorMomentum = false ;
  expandToMaxSize = false ;
  binarize = false ;
  binThreshold = 1 ;
  nb_threads = 1 ;
  gs = false ;
  keepTarget = false ;
  applyAffineToTemplate = false ;
  initTimeData = false ;
  keepFFTPlans = false ;
  readBinaryTemplate = false ;
  doNotModifyTemplate = false ;
  doNotModifyImages = false ;
  foundTemplate = false ;
  foundTarget = false ;
  foundResult = false ;
  foundAuxiliaryFile = false ;
  foundAuxiliaryParam = false ;
  foundInitialMomentum =  false ;
  flipTarget = false;
  flipDim = 0 ;
  saveProjectedMomentum = true ;
  matchDensities = false ;
  foundScale = false ;
  gradInZ0 = true ;
  lambda = 1 ; 
  time_disc =30; 
  nb_iter=1000 ; 
  nbCGMeta = 3 ;
  sigma = 1 ;
  epsMax = 10 ;
  accuracy = 1.1 ;
  sigmaGauss = -1; 
  gradientThreshold = -1 ;
  epsilonTangentProjection = 0.01 ;
  sizeGauss = 50; 
  kernel_type = GAUSSKERNEL ;
  sigmaKernel = .1; 
  orderKernel = 0 ;
  inverseKernelWeight = 0.0001 ;
  sizeKernel = 100; 
  nb_iterAff = 1000 ;
  affine_time_disc = 20 ;
  tolGrad = 0 ;
  verb=true;
  printFiles = 1 ;
  nb_semi = 3 ;
  minVarEn = 0.001 ;
  type_group = ID ;
  scaleScalars = false ;
  scaleThreshold = 100 ;
  crop1 = false ;
  crop2 = false ;
  affine = false ;
  spRes = false ;
  projectImage = false;
  expand_value = -1 ;
  Tmax = 10 ; 
  parallelTimeDisc = 20 ;
  kernelNormalization = 1 ;
  normalizeKernel = true ; 
  doDefor = true ;
  periodic = 0 ;
}


void param_matching::read(char * fname)
{
  ifstream ifs ;
  ifs.open(fname) ;

  if (!ifs)
    {
      cerr << "Unable to open " << fname << endl ;
      exit(1) ;
    }
  else {
    cout << "reading " << fname << endl ;
  }


  string keyword, line ;
  char lstr[256], *pch ;
  std::vector<string> input ;
  bool readDim = false ;

  while (!ifs.eof()) {
    getline(ifs, line) ;
    //    cout << line << endl ; 
    if (line.length() > 0 && line[0] != '#') {
      //      istringstream str(line) ;
      strcpy(lstr, line.c_str()) ;
      //      while(!str.eof()) {
      pch = strtok(lstr, " \t\n\r") ;
      while(pch != NULL) {
	if (!readDim) {
	  //	  str >> ndim ;
	  ndim = atoi(pch) ;
	  readDim = true ;
	  pch = strtok(NULL, " \t\n\r") ;
	}
	else {
	  keyword = pch ; 
	  pch = strtok(NULL, " \t\n\r") ;
	  //	  str >> keyword ;
	  //	if (!str.eof()) {
	  input.resize(input.size()+1) ;
	  input[input.size()-1] = keyword ;
	  //	  cout << "*" << keyword<< endl ;
	}
      }
    }
  }
  read(input) ;
  ifs.close() ;
}

void param_matching::read(int argc, char ** argv)
{
  std::vector<string> input ;
  input.resize(argc-2) ;
  for (int i=2; i<argc; i++) {
      input[i-2] = argv[i] ;
      // cout << input[i-2] << endl ;
  }
  read(input) ;
}


void param_matching::read(std::vector<string> &input)
{
  int nb ;
  string keyword ;
  Ivector im, iM ;
  string affKey, kernelKey ;
  unsigned int k =0 ;
  // cout << "input size " << input.size() << endl ; 
  // for (int kk=0; kk< (int) input.size(); kk++)
  // cout << input[kk] << endl ;

  //  for (map<string, int>::iterator I = paramMap.begin(); I != paramMap.end(); ++I)
  // cout << I->first << endl ;


  while (k<input.size()) {
    keyword = input[k++] ;
    //    if (k<input.size()) {
    // cout << "keyword = " << keyword << "   " << k  << "  " << input.size()<< endl ;
      map<string, int>::iterator I = paramMap.find(keyword);
      map<string, int>::iterator IA ;
      if (I == paramMap.end()) {
	cerr << "unknown keyword in param file: " << keyword << endl ;
	exit(1) ;
      }
      switch (I->second) {
      case TEMPLATE:
	strcpy(fileTemp, input[k++].c_str()) ;
	foundTemplate = true ;
	break ;
      case RESULT:
	strcpy(fileResult, input[k++].c_str()) ;
	foundResult = true ;
	break ;
      case TARGET:
	strcpy(fileTarg, input[k++].c_str()) ;
	foundTarget = true ;
	break ;
      case DATASET:
	nb = atoi(input[k++].c_str()) ;
	dataSet.resize(nb) ;
	foundTarget = true ;
	for (int i=0; i<nb; i++)
	  dataSet[i] = input[k++] ;
	break ;
      case TEMPLIST:
	nb = atoi(input[k++].c_str()) ;
	fileTempList.resize(nb) ;
	foundTemplate = true ;
	for (int i=0; i<nb; i++)
	  fileTempList[i] = input[k++] ;
	break ;
      case TARGLIST:
	nb = atoi(input[k++].c_str()) ;
	fileTargList.resize(nb) ;
	foundTarget = true ;
	for (int i=0; i<nb; i++)
	  fileTargList[i] = input[k++] ;
	break ;
      case DATAMOM:
	nb = atoi(input[k++].c_str()) ;
	dataMom.resize(nb) ;
	for (int i=0; i<nb; i++)
	  dataMom[i] = input[k++] ;
	break ;
      case AFILES:
	foundAuxiliaryFile = true ;
	nb = atoi(input[k++].c_str()) ;
	auxFiles.resize(nb) ;
	for (int i=0; i<nb; i++)
	  auxFiles[i] = input[k++] ;
	break ;
      case APARAM:
	foundAuxiliaryParam = true ;
	nb = atoi(input[k++].c_str()) ;
	auxParam.resize(nb) ;
	for (int i=0; i<nb; i++)
	  auxParam[i] = atof(input[k++].c_str()) ;
	break ;
      case INITIALMOMENTUM:
	strcpy(fileInitialMom, input[k++].c_str()) ;
	foundInitialMomentum = true ;
	break ;
      case MOMENTUM:
	strcpy(fileMom1, input[k++].c_str()) ;
	strcpy(fileMom2, input[k++].c_str()) ;
	break ;
      case USEVECTORMOMENTUM:
	useVectorMomentum = true ;
	break ;
      case REVERTINTENSITIES:
	revertIntensities = true ;
	break ;
      case PERIODIC:
	periodic = true ;
	break ;
      case QUIET:
	verb = false ;
	break ;
      case CONTINUE:
	cont = true ;
	break ;
      case NB_THREADS:
	nb_threads = atoi(input[k++].c_str()) ;
	break ;
      case MATCHDENSITIES:
	matchDensities = true ;
	break ;
      case SCALE:
	foundScale = true ; 
	dim.resize(ndim) ;
	for(int i=0; i<ndim; i++) {
	  dim[i] = atof(input[k++].c_str());
	}
	break ;
      case BINARIZE:
	binarize = true ; 
	binThreshold = atof(input[k++].c_str());
	break ;
      case FLIPTARGET:
	flipTarget = true ; 
	flipDim = atoi(input[k++].c_str());
	break ;
      case EXPANDTOMAXSIZE:
	expandToMaxSize = true ; 
	break ;
      case KERNEL:
	kernelKey = input[k++] ;
	IA = kernelMap.find(kernelKey);
	if (IA == kernelMap.end()) {
	  cerr << "unknown kernel key" << endl ;
	  exit(1) ;
	}
	else {
	  kernel_type = IA -> second ;
	  if (kernel_type == param_matching::GAUSSKERNELLMK)
	    sigmaKernel = atof(input[k++].c_str()) ;
	  else if (kernel_type == param_matching::GAUSSKERNEL) {
	    sigmaKernel = atof(input[k++].c_str()) ;
	    sizeKernel = 2 * atoi(input[k++].c_str()) ;
	    sizeKernel = 4 * (sizeKernel/4) ;
	  }
	  else if (kernel_type == param_matching::LAPLACIANKERNEL) {
	    sigmaKernel = atof(input[k++].c_str()) ;
	    sizeKernel = 2 * atoi(input[k++].c_str()) ;
	    sizeKernel = 4 * (sizeKernel/4) ;
	    orderKernel = atoi(input[k++].c_str()) ;
	  }
	}
	break;
      case GAUSS_KERNEL :
	sigmaGauss = 0.05 ;
	sizeGauss = 2 * atoi(input[k++].c_str());
	break;
      case MINVAREN :
	minVarEn = atof(input[k++].c_str());
	break;
      case GRADIENT_THRESHOLD :
	gradientThreshold = atof(input[k++].c_str());
	break;
      case ACCURACY :
	accuracy = atof(input[k++].c_str());
	break;
      case OUTDIR:
	strcpy(outDir, input[k++].c_str()) ;
	break;
      case EXPAND:
	expand_margin.resize(ndim) ;
	for(int i=0; i<ndim; i++)
	  expand_margin[i] = atoi(input[k++].c_str()); ;
	expand_value = atof(input[k++].c_str());;
	break;
      case CROP1:
	im.resize(ndim) ;
	iM.resize(ndim) ;
	for(int i=0; i<ndim ; i++) {
	  im[i] = atoi(input[k++].c_str()) ;
	  iM[i] = atoi(input[k++].c_str()); 
	  // cout << iM[i] << endl ; 
	}
	// cout << "in crop1 " << input[k] << endl ; 
	crop1 = true ;
	cropD1.create(im, iM) ;
	break ;
      case CROP2:
	im.resize(ndim) ;
	iM.resize(ndim) ;
	for(int i=0; i<ndim ; i++) {
	  im[i] = atoi(input[k++].c_str()) ;
	  iM[i] = atoi(input[k++].c_str()); 
	}
	crop2 = true ;
	cropD2.create(im, iM) ;
	break ;
      case AFFINE:
	for(int i=0; i<ndim; i++)
	  for(int j=0; j<=ndim; j++)
	    affMat[i][j] = atof(input[k++].c_str()) ;
	affine = true ;
	break ;
      case MAXSHOOTTIME:
	Tmax =  atoi(input[k++].c_str());
	break ; 
      case NBCGMETA:
	nbCGMeta =  atoi(input[k++].c_str());
	break ;
      case SPACERES:
	spaceRes.resize(ndim) ;
	for (int i=0; i<ndim; i++)
	  spaceRes[i] =  atof(input[k++].c_str()) ;
	spRes = true ;
	break ;
      case PROJECTIMAGE:
	projectImage = true ;
	break ;
      case GS:
	gs = true ;
	break ;
      case AFFINEREG:
	int nb_it ;
	affKey = input[k++] ;
	nb_it =  atoi(input[k++].c_str());
	//cout << "affKey = " << affKey << endl ;
	IA = affMap.find(affKey);
	if (IA == affMap.end()) {
	  cerr << "unknown affine key" << endl ;
	  exit(1) ;
	}
	else
	  type_group = IA -> second ;
	if (nb_it > 0)
	 nb_iterAff = nb_it ;
	break ;
      case KEEPTARGET:
	keepTarget = true ;
	break;
      case APPLYAFFINETOTEMPLATE:
	applyAffineToTemplate = true ;
	break;
      case SCALESCALARS:
	scaleScalars = true ;
	scaleThreshold =  atof(input[k++].c_str());
	break ;
      case PARALLELTIMEDISC:
	parallelTimeDisc =  atoi(input[k++].c_str());
	break ;
      case TIME_DISC:
	time_disc =  atoi(input[k++].c_str());
	break ;
      case LAMBDA:
	lambda =  atof(input[k++].c_str());
	sigma = 1/sqrt(lambda) ;
	break;
      case NB_ITER:
	nb_iter =  atoi(input[k++].c_str());
	break ;
      case NBSEMI:
	nb_semi =  atoi(input[k++].c_str());
	break ;
      case SIGMA:
	sigma =  atof(input[k++].c_str());
	lambda = 1 / (sigma*sigma) ;
	break ;
      case EPSMAX:
	epsMax =  atof(input[k++].c_str());
	break ;
      case DNMT:
	doNotModifyTemplate = true ;
	break;
      case DNMI:
	doNotModifyImages = true ;
	break;
      case SAVEMOVIE:
	saveMovie = true ;
	break ;
      default:
	cerr << "Error in parameter syntax " << endl ;
	exit(1) ;
	//      }
    }
  }

  if (!foundScale) {
    // cout << "no scale" << endl ;
    dim.resize(ndim) ;
    for (int i=0; i<ndim; i++) 
      dim[i] = 1 ;
  }

  if (!spRes) {
    spaceRes.resize(ndim) ;
    for (int i=0; i<ndim; i++) 
      spaceRes[i] = 1 ;
  }
  if (scaleScalars && gradientThreshold < 0)
    gradientThreshold = scaleThreshold / 10 ;

}

