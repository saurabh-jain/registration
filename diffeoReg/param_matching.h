/**
   param_matching.h
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
#ifndef _PARAM_MATCHING_
#define _PARAM_MATCHING_
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "Ivector.h"
using namespace std ;




/** 
    parameter class. Reads parameter files written as follows

Start with an integer which is the dimension of the dataset
(dim = 2,3; 1 will come soon). It then consists in a sequence of keywords followed by some
complementary input. Here is a detailed description (keywords always
end with ':') 

  template: filename 
	First image file
  target: filename 
	Second image file
  tempList: nbFiles file1 file2
             A list of template files (for multimodal matching)
  targList: nbFiles file1 file2
             A list of target files (for multimodal matching)

   result: fileName
   An ascii file to store numerical output (energy, distance)

 dataSet: N filename1 ... filenameN 
             List of images (for procrustean averaging)

dataMomenta: N filename1 ... filenameN 
             List of initial momenta (relative to a template for simple averaging)

momentum: filename1 filename2 
        Two files for parallel translation. The first momentum provides 
	the geodesic along which the second momentum is transported. 
	The momenta are outputs of matchShoot.

initialMomentum: fileName
         To initialize matching procedures with a non zero momentum

outdir: the directory where the output images are saved

scale: s1 .. sdim
	Integers providing scaling factors for each dimension; default 1 for each dimension

kernel: type shape size [degree]
	parameters defining the smoothing kernel for vector fields:
            type is either gauss or laplacian
            shape defines a normalized scale
            size is a scaling factor (in pixels)
           degree is required for the laplacian kernel only, and is an integer from 0 to 4

gauss_kernel: size
	parameters specifying a gaussian kernel spread for preliminar
	image smoothing; default 1 

crop1: min1 max1 .... mindim maxdim
	specifies a multidimensional rectangle to crop the first image
	(default: no cropping).

crop2: min1 max1 .... mindim maxdim
	specifies a multidimensional rectangle to crop the second image
	(default: no cropping).

affine: a11 a12 ...
	applies the inverse of dim x dim+1 matrix A to the target
	image. No affine registration is run.

affineReg: affKey num_iterations
	computes an optimal affine registration of the target before
	matching. affKey is one of the following: rotation (rigid
	transf.), similitude (rigid+scaling), special (volume
	conserving) or general (the whole affine group).

space_resolution: res1 ... resdim
	allows for non constant resolutions across dimensions. Default
	1 ... 1

minVarGrad: m
	upper bound on the energy variation in gradient descent stopping rule.
	Default: 0.001

* sigma: s
	the coefficient before the data attchment term is 1/s^2 for
	matchShoot. default: 1

* lambda: s
	the coefficient before the data attachment term is s for
	metamorphosis (sorry...) default:1

* nb_iter: N
	maximal number of gradient descent iterations in matching
	programs

* nb_semi: n
   number of iteration for semi implicit integration (default: 0)

* epsMax: eps
	maximum step in gradient descent. Default: 10

* time_disc: T
	the time discretization for metamorphoses; default: 30

parallelTimeDisc: T
	Time discretization for parallel translation; default 20

keepTarget:
	Work with the original target file (apply affine transformations to the template)

expand: margin1 ... marginDim value
    Expands image with margink in kth dimension, with specified value. If value =-1, picks average value on the image boundary

expandToMaxSize:
    If present, expand images to fit the largest one

maxShootTime:
   Maximum number of steps for shooting procedure

scaleScalars:
   Rescale all image values between 0 and 100

goldenSearch:
   Use golden line search for gradient descent

binarize:
    Work with Thresholded binary images

revertIntensities:
     Revert image internsities

nb_threads:
    Maximum number of threads for parallel implementation


******   Input file format
	Images are read in most image formats in 2D, and in Analyze format in 3D.
*/

class param_matching
{
public:
  int ndim ;
  char fileTemp[256] ;
  char fileTarg[256] ;
  char fileMom1[256] ;
  char fileMom2[256] ;
  char fileInitialMom[256] ;
  char fileResult[256] ;
  char outDir[256] ;
  std::vector<string> fileTempList ; 
  std::vector<string> fileTargList ; 
  std::vector<string> dataSet ; 
  std::vector<string> dataMom ;
  std::vector<string> auxFiles ;
  std::vector<_real> auxParam ;
  std::vector<_real> dim ;
  std::vector<_real> spaceRes ;
  int nb_threads ;
  string kernelType ;
  //for radial kernel
  _real sigmaKernel ;
  int sizeKernel ;
  int orderKernel ;
  _real inverseKernelWeight ;
  // for Laplacian Kernel
  //  _real alpha ;
  _real sigmaGauss ;
  int Tmax;
  int sizeGauss ;
  void read(char *) ;
  void read(int argc, char ** argv) ;
  void read(std::vector<string> &input) ;
  int type_group ;
  int kernel_type ;
  bool verb ;
  int printFiles ;
  bool cont ;
  bool gs ;
  bool gradInZ0 ;
  bool foundTemplate ;
  bool foundResult ;
  bool foundTarget ;
  bool foundScale ;
  bool foundAuxiliaryFile ;
  bool foundAuxiliaryParam ;
  bool crop1, crop2 ;
  bool affine, spRes ;
  bool projectImage ;
  bool scaleScalars ;
  bool binarize ;
  bool expandToMaxSize ;
  bool matchDensities ;
  bool revertIntensities ;
  bool flipTarget ;
  int flipDim ;
  double binThreshold ;
  int affine_time_disc ;
  int nb_iterAff ;
  int nb_semi ;
  int nbCGMeta ;
  _real tolGrad ;
  _real accuracy ;
  _real minVarEn ;
  _real expand_value ;
  std::vector<int> expand_margin ;
  _real affMat[DIM_MAX][DIM_MAX+1] ;
  Domain cropD1, cropD2 ;

  _real sigma ;
  _real epsMax ;
  _real gradientThreshold ;
  _real epsilonTangentProjection ;
  int nb_iter ;

  int parallelTimeDisc ;
  int time_disc ;
  bool useVectorMomentum ;
  bool doDefor ;
  bool doNotModifyTemplate ;
  bool doNotModifyImages ;
  bool readBinaryTemplate ;
  bool keepFFTPlans ;
  bool initTimeData ;
  bool foundInitialMomentum ;
  bool keepTarget ;
bool applyAffineToTemplate ;
  bool saveProjectedMomentum ;
  _real lambda ;
_real scaleThreshold ;
  _real kernelNormalization ;
bool saveMovie ;
bool periodic ;

  void defaultArray(int dm)  {
    ndim = dm ;
    dim.resize(ndim) ;
    for (int i=0; i<ndim; i++)
      dim[i] = 1 ;
    
    spaceRes.resize(ndim) ;
      for (int i=0; i<ndim; i++)
	spaceRes[i] = 1 ;
  }

  void copy(const param_matching p0) 
    {
      ndim = p0.ndim ;
      dim.resize(p0.dim.size()) ;
      for (unsigned int i=0; i<p0.dim.size(); i++)
	dim[i] = p0.dim[i] ;

      cropD1.copy(p0.cropD1) ;
      cropD2.copy(p0.cropD2) ;
      strcpy(fileTemp, p0.fileTemp) ;
      strcpy(fileTarg, p0.fileTarg) ;
      strcpy(fileMom1, p0.fileMom1) ;
      strcpy(fileMom2, p0.fileMom2) ;
      strcpy(fileInitialMom, p0.fileInitialMom) ;
      strcpy(outDir, p0.outDir) ;
      fileTempList.resize(p0.fileTempList.size()) ; 
      for(unsigned int i=0; i<fileTempList.size(); i++)
	fileTempList[i] = p0.fileTempList[i] ;
      fileTargList.resize(p0.fileTargList.size()) ; 
      for(unsigned int i=0; i<fileTargList.size(); i++)
	fileTargList[i] = p0.fileTargList[i] ;
      dataSet.resize(p0.dataSet.size()) ; 
      for(unsigned int i=0; i<dataSet.size(); i++)
	dataSet[i] = p0.dataSet[i] ;
      dataMom.resize(p0.dataMom.size()) ; 
      for(unsigned int i=0; i<dataMom.size(); i++)
	dataMom[i] = p0.dataMom[i] ;
      auxFiles.resize(p0.auxFiles.size()) ; 
      for(unsigned int i=0; i<auxFiles.size(); i++)
	auxFiles[i] = p0.auxFiles[i] ;
      auxParam.resize(p0.auxParam.size()) ; 
      for(unsigned int i=0; i<auxParam.size(); i++)
	auxParam[i] = p0.auxParam[i] ;
      spaceRes.resize(p0.spaceRes.size()) ;
      for (unsigned int i=0; i<p0.spaceRes.size(); i++)
	spaceRes[i] = p0.spaceRes[i] ;

      for (int i = 0; i<DIM_MAX; i++)
	for (int j=0; j<= DIM_MAX; j++)
	  affMat[i][j] = p0.affMat[i][j] ;

      nb_threads = p0.nb_threads ;
      kernelType = p0.kernelType ;
      sigmaKernel =p0.sigmaKernel ;
      sizeKernel = p0.sizeKernel ;
      orderKernel =p0.orderKernel ;
      inverseKernelWeight = p0.inverseKernelWeight ;
      sigmaGauss = p0.sigmaGauss ;
      Tmax = p0.Tmax ;
      sizeGauss = p0.sizeGauss ;
      type_group = p0.type_group ;
      kernel_type = p0.kernel_type ;
      verb = p0.verb ;
      printFiles = p0.printFiles ;
      gs = p0.gs ;
      gradInZ0 = p0.gradInZ0 ;
      foundTemplate = p0.foundTemplate ;
      foundResult = p0.foundResult ;
      foundTarget = p0.foundTarget ;
      foundScale = p0.foundScale ;
      flipTarget = p0.flipTarget ;
      flipDim = p0.flipDim ;
      crop1 =p0.crop1 ;
      crop2 = p0.crop2 ;
      affine = p0.affine ;
      spRes = p0.spRes ;
      projectImage = p0.projectImage ;
      scaleScalars = p0.scaleScalars ;
      binarize = p0.binarize ;
      expandToMaxSize = p0.expandToMaxSize ;
      matchDensities = p0.matchDensities ;
      revertIntensities = p0.revertIntensities ;
      binThreshold =p0.binThreshold ;
      affine_time_disc = p0.affine_time_disc ;
      nb_iterAff =  p0.nb_iterAff ;
      nb_semi =  p0.nb_semi ;
      tolGrad = p0.tolGrad ;
      accuracy = p0.accuracy ;
      minVarEn  = p0.minVarEn ;
      expand_value = p0.expand_value ;
      expand_margin.resize(p0.expand_margin.size()) ;
      for (unsigned int i=0; i<expand_margin.size(); i++)
	expand_margin[i] = p0.expand_margin[i] ;
      sigma = p0.sigma ;
      epsMax = p0.epsMax ;
      gradientThreshold = p0.gradientThreshold ;
      epsilonTangentProjection = p0.epsilonTangentProjection ;
      nb_iter = p0.nb_iter ;
      cont = p0.cont ;
      applyAffineToTemplate = p0.applyAffineToTemplate ;
      kernelNormalization = p0.kernelNormalization ;

      parallelTimeDisc = p0.parallelTimeDisc ;
      time_disc = p0.time_disc ;
      useVectorMomentum = p0.useVectorMomentum ;
      doDefor = p0.doDefor ;
      doNotModifyTemplate = p0.doNotModifyTemplate ;
      doNotModifyImages = p0.doNotModifyImages ;
      readBinaryTemplate = p0.readBinaryTemplate ;
      keepFFTPlans = p0.keepFFTPlans ;
      initTimeData = p0.initTimeData ;
      foundInitialMomentum = p0.foundInitialMomentum ;
      keepTarget = p0.keepTarget ;
      saveProjectedMomentum = p0.saveProjectedMomentum ;
      lambda = p0.lambda ;
      scaleThreshold = p0.scaleThreshold ;
      saveMovie = p0.saveMovie ;
      periodic = p0.periodic ;
    }

  void printDefaults() 
    {
      param_matching p0 ;
      cout << "kernelType: gauss" << endl ;
      cout << "kernel shape parameter: " << p0.sigmaKernel << endl ;
      cout << "kernel size parameter: " << p0.sizeKernel << endl ;
      cout << "kernel order parameter: " << p0.orderKernel << endl << endl ;

      cout << "match densities instead of images(flag): "<< p0.matchDensities << endl << endl;

      cout << "initial smoothing parameter: " << p0.sizeGauss << endl ;
      cout << "flip target (flag): " << p0.flipTarget << endl;
      cout << "scale scalars (flag): "<<p0.scaleScalars << endl;
      cout << "recert image intensity (flag): "<<p0.revertIntensities << endl;
      cout << "expand images to maximum size (flag): "<<p0.expandToMaxSize << endl << endl;

      cout << "Spatial resolution: " << 1 << " for all dimensions" << endl ;
      cout << "threshold for image gradients (relative): " << gradientThreshold << endl ;
      cout << "data attachment penalty (sigma): " <<  p0.sigma << endl << endl;

      cout << "number of iterations for semi-Lagrangian integration: " << p0.nb_semi << endl ;
      cout << "increase tolerance parameter for gradient descent: " << p0.tolGrad << endl ;
      cout << "Minimal relative variation of objective function for optimization: " << p0.minVarEn << endl ;
      cout << "max step in gradient descent (epsMax): " <<  p0.epsMax << endl ;
      cout << "maximal number of iterations (nb_iter): " << p0.nb_iter << endl ;
      cout << "line search flag: " << p0.gs << endl;
      cout << "gradient in Z0 (flag): " << p0.gradInZ0 << endl << endl ;

      cout << "maximal shooting time: " << p0.Tmax << endl ;
      cout << "time discretization for parallel transport (inages): " <<  p0.parallelTimeDisc << endl ;
      cout << "time discretization for velocity matching and metamorphosis: " << p0.time_disc << endl << endl ;

      cout <<"use vector momentum (flag for covMatrix): " <<  p0.useVectorMomentum << endl;
    }

  param_matching() ;
  enum GROUPS {ID, TRANSLATION, ROTATION, SIMILITUDE, SPECIAL, GENERAL}  ; 
  map<string, int> affMap ;
  enum kernels {GAUSSKERNEL, LAPLACIANKERNEL,GAUSSKERNELLMK} ;
  map<string, int> kernelMap ;
 private:
  enum command {TEMPLATE, TEMPLIST, DATASET, TARGET, TARGLIST, RESULT, DATAMOM, AFILES, 
		APARAM, MOMENTUM, SCALE, KERNEL, GAUSS_KERNEL,
		OUTDIR, CROP1, CROP2, AFFINE, SPACERES, AFFINEREG, DNMT, DNMI,
		PROJECTIMAGE, ACCURACY, SIGMA, NB_ITER, EPSMAX, NBSEMI, NBCGMETA, GS,
		TIME_DISC, LAMBDA, MINVAREN, EXPAND, MAXSHOOTTIME, SCALESCALARS, KEEPTARGET,
		APPLYAFFINETOTEMPLATE, FLIPTARGET, GRADIENT_THRESHOLD,
		PARALLELTIMEDISC, INITIALMOMENTUM, BINARIZE, EXPANDTOMAXSIZE,USEVECTORMOMENTUM,
                REVERTINTENSITIES, MATCHDENSITIES,NB_THREADS, CONTINUE,SAVEMOVIE,PERIODIC,QUIET
  } ;
  map<string, int> paramMap ;
} ;

#endif
