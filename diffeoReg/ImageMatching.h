/**
   ImageMatching.h
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

#ifndef _IMAGEMATCHING_
#define _IMAGEMATCHING_
#include "deformableImage.h"
#include "matchingBase.h"
#include "affineReg.h"



/**
  Class for image comparison; extends MatchingBase and adds smoothing and affine registration functions
*/
class ImageMatching: public MatchingBase<deformableImage>
{
public:
  using MatchingBase<deformableImage>::param ;
  using MatchingBase<deformableImage>::Template ;
  using MatchingBase<deformableImage>::Target ;
  using MatchingBase<deformableImage>::Load ;
  using MatchingBase<deformableImage>::LoadTemplate ;
  using MatchingBase<deformableImage>::imageDim ;
  using MatchingBase<deformableImage>::_kern ;
  //  using MatchingBase<deformableImage>::paddedMap ;
  //  using MatchingBase<deformableImage>::_nM ;
  //  using MatchingBase<deformableImage>::_inMap ;
  //  using MatchingBase<deformableImage>::_outMap ;
  //  using MatchingBase<deformableImage>::_inMapc ;
  //  using MatchingBase<deformableImage>::_outMapc ;
  //  using MatchingBase<deformableImage>::_fMapKernel ;
  //  using MatchingBase<deformableImage>::_ptoMap ;
  //  using MatchingBase<deformableImage>::_pfromMap ;
  //  using MatchingBase<deformableImage>::FFT_INITIALIZED ;
  using MatchingBase<deformableImage>::gamma ;
  using MatchingBase<deformableImage>::affTrans ;

  ImageMatching(){epsMin = 1e-7; epsSmall = 1e-10;}
  ImageMatching(param_matching &par){init(par);}
  void   init(param_matching &par) {
    epsMin = 1e-7; 
    epsSmall = 1e-10;
    param.copy(par) ;
    cout << "load file" << endl ;
    Load() ;
    Template.computeGradient(param.spaceRes,param.gradientThreshold);
  }

  ImageMatching(char *file, int argc, char** argv) {init(file, argc, argv) ;} 
  void init(char *file, int argc, char** argv) {
    epsMin = 1e-7; 
    epsSmall = 1e-10;
    param.read(file) ;
    param.read(argc, argv) ;
    cout << "load file" << endl ;
    Load() ;
    Template.computeGradient(param.spaceRes,param.gradientThreshold);
    cout << "loaded" << endl ;
  }
  ImageMatching(char *file, int k){init(file, k);}
  void init(char *file, int k){
    epsMin = 1e-7; 
    epsSmall = 1e-10;
    param.read(file) ;
    if (k==0)
      Load() ;
    else
     LoadTemplate() ;
    Template.computeGradient(param.spaceRes,param.gradientThreshold);
  }

  //  _real integrateMatrix(const Matrix &gm, Matrix &res) const ;
  template <class foo> void tryfoo(foo & x){cout << x.size() << endl ;}
  void f2(){Vector z ; tryfoo(z);}
  //  void convImage(const deformableImage & in, deformableImage& out) {convImage(in.img(), out.img(), param.sigmaGauss, param.sizeGauss);}
  void convImage(const deformableImage & in, deformableImage& out) {convImage(in.img(), out.img());}
  void revertIntensities(deformableImage &tmp){tmp.revertIntensities();}
  void crop1(deformableImage &src, deformableImage & dest) {cout << "Running crop1" << endl ; src.crop(param.cropD1, dest) ;}
  void crop2(deformableImage &src, deformableImage & dest) {src.crop(param.cropD2, dest) ;}
  void expandBoundary(deformableImage &img){
    if (param.expand_value > -0.000000001)
      img.expandBoundary(param.expand_margin, param.expand_value) ;
    else
      img.expandBoundary(param.expand_margin) ;
  }

  void rescaleDim(const deformableImage &src, deformableImage &dest) {
    imageDim.resize(param.dim.size()) ;
    if (param.dim.size())
      for(unsigned int i=0; i<param.dim.size(); i++) {
	imageDim[i] =  (int) floor(src.domain().getM(i)/param.dim[i]) ;
      }
    else
      src.domain().putM(imageDim) ;
    Domain D(imageDim) ;
    src.rescale(D, dest) ;
    //    cout << imageDim << endl ;
    if (param.verb)
      cout << "New dimensions: " << param.dim.size() << endl << dest.domain() << endl ;
  }

  void rescaleTarget(const deformableImage &src, deformableImage &dest) {
    Domain D(imageDim) ;
    //    cout << imageDim << endl ;
    src.rescale(D, dest) ;
    if (param.verb)
      cout << "New dimensions: " << endl << dest.domain() << endl;
  }
  
  void binarize(deformableImage &tmp){ tmp.binarize(param.binThreshold, 100) ;}
  void flip(deformableImage &tmp, int dm){tmp.flip(dm) ;}

  void get_template(deformableImage &tmp) {tmp.get_image(param.fileTemp, param.dim.size()); tmp.domain().putM(imageDim) ; cout << "get_t " << tmp.img().maxAbs() << endl ;}
  void get_binaryTemplate(deformableImage &tmp) {tmp.img().read(param.fileTemp);tmp.domain().putM(imageDim) ;}
  void get_target(deformableImage &tmp) {tmp.get_image(param.fileTarg, param.dim.size());}
  void get_binaryTarget(deformableImage &tmp) {tmp.img().read(param.fileTarg);}
  void scaleScalars() {
      Template.scaleScalars(param.scaleThreshold) ;
      if (param.foundTarget)
	Target.scaleScalars(param.scaleThreshold) ;
  }

  void expandToMaxSize(deformableImage &tmp1, deformableImage &tmp2) {
    Ivector Min, Max, Min0, Max0 ;
    Min.resize(param.ndim);
    Max.resize(param.ndim);
    Min0.resize(param.ndim);
    Max0.resize(param.ndim);
    for (int k=0; k<param.ndim; k++) {
      Min[k] = tmp1.domain().getm(k) ;
      if (tmp2.domain().getm(k) < Min[k])
	Min[k] = tmp2.domain().getm(k) ;
      Max[k] = tmp1.domain().getM(k) ;
      if (tmp2.domain().getM(k) > Max[k])
	Max[k] = tmp2.domain().getM(k) ;
    }
    Domain D(Min, Max) ;
    deformableImage tmp0 ;
    tmp0.zeros(D) ;
    for (int k=0; k< param.ndim; k++) {
      Min0[k] = Min[k] + (Max[k] - Min[k] - tmp1.domain().getM(k) + tmp1.domain().getm(k)) / 2 ;
      Max0[k] = Min0[k] + (tmp1.domain().getM(k) - tmp1.domain().getm(k)) ;
    }
    //	  cout << D << endl ;
    tmp0.subCopy(tmp1, Min0, Max0) ;
    tmp1.copy(tmp0) ;
    tmp1.expandBoundary(D) ;
    tmp0.zero() ;
    for (int k=0; k< param.ndim; k++) {
      Min0[k] = Min[k] + (Max[k] - Min[k] - tmp2.domain().getM(k) + tmp2.domain().getm(k)) / 2 ;
      Max0[k] = Min0[k] + (tmp2.domain().getM(k) - tmp2.domain().getm(k)) ;
    }
    //	  cout << D << endl ;
    tmp0.subCopy(tmp2, Min0, Max0) ;
    tmp2.copy(tmp0) ;
    //	  tmp1.expandBoundary(D) ;
    //	  tmp2.expandBoundary(D) ;
    if (param.doNotModifyTemplate) {
      cout << "Warning: Template may have been modified" << endl ;
    }
    imageDim.copy(Max) ;
  }

  void affineInterpolation(deformableImage &src, deformableImage &dest){ affineInterp(src, dest, param.affMat) ;}
  
  /*
  void init_fft()
  {
    Vector kern ;
    unsigned int TYPEINIT = FFTW_MEASURE ;

    cout << "Initializing vector field fft's" << endl ;
    
    Diffeomorphisms::init_fft(imageDim, param.kernel_type, param.sigmaKernel, param.sizeKernel, param.orderKernel, _kern, paddedMap, &_nM, & _inMap, &_outMap, 
			      & _inMapc, &_outMapc, &_fMapKernel, _ptoMap, _pfromMap, TYPEINIT) ;
    FFT_INITIALIZED = true ;
    if (param.sigmaGauss > 0) {
      cout << "Initializing image fft's " << endl ;
      Diffeomorphisms::init_fft(imageDim, param_matching::GAUSSKERNEL, param.sigmaGauss, param.sizeGauss, 0, kern, paddedImage, &_nI, & _inImage, &_outImage, 
				& _inImagec, &_outImagec, &_fImageKernel, _ptoImage, _pfromImage, TYPEINIT) ;
      FFT_IMG_INITIALIZED = true ;
    }
  }
  */



  void convImage(const Vector &in, Vector &out) {
    // padding the in image ;
    cout << "CI1 " << in.min() << endl ;
    if (param.sigmaGauss <0) {
      //    cout << "Not smoothing image" << endl ;
      out.copy(in) ;
    }
    else {
      if (_imageKern.initialized() == false) {
	_imageKern.setType(param_matching::GAUSSKERNEL) ;
	_imageKern.setParamGaussian(param.ndim, param.sigmaGauss, param.sizeGauss) ;
	_imageKern.initFFT(imageDim, FFTW_MEASURE) ;
      }

      //      _imageKern.apply(in, out) ;
      Vector in2 ;
      _real mm = 0 ;
      in2.copy(in) ;

      mm = in.avgBoundary() ;
      in2 -= mm ;
      _imageKern.apply(in2, out) ;
      out /= _imageKern.kern.sum() ;
      
      out += mm ;
      out -= out.min();
      /* bool check = false ; */
      /* if (check) { */
      /* 	Vector tmp, diff ; */
      /* 	convImageNofft(in, tmp) ; */
      /* 	diff.copy(tmp) ; */
      /* 	diff -= out ; */
      /* 	double chk = diff.norm2() ; */
      /* 	cout << "Check " << chk << endl ; */
      /* 	if (chk > 1e-5) */
      /* 	  out.copy(tmp) ; */
      /* } */
    cout << "CI2 " << mm << " " <<  out.max() << " " << _imageKern.kern.sum() << endl ;
    }

  }


  /* void convImage(const Vector &in, Vector &out, _real sig, int sz) { */
  /*   // padding the in image ; */
  
  /*   int *_n ; */
  /*   Vector paddedI, kern ; */
  /*   fftw_plan _ptoI, _pfromI ; */
  /*   _real *_inI, *_outI ; */
  /*   fftw_complex *_inIc, *_outIc ; */
  /*   fftw_complex *_fIKernel ; */
  /*   Ivector imDim, MIN, MAX ; */

  /*   in.d.putM(MAX) ; */
  /*   in.d.putm(MIN) ; */
  /*   imDim.copy(MAX) ; */
  /*   imDim -= MIN ; */

  /*   //  cout << "imge:" << sz << endl << endl ; */

  /*   unsigned int TYPEINIT = FFTW_ESTIMATE ; */
  /*   Diffeomorphisms::init_fft(imDim, param_matching::GAUSSKERNEL, sig, sz, 0, kern, paddedI, &_n, & _inI, &_outI,  */
  /* 			      & _inIc, &_outIc, &_fIKernel, _ptoI, _pfromI, TYPEINIT) ; */

  /*   Vector in2 ; */
  /*   _real mm = 0 ; */
  /*   in2.copy(in) ; */
  /*   in2.domain().shiftMinus(MIN) ; */
    
  /*   mm = in2.avgBoundary() ; */
  /*   in2 -= mm ; */

  /*   _real nnn =  paddedI.length() ; */
  /*   paddedI.zero() ; */
  /*   paddedI.subCopy(in2) ; */
  /*   for(unsigned int i=0; i<paddedI.length(); i++) { */
  /*     _inI[i] = paddedI[i] ; */
  /*   } */
  /*   fftw_execute(_ptoI) ; */
  /*   for(unsigned int i=0; i<paddedI.length(); i++) { */
  /*     _inIc[i][0] = (_outIc[i][0] * _fIKernel[i][0] - _outIc[i][1] * _fIKernel[i][1]) ; */
  /*     _inIc[i][1] = (_outIc[i][0] * _fIKernel[i][1] + _outIc[i][1] * _fIKernel[i][0]) ; */
  /*   } */
  /*   fftw_execute(_pfromI) ; */
  /*   for(unsigned int i=0; i<paddedI.length(); i++) */
  /*     paddedI[i] = _outI[i]  / nnn ; */

  /*   Ivector MIN2, MAX2 ; */
  /*   in2.d.putm(MIN2) ; */
  /*   in2.d.putM(MAX2) ; */
  /*   out.al(in2.d) ; */
  /*   paddedI.extract(out, MIN2, MAX2) ; */
  /*   out += mm ; */

  /*   out.domain().shiftPlus(MIN) ; */
  /*   out /= kern.sum() ; */
  
  /*   fftw_destroy_plan(_pfromI) ; */
  /*   fftw_destroy_plan(_ptoI) ; */
  /*   fftw_free((void *) _inI)  ; */
  /*   fftw_free((void *) _outI)  ; */
  /*   fftw_free((void *) _inIc)  ; */
  /*   fftw_free((void *) _outIc)  ; */
  /*   fftw_free((void *) _fIKernel)  ; */
  /*   delete [] _n ; */
  /* } */


  /* void convImageNofft(const Vector &in, Vector &out) { */
  /*   // padding the in image ; */

  /*     Vector in2 ; */
  /*     _real mm = 0 ; */
  /*     in2.copy(in) ; */

  /*     mm = in.avgBoundary() ; */
  /*     in2 -= mm ; */

  /*     Vector pI ; */
  /*   //    paddedImage.zero() ; */
  /*   Vector kern; */
  /*   makeKernelGauss(kern, param.sigmaGauss, param.dim.size(), param.sizeGauss) ; */
  /*   Ivector minD, maxD ; */
  /*   minD.resize(in.d.n) ; */
  /*   maxD.resize(in.d.n) ; */
  /*   in.d.putm(minD) ; */
  /*   in.d.putM(maxD) ; */
  /*   for(unsigned int k=0; k<in.d.n; k++) { */
  /*     minD[k] -= param.sizeGauss ; */
  /*     maxD[k] += param.sizeGauss ; */
  /*   } */
  /*   Domain d(minD, maxD) ; */
  /*   pI.zeros(d) ; */

  /*   pI.subCopy(in2) ; */
    
  /*   /\*  char path[256] ; */
  /* sprintf(path, "%s/PI", param.outDir) ; */
  /* pI.write_image(path) ; */
  /*   *\/ */
  /*   out.al(in.d) ; */
  /*   Ivector I, J, K ; */
  /*   I.resize(in.d.n) ; */
  /*   J.resize(in.d.n) ; */
  /*   K.resize(in.d.n) ; */
  /*   out.d.putm(I) ; */
  /*   //cout << "loop conv nofft" << endl ; */
  /*   //cout << in.d << endl ; */
  /*   //cout << kern.d << endl ; */
  /*   //cout << pI.d << endl ; */
  /*   for(unsigned int i=0; i<out.length(); i++) { */
  /*     //cout << "i " << flush ; */
  /*     out[i] = 0 ; */
  /*     //cout << "1" << flush ; */
  /*     kern.d.putm(J) ; */
  /*     //cout << "2" << flush ; */
  /*     //_real test = 0 ; */
  /*     for (unsigned int j=0; j<kern.length(); j++) { */
  /* 	for (unsigned int k=0; k < I.size(); k++){  */
  /* 	  K[k] = I[k] - J[k];  */
  /* 	  //	  cout << ":" << K[k] << ": " << flush ;  */
  /* 	} */
  /* 	//      cout << "3 " << pI.d.position(K) << " " << flush ; */
  /* 	_real u = pI[pI.d.position(K)] ; */
  /* 	//cout << "4" << flush ; */
  /* 	//	test += u ; */
  /* 	out[i] += kern[j] * u ; */
  /* 	//cout << "5" << flush ; */
  /* 	kern.d.inc(J) ; */
  /* 	//cout << "6" << flush ; */
  /*     } */
  /*     out.d.inc(I) ; */
  /*   } */
  /*   //cout << "end conv nofft " << " " << in2.sum() << " " << out.sum() << endl ; */
  /*   out += mm ; */
  /*   out /= kern.sum() ; */
  /*   /\* */
  /* sprintf(path, "%s/in", param.outDir) ; */
  /* in.write_image(path) ; */
  /* sprintf(path, "%s/in2", param.outDir) ; */
  /* in2.write_image(path) ; */
  /* sprintf(path, "%s/out", param.outDir) ; */
  /* out.write_image(path) ; */
  /* sprintf(path, "%s/kernIm", param.outDir) ; */
  /* kern.write_image(path) ; */
  /*   *\/ */
  /* } */


  /**
     Affine registration between two images (calls affineMatch)
  */
  virtual void affineReg() {cout << "Warning affineReg: component is not implemented (1)" << endl ;}

  void getPreprocessedImagesAffine(Vector &Tp, Vector &Tg, Domain &D) {
    Vector Te, Ta,tmp ;
    // cropping out empty regions
    cropEmptyRegions(Template.img(), Target.img(), Te, Ta, D) ;
    tmp.copy(Te) ;
    _real m1 = tmp.stdBoundary() ;
    if (m1 < 0.001)
      m1 = tmp.medianBoundary() ;
    else
      m1 = 0 ;
    tmp -= m1;
    tmp.expandBoundaryCentral(5, 0) ;
    //    convImage(tmp, Tp, 0.05, param.sizeGauss) ;
    Tp.copy(tmp) ;
    Tp.expandBoundaryCentral(1, 0) ;
    
    tmp.copy(Ta) ;
    m1 = tmp.stdBoundary() ;
    if (m1 < 0.001)
      m1 = tmp.medianBoundary() ;
    else
      m1 = 0 ;
    tmp -= m1;
    tmp.expandBoundaryCentral(5, 0) ;
    Tg.copy(tmp) ;
    //  convImage(tmp, Tg, 0.05, param.sizeGauss) ;
    Tg.expandBoundaryCentral(1, 0) ;
  }

  template <class AE>
  void initializeAffineReg(AE & enerAff) {
    if (param.type_group != param_matching::ID) {
      Vector Tp, Tg, tmp ;
      Domain D ;
      std::vector<_real> sz, x0 ;
      Ivector MIN ;
      getPreprocessedImagesAffine(Tp, Tg, D) ;
      D.putm(MIN) ;
      
      int N = param.dim.size() ;
      x0.resize(N) ;
      sz.resize(N) ;
      for (int i=0; i<N; i++) {
	x0[i] = (Tp.d.getM(i) + Tp.d.getm(i))/2.0 ;
	sz[i] = (Tp.d.getM(i) - Tp.d.getm(i)+1) ;
      }
      
      enerAff.init(Tp, Tg, sz, x0, MIN) ;
    }
  }


  template <class AE>
  void affineReg(AE & enerAff) {
    cout << "affine registration" << endl ;
    int N = param.dim.size() ;
    gamma.zeros(N+1, N+1) ;

    if (param.type_group != param_matching::ID) {
      //    cout << "affine match" << endl;
      affineMatch(gamma, enerAff) ;
    }
  }


  void finalizeAffineTransform(Matrix &gamma, Ivector &MIN) {
    int N = param.dim.size() , T = param.affine_time_disc ;
    std::vector<Matrix> AT ;
    AT.resize(T + 1) ;
    AT[0].idMatrix(N+1) ;
    AT[1].copy(gamma) ;
    AT[1] /= T ;
    AT[1] += AT[0] ;
    for (int t=2; t<=T; t++)
      AT[t].matProduct(AT[1], AT[t-1]) ;
    _real mat[DIM_MAX][DIM_MAX+1] ;

    if (!param.applyAffineToTemplate) {
      affTrans.copy(AT[T]) ;
      cout << affTrans << endl ;
      double x ;
      for (int k=0; k<N; k++) {
	x = MIN[k] ;
	for(int l=0; l<N; l++)
	  x -= affTrans(k,l)*MIN[l] ;
	affTrans(k, N) += x ;
      }
      deformableImage tmp ;
      for (int i=0; i<N; i++)
	for (int j=0; j<=N; j++)
	  mat[i][j] = affTrans(i, j) ;
      ::affineInterp(Target.img(), tmp.img(), mat) ;
      Target.copy(tmp) ;
    }
    else {
      Matrix  invAT ;
      invAT.inverse(AT[T]) ;
      affTrans.copy(invAT) ;
      double x ;
      for (int k=0; k<N; k++) {
	x = MIN[k] ;
	for(int l=0; l<N; l++)
	  x -= affTrans(k,l)*MIN[l] ;
	affTrans(k, N) += x ;
      }
      for (int i=0; i<N; i++)
	for (int j=0; j<=N; j++)
	  mat[i][j] = affTrans(i, j) ;
      deformableImage tmp ;
      ::affineInterp(Template.img(), tmp.img(), mat) ;
      Template.copy(tmp) ;
      Template.computeGradient(param.spaceRes,param.gradientThreshold);
    }

    if (param.verb) {
      cout << "Estimated matrix: " << endl; 
      affTrans.Print() ;
    }
  }

  /**
     affine registration between images; multigrid local exploration
  */
  template <class AE>
  _real affineMatch(Matrix &gamma0, AE& enerAff) {
    unsigned int N = param.dim.size() ;
    gamma.zeros(N+1, N+1) ;

    //    std::vector<_real> sz, x0 ;
    //    initializeAffineReg(enerAff, sz,x0, MIN) ;
    affineRegistration<AE> affreg ;
    return affreg.localSearch(gamma, enerAff.sz, enerAff.x0, param.nb_iterAff, param.type_group, enerAff) ;
  }

  /** Affine registration: Gradient descent
   */
  _real gradientStepAff(Vector& Tp, Vector &Tg, _real &step) {
    Matrix grad, gamma0 ;
    _real energy, energy0 = enerAff(Tp, Tg) ;
    
    affineGradient(Tp, Tg, grad) ;
    gamma0.copy(gamma) ;

    energy = 2*energy0 + 1 ;
    while( (energy > energy0 *(1 + param.tolGrad)  + 1e-10) & (step > 0.0000001)) {
      gamma.copy(grad) ;
      gamma *= -step/100;
      gamma += gamma0 ;
      energy = enerAff(Tp, Tg) ;
      
      if (energy > (1 + param.tolGrad) * energy0 + 1e-10) {
	step = step * 0.5 ;
	cout << "step reduction (gradient): " << step << " " << energy0 << " " << energy << endl ;
      }
    }
    step *= 2 ;
    if (step < 0.0001)
      step = 0.0001 ;
    
    return energy ;
  }


  _real enerAff(const Vector &Tp, const Vector &Tg) const {
    unsigned int N = param.dim.size() ;
    Vector DI ;
    Matrix AT ;
    _real det = gamma.integrate(AT, 10), sqdet = sqrt(det) ;
    
    _real mat[DIM_MAX][DIM_MAX+1] ;
    for (unsigned int i=0; i<N; i++)
      for (unsigned int j=0; j<=N; j++)
	mat[i][j] = AT(i, j) ;
    
    affineInterp(Tg, DI, mat) ;
    DI *= Tp ;

    _real res = (Tp.norm2() * sqdet + Tg.norm2()/sqdet - 2* DI.sum()*sqdet) /DI.length() ;
    return res ;
  }

  _real enerAff(const Vector &Tp, const Vector &Tg, const _real TpNorm, const _real TgNorm) const {
    Vector DI ;
    Matrix AT ;
    unsigned int N = param.dim.size() ;
    _real det = gamma.integrate(AT, 10), sqdet = sqrt(det) ;

    _real mat[DIM_MAX][DIM_MAX+1] ;
    for (unsigned int i=0; i<N; i++)
      for (unsigned int j=0; j<=N; j++)
	mat[i][j] = AT(i, j) ;
    
    affineInterp(Tg, DI, mat) ;
    
    _real res = (TpNorm * sqdet + TgNorm/sqdet - 2* DI.sumProd(Tp)*sqdet) /DI.length() ;
    return res ;
  }


  /**
     gradient for affine registration
  */
  void affineGradient(Vector& Tp, Vector & Tg, Matrix &grad) {
    unsigned int N = param.dim.size() ;
    Matrix AT, tmp, tmp1, tmp2, grad2, invAT ;
    std::vector<Matrix> allAT ;
    unsigned int T = param.affine_time_disc ;
    //  T = 1000 ;
    
    allAT.resize(T+1) ;
    
    AT.idMatrix(N+1) ;
    tmp.copy(gamma) ;
    tmp /= T ;
    AT += tmp ;
    allAT[0].idMatrix(N+1) ;
    allAT[1].copy(AT) ;
    _real det = std::pow(allAT[1].determinant(), (int) T) ;
    _real sqdet = sqrt(det) ;

    for (unsigned int t=2; t<=T; t++) {
      allAT[t].matProduct(allAT[1], allAT[t-1]) ;
    }
    invAT.inverse(allAT[T]) ;
    
    _real imat[DIM_MAX][DIM_MAX+1] ;
    for (unsigned int i=0; i<N; i++)
      for (unsigned int j=0; j<=N; j++) {
	imat[i][j] = allAT[T](i, j) ;
      }

    if (param.verb){
      cout << endl << "Current matrix" << endl ;
      allAT[T].Print() ;
    }

    Vector DI ;
    VectorMap G0, G1 ;

    VectorMap Id ;
    Id.idMesh(Tp.d) ;

    gradient(Tg, G0, param.spaceRes) ;
    G1.zeros(Tg.d) ;
    for (unsigned int i = 0 ; i<N; i++)
      affineInterp(G0[i], G1[i], imat) ;
    affineInterp(Tg, DI, imat) ;
    DI *= Tp ;
    
    G1 *= Tp ;
    G1 *= sqdet ;

    Matrix Omega1 ;
    Omega1.resize(N+1, N+1) ;
    _real I2 = Tp.norm2(), J2 = Tg.norm2(), DI2 = DI.sum() ;

    for (unsigned int i=0; i<N; i++) {
      for (unsigned int j=0; j<N; j++) { 
	Omega1(i, j) = (- 2*G1[i].sumProd(Id[j])*param.spaceRes[j] + ((0.5 * I2 - DI2)*sqdet - 0.5*J2/sqdet)*invAT(j,i))
	  /Tp.length() ;
      }
      Omega1(i, N) = - 2*G1[i].sum()/Tp.length() ;
    }
    for (unsigned int j=0; j<=N; j++) {
      Omega1(N, j) = 0 ;
    }

    grad.matProductT(Omega1,allAT[T-1]) ;
    for (unsigned int t=2; t<=T; t++) {
      tmp.TmatProduct(allAT[t-1], Omega1) ;
      tmp1.matProductT(tmp, allAT[T-t]) ;
      grad += tmp1 ;
    }

    grad /= T ;
    for (unsigned int j=0; j<=N; j++) {
      grad(N, j) = 0 ;
    }


    for (unsigned int i=0; i<N; i++) 
      grad(i, N) *= (Tp.d.getM(i) - Tp.d.getm(i) + 1) ;


    Matrix grad0 ;

    if (param.type_group == param_matching::ROTATION) {
      grad0.resize(N, N) ;
      for (unsigned int i=0; i<N; i++)
	for (unsigned int j=0; j<N; j++)
	  grad0(i, j) = (grad(i, j) - grad(j, i)) /2 ;
      for (unsigned int i=0; i<N; i++)
	for (unsigned int j=0; j<N; j++)
	  grad(i, j) = grad0(i, j) ;
    }
    else if (param.type_group == param_matching::SIMILITUDE) {
      _real tr = 0  ;
      grad0.resize(N, N) ;
      for (unsigned int i=0; i<N; i++) {
	tr += grad(i, i) ;
	for (unsigned int j=0; j<N; j++)
	  grad0(i, j) = (grad(i, j) - grad(j, i)) /2 ;
      }
      tr /= N ;
      for (unsigned int i=0; i<N; i++) {
	for (unsigned int j=0; j<N; j++)
	  grad(i, j) = grad0(i, j) ;
	grad(i, i) += tr ;
      }
    }
    else if (param.type_group == param_matching::SPECIAL) {
      _real tr = 0  ;
      for (unsigned int i=0; i<N; i++) {
	tr += grad(i, i) ;
      }
      tr /= N ;
      for (unsigned int i=0; i<N; i++) {
	grad(i, i) -= tr ;
      }
    }
    else if (param.type_group == param_matching::TRANSLATION) {
      for (unsigned int i=0; i<N; i++) 
	for (unsigned int j=0; j<N; j++)
	  grad(i, j) = 0 ;
    }
    
    if (param.verb) {
      cout << endl << "Gradient: " << endl ;
      grad.Print() ;
    }
  }







  //  ~ImageMatching(){}
  protected:
  VectorKernel _imageKern ;
  _real epsMin ;
  _real epsSmall ;
};


#endif
