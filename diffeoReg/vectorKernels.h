/**
   kernels.h
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

#ifndef _VECTORKERNELS_
#define _VECTORKERNELS_
#include "Vector.h"
#include <fftw3.h>
#include "param_matching.h"

class VectorKernel{
public:
  ~VectorKernel(){
    if (FFT_INITIALIZED) {
      delete [] _nM ;
      fftw_free((void *)_inMap) ;
      fftw_free((void *)  _outMap) ;
      fftw_free((void *) _inMapc) ;
      fftw_free((void *) _outMapc);
      fftw_free((void *) _fMapKernel) ;
      fftw_destroy_plan(_ptoMap) ;
      fftw_destroy_plan(_pfromMap) ;
    }
    FFT_INITIALIZED = false ;
  }
  VectorKernel() {FFT_INITIALIZED=false; TYPE=param_matching::GAUSSKERNEL; dim = 3 ;sigma = 1 ; _sig2 = 2 ; size = 100; ord=0; periodic = 0;} 
  void setParamGaussian(const unsigned int dm, const double sig, const unsigned int sz){dim = dm;  size = sz; sigma=sig; _sig2 = 2*sig*sig;}
  void setParamLaplacian(const unsigned int dm, const double sig, const unsigned int sz, const unsigned int od){dim = dm;  size = sz; sigma=sig; _sig2 = 2*sig*sig; ord=od;}
  void setParam(const param_matching& par){
    TYPE=par.kernel_type ;
    if (TYPE==param_matching::GAUSSKERNEL)
      setParamGaussian(par.dim.size(), par.sigmaKernel, par.sizeKernel);
    else if (TYPE==param_matching::LAPLACIANKERNEL)
      setParamLaplacian(par.dim.size(), par.sigmaKernel, par.sizeKernel, par.orderKernel);
    periodic = par.periodic ;
  }
  void setType(unsigned int tp) {TYPE = tp;}


  void initFFT(Ivector dim,   unsigned int TYPEINIT) {
    //cout << "init kernel" << endl ;
    Ivector Min, Max ;
    _nM = new int[dim.size()] ;
    Max.resize(dim.size()) ;
    Min.resize(dim.size()) ;
#ifdef _PARALLEL_
    fftw_init_threads() ;
    fftw_plan_with_nthreads(param.nb_threads) ;
    //#else
    //  fftw_plan_with_nthreads(1) ;
#endif

    //cout << "build" << endl ;
    build() ;

    //cout << "fft" << endl ;
    for (unsigned int i=0; i < dim.size(); i++)  {
      int delta1 ;
      //    delta1 = (int) floor(size/2.0) ;
      if (periodic == 1)
	delta1 = (kern.d.getM(i) - kern.d.getm(i))/2 ;
      else
	delta1 = 0 ;
      if ((delta1 + 1) < dim[i]) {
	(_nM)[i] = dim[i] + delta1 + 1 ;
	Max[i] = dim[i] + delta1     ;
	Min[i] = 0 ;
      }
      else {
	(_nM)[i] = 2*delta1 +1  ;
	Max[i] = 2*delta1    ;
	Min[i] = 0 ;
      }
    }


    paddedMap.al(Min, Max) ;
    _inMap = (_real *) fftw_malloc(sizeof(_real) * paddedMap.length()) ;
    _outMap = (_real *) fftw_malloc(sizeof(_real) * paddedMap.length()) ;
    _inMapc = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * paddedMap.length()) ;
    _fMapKernel = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * paddedMap.length()) ;
    _outMapc = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * paddedMap.length()) ;
    paddedMap.zero() ;
    paddedMap.symCopy(kern) ;

    //cout << "pto " << _nM[0] << " " << _nM[1] << endl ;
    _ptoMap = fftw_plan_dft_r2c(dim.size(), _nM, _inMap, _outMapc, TYPEINIT) ;
    for (unsigned int i=0; i<paddedMap.length(); i++) {
      (_inMap)[i] = paddedMap[i] ;
    }
    //cout << "execute" << endl ;
    fftw_execute(_ptoMap) ;
    double test1 =0, test2 =0 ;
    for (unsigned int i=0; i<paddedMap.length(); i++) {
      test1 += fabs((_outMapc)[i][0]) ;
      test2 += fabs((_outMapc)[i][1]) ;
      (_fMapKernel)[i][0] = fabs((_outMapc)[i][0]) ;
      (_fMapKernel)[i][1] = 0 ;
    }

    //cout << "pfrom" << endl ;
    _pfromMap = fftw_plan_dft_c2r(dim.size(), _nM, _inMapc, _outMap,  TYPEINIT) ;
    //cout << "end init kernel" << endl ;
    FFT_INITIALIZED = true ;
  }

  void applyWithMask(const VectorMap &in0, const VectorMap &mask, VectorMap &out, std::vector<double> &spaceRes)
  {
    _real nnn = paddedMap.length() ;
    Vector in ;
    out.al(in0.d) ;
    Ivector MIN, MAX ;
    in0.d.putm(MIN) ;
    in0.d.putM(MAX) ;

    for (unsigned int k=0; k<in0.size(); k++) {
      in.copy(in0[k]) ;
      in *= mask[k] ;
      in /= spaceRes[k] ;
      paddedMap.zero() ;
      paddedMap.subCopy(in) ;
      for(unsigned int i=0; i<paddedMap.length(); i++) {
	_inMap[i] = paddedMap[i] ;
      }
      fftw_execute(_ptoMap) ;
      for(unsigned int i=0; i<paddedMap.length(); i++) {
	_inMapc[i][0] = (_outMapc[i][0] * _fMapKernel[i][0] - _outMapc[i][1] * _fMapKernel[i][1]) ;
	_inMapc[i][1] = (_outMapc[i][0] * _fMapKernel[i][1] + _outMapc[i][1] * _fMapKernel[i][0]) ;
      }
      fftw_execute(_pfromMap) ;
      for(unsigned int i=0; i<paddedMap.length(); i++)
	paddedMap[i] = _outMap[i]/nnn ;
      
      paddedMap.extract(out[k], MIN, MAX) ;
      //cout << k << " " << in0.maxNorm() << "(apply kern)" << in.maxAbs() << " " << nnn  << endl ;
      out[k] *= mask[k] ;
      out[k] /= spaceRes[k] ;
    }
  }

  void apply(const VectorMap &in0, VectorMap &out, std::vector<double> &spaceRes)
  {
    _real nnn = paddedMap.length() ;
    Vector in ;
    out.al(in0.d) ;
    Ivector MIN, MAX ;
    in0.d.putm(MIN) ;
    in0.d.putM(MAX) ;

    for (unsigned int k=0; k<in0.size(); k++) {
      in.copy(in0[k]) ;
      in /= spaceRes[k] ;
      paddedMap.zero() ;
      paddedMap.subCopy(in) ;
      for(unsigned int i=0; i<paddedMap.length(); i++) {
	_inMap[i] = paddedMap[i] ;
      }
      fftw_execute(_ptoMap) ;
      for(unsigned int i=0; i<paddedMap.length(); i++) {
	_inMapc[i][0] = (_outMapc[i][0] * _fMapKernel[i][0] - _outMapc[i][1] * _fMapKernel[i][1]) ;
	_inMapc[i][1] = (_outMapc[i][0] * _fMapKernel[i][1] + _outMapc[i][1] * _fMapKernel[i][0]) ;
      }
      fftw_execute(_pfromMap) ;
      for(unsigned int i=0; i<paddedMap.length(); i++)
	paddedMap[i] = _outMap[i]/nnn ;
      
      paddedMap.extract(out[k], MIN, MAX) ;
      out[k] /= spaceRes[k] ;
    }
  }
  void apply(const Vector &in, Vector &out)
  {
    _real nnn = paddedMap.length() ;
    out.al(in.d) ;
    Ivector MIN, MAX ;
    in.d.putm(MIN) ;
    in.d.putM(MAX) ;

 
    paddedMap.zero() ;
    paddedMap.subCopy(in) ;
    for(unsigned int i=0; i<paddedMap.length(); i++) {
      _inMap[i] = paddedMap[i] ;
    }
    fftw_execute(_ptoMap) ;
    for(unsigned int i=0; i<paddedMap.length(); i++) {
      _inMapc[i][0] = (_outMapc[i][0] * _fMapKernel[i][0] - _outMapc[i][1] * _fMapKernel[i][1]) ;
      _inMapc[i][1] = (_outMapc[i][0] * _fMapKernel[i][1] + _outMapc[i][1] * _fMapKernel[i][0]) ;
    }
    fftw_execute(_pfromMap) ;
    for(unsigned int i=0; i<paddedMap.length(); i++)
      paddedMap[i] = _outMap[i]/nnn ;
      
    paddedMap.extract(out, MIN, MAX) ;
  }

  void build(){
    //    TYPE=param.kernel_type ;
    if (TYPE==param_matching::GAUSSKERNEL)
      buildGaussian();
    else if (TYPE==param_matching::LAPLACIANKERNEL)
      buildLaplacian();
  }


  void buildGaussian() {
    VectorMap Id ;
    Vector tmp, r ;
    Ivector I, J ;
    I.resize(dim) ;
    J.resize(dim) ;

    int S4 = size/2 ;
    for (unsigned int i=0; i<dim; i++) {
      //      I[i] = (int) ceil(S4/param.spaceRes[i]) ;  ;
      I[i] = (int) ceil(S4) ;  ;
      J[i] = -I[i] ;
    }
    Domain D(J, I) ;

    Id.idMeshNorm(D) ;
    r.zeros(D) ;
    for (unsigned int i=0; i<dim; i++) {
      tmp.copy(Id[i]) ;
      tmp.sqr() ;
      r += tmp ;
    }

    r /= - 2*sigma*sigma ;
    r.execFunction(tmp, exp) ;
    // tmp /= tmp.sum() ;

    kern.copy(tmp) ;
  }

  void buildLaplacian() {
    VectorMap Id ;
    Vector tmp, r, r1, r2, pol ;
    Ivector I, J ;
    I.resize(dim) ;
    J.resize(dim) ;
    
    int S2 = size/2 ;
    for (unsigned int i=0; i<dim; i++) {
      //     I[i] = (int) ceil(S2/param.spaceRes[i]) ;  ;
      I[i] = (int) ceil(S2) ;  ;
      J[i] = -I[i] ;
    }
    Domain D(J, I) ;

    Id.idMeshNorm(D) ;
    r.zeros(D) ;
    for (unsigned int i=0; i<dim; i++) {
      tmp.copy(Id[i]) ;
      tmp.sqr() ;
      r += tmp ;
    }
    r.execFunction(tmp, sqrt) ;
    r /= sigma ;
    r1.copy(r) ;
    pol.zeros(D) ;
    if (ord == 0)
      pol = 1 ;
    else if (ord == 1) {
      pol = 1 ;
      pol += r ;
    }
    else if (ord == 2) {
      pol = 3 ;
      r1.copy(r) ;
      r1 *= 3 ;
      pol += r1 ;
      r2.copy(r) ;
      r2 *= r ;
      pol +=  r2 ;
    }
    else if (ord == 3) {
      pol = 15 ;
      r1.copy(r) ;
      r1 *= 15 ;
      pol += r1 ;
      r2.copy(r) ;
      r2 *= r ;
      r1.copy(r2) ;
      r1*= 6 ;
      pol += r1 ;
      r2 *= r ;
      pol += r2 ; 
    }
    else if (ord == 4) {
      pol = 105 ;
      r1.copy(r) ;
      r1 *= 105 ;
      pol += r1 ;
      r2.copy(r) ;
      r2 *= r ;
      r1.copy(r2) ;
      r1*= 45 ;
      pol += r1 ;
      r2 *= r ;
      r1.copy(r2) ;
      r1*= 10 ;
      pol +=  r1 ;
      r2 *= r ;
      pol +=  r2 ; 
    }
    else {
      cerr << "Laplacian kernel defined up to order 4" << endl ;
      exit(1) ;
    }
    
    r *= -1 ;
    r.execFunction(tmp, exp) ;
    tmp *= pol ;
    //tmp /= tmp.sum() ;
    kern.copy(tmp) ;
  }

  Vector kern; 
  bool initialized() {return FFT_INITIALIZED;}

private:
  int  *_nM ;
  Vector paddedMap ;
  Vector _kern ;
  fftw_plan  _ptoMap, _pfromMap ;
  _real *_inMap, *_outMap ;
  fftw_complex *_inMapc, *_outMapc ;
  fftw_complex *_fMapKernel ;
  Vector paddedV ;
  bool periodic ;
  //  int ** _n; 
  //  _real ** _in; 
  //  _real **_out ; 
  //  fftw_complex ** _inc ; 
  //  fftw_complex **_outc ; 
  //  fftw_complex **_fKernel;
  //  fftw_plan &_pto;
  //  fftw_plan &_pfrom; 
  bool FFT_INITIALIZED ;
  double sigma, _sig2 ;
  unsigned int dim, size, ord;
  unsigned int TYPE ;

} ;




#endif 
