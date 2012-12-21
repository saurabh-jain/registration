/**
   pointSetMatching.h
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
#ifndef _POINTSETMATCHING_
#define _POINTSETMATCHING_
#include "pointSet.h"
#include "matchingBase.h"
#include "affineReg.h"



/**
  Class for image comparison; extends MatchingBase and adds smoothing and affine registration functions
*/
class PointSetMatching: public MatchingBase<PointSet>
{
public:
  using Diffeomorphisms::param;
  //  using MatchingBase<PointSet>::param ;
  using MatchingBase<PointSet>::Load ;
  using MatchingBase<PointSet>::LoadTemplate ;


  PointSetMatching(){init();}
  PointSetMatching(param_matching &par) {init(par);  }

  PointSetMatching(char *file, int argc, char** argv) {init(file, argc, argv) ;  }
  PointSetMatching(char *file, int k){ init(file, k);  }

  void init() {epsMin = 1e-7; epsSmall = 1e-10;}
  void init(param_matching &par) {
    init() ;
    param.copy(par) ;
    cout << "load file" << endl ;
    Load() ;
  }

  void init(char *file, int argc, char** argv) {
    init() ;
    param.read(file) ;
    param.read(argc, argv) ;
    cout << "load file" << endl ;
    Load() ;
  }

  void init(char *file, int k){
    init() ;
    param.read(file) ;
    if (k==0)
      Load() ;
    else
     LoadTemplate() ;
  }


  template <class AE>
  void initializeAffineReg(AE & enerAff) {
    if (param.type_group != param_matching::ID) {
      std::vector<_real> sz, x0 ;
      Point Mx, Mn ;
      Mx.copy(Template[0]) ;
      Mn.copy(Template[0]) ;
      for(unsigned int i=0; i<Template.size(); i++) 
	for (unsigned int k=0; k<Template[i].size(); k++) {
	  if (Template[i][k] < Mn[k])
	    Mn[k] = Template[i][k] ;
	  if (Template[i][k] > Mx[k])
	    Mx[k] = Template[i][k] ;
	}
      for(unsigned int i=0; i<Target.size(); i++) 
	for (unsigned int k=0; k<Target[i].size(); k++) {
	  if (Target[i][k] < Mn[k])
	    Mn[k] = Target[i][k] ;
	  if (Target[i][k] > Mx[k])
	    Mx[k] = Target[i][k] ;
	}
      int N = param.dim.size() ;
      x0.resize(N) ;
      sz.resize(N) ;
      for (int i=0; i<N; i++) {
	x0[i] = (Mx[i] + Mn[i])/2.0 ;
	sz[i] = (Mx[i] - Mn[i]+1) ;
      }
      
      enerAff.init(Template, Target, sz, x0) ;
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


  void finalizeAffineTransform(Matrix &gamma) {
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
      PointSet tmp ;
      for (int i=0; i<N; i++)
	for (int j=0; j<=N; j++)
	  mat[i][j] = affTrans(i, j) ;
      ::affineInterp(Target, tmp, mat) ;
      Target.copy(tmp) ;
    }
    else {
      Matrix  invAT ;
      invAT.inverse(AT[T]) ;
      affTrans.copy(invAT) ;
      for (int i=0; i<N; i++)
	for (int j=0; j<=N; j++)
	  mat[i][j] = affTrans(i, j) ;
      PointSet tmp ;
      ::affineInterp(Template, tmp, mat) ;
      Template.copy(tmp) ;
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
  /*
  _real affineMatch(Matrix &gamma0)  ;
  void affineGradient(Vector &Tp, Vector &Tg, Matrix &grad) ;
  void affineReg() ;
  _real enerAff(const Vector &Tp, const Vector &Tg) const ;
  _real enerAff(const Vector &Tp, const Vector &Tg, const _real TpNorm, const _real TgNorm) const ;
  _real integrateMatrix(const Matrix &gm, Matrix &res) const ;
  _real gradientStepAff(Vector &Tp, Vector &Tg, _real &step) ;
  */

  void get_template(PointSet &tmp) {tmp.get_points(param.fileTemp, param.ndim);}
  void get_binaryTemplate(PointSet &tmp) {tmp.read(param.fileTemp);}
  void get_target(PointSet &tmp) {tmp.get_points(param.fileTarg, param.ndim);}
  void get_binaryTarget(PointSet &tmp) {tmp.read(param.fileTarg);}

  //  void affineInterpolation(deformableImage &src, deformableImage &dest){ affineInterp(src, dest, param.affMat) ;}
 
  /*
  void init_fft()
  {
    Vector kern ;
    unsigned int TYPEINIT = FFTW_MEASURE ;
    
    Diffeomorphisms::init_fft(imageDim, param.kernel_type, param.sigmaKernel, param.sizeKernel, param.orderKernel, _kern, paddedMap, &_nM, & _inMap, &_outMap, 
			      & _inMapc, &_outMapc, &_fMapKernel, _ptoMap, _pfromMap, TYPEINIT) ;
    FFT_INITIALIZED = true ;
  }
  */

  ~PointSetMatching(){;}
  protected:
  _real epsMin ;
  _real epsSmall ;
};


#endif
