/**
   shooting.h
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
#ifndef _SHOOTING_
#define _SHOOTING_
#include "ImageEvolution.h"
#include "ImageMatchingAffine.h"
#include "optimF.h"

_real LDDMMenergy(const VectorMap& psi, 
		  const deformableImage& I0, const deformableImage& I1) ;
void LDDMMgradient(const VectorMap& phi, const deformableImage& I0, const deformableImage& I1, VectorMap &b, vector<_real>&) ;
void LDDMMgradientInverse(const VectorMap& phi, const deformableImage& I0, const deformableImage& I1, VectorMap &b, vector<_real> &resol) ;
void LDDMMgradientInPsi(const VectorMap& phi, const deformableImage& I0, const deformableImage& I1, VectorMap &b, vector<_real>&) ;
_real SYMenergy(const VectorMap& psi, const deformableImage& I0, const deformableImage& I1, vector<_real>&) ;
void SYMgradient(const VectorMap& phi, const deformableImage& I0, const deformableImage& I1, VectorMap &b, vector<_real>&) ;
void SYMgradient(const VectorMap &phi, const VectorMap& psi, const deformableImage& I0, const deformableImage& I1, VectorMap &b, vector<_real> &resol) ;
_real SEGenergy(const VectorMap& psi, const deformableImage& I0, const Vector &ps) ;
void SEGgradientInPsi(const VectorMap& psi, const deformableImage& I0, const Vector &ps, VectorMap &b, vector<_real> &resol) ;



/** 
    Class implementing LDDMM with respect to the initial momentum
*/
class BasicShooting: public ImageEvolution
{
public:
  BasicShooting(param_matching &par) {init(par);}
  BasicShooting(char *file, int argc, char** argv) {init(file, argc, argv);}
  BasicShooting(char *file, int k) {init(file, k);}
  double _epsBound ;

  typedef deformableImage::Tangent Tangent ;
  // Template, initial image momentum, 
  Tangent Z0;
  VectorMap Lv0 ;
  //int nbFreeVar ;

  BasicShooting(){} ;

  virtual void affineReg(){
    ImageAffineEnergy enerAff ;
    Template.domain().putm(enerAff.MIN) ;
    initializeAffineReg(enerAff) ;
    ImageMatching::affineReg(enerAff) ;
    finalizeAffineTransform(gamma, enerAff.MIN) ;    
  }

  class scalProd {
  public:
    scalProd(){};
    double operator()(Tangent &Z, Tangent &ZZ){
      return Z.sumProd(ZZ);
    }
  } ;



  ~BasicShooting(){}

 
};


class Shooting: public BasicShooting
{
public:
  Shooting(param_matching &par) {init(par);}
  Shooting(char *file, int argc, char** argv) {init(file, argc, argv);}
  Shooting(char *file, int k) {init(file, k);}
  Shooting(){} ;

  class optimShooting: public optimFunBase<Tangent, Tangent>{
  public:
    virtual ~optimShooting(){};
    optimShooting(Shooting *sh) {
      _sh = sh ;
    }

    double epsBound() {
      return _sh->_epsBound ;
    }

    double computeGrad(Tangent &Z, Tangent& gradZ){
      VectorMap grad, b, vI, LvI ;
      Vector foog ;
      double ng, c0, c1 ;
      LDDMMgradientInPsi(_sh->_psi, _sh->Template, _sh->Target, b, _sh->param.spaceRes) ;
      b /= (_sh->param.sigma)*(_sh->param.sigma) ;
      _sh->Template.getMomentum(Z, LvI) ;
      _sh->kernel(LvI, vI) ;
      c0 = vI.maxNorm() ;
      //      vI /= _sh->nbFreeVar ;
      _sh->DualVarGeodesicDiffeoEvolutionDiscrete(LvI, b, grad) ;
      cout << "grad: " << b.norm2() << " " << grad.norm2() << endl ;
      grad += vI ;
      _sh->Template.normal().scalProd(grad, foog) ;
      foog *=-1 ;
      //    cout << "projection" << endl;
      //_sh->imageTangentProjection(_sh->Template, foog, gradZ) ;
      gradZ.copy(foog) ;
      ng = gradZ.norm2() ;
      _sh->Template.getMomentum(gradZ, LvI) ;
      _sh->kernel(LvI, vI) ;
      c1 = vI.maxNorm() ;
      _sh->_epsBound = (1+c0)/(1+c1) ;
      return ng ;
    }

    double objectiveFun(Tangent &Z) {
      VectorMap LvI, vI ;
      //      cout << "obj: " << Z.maxAbs() << " " << _sh->Template.img().maxAbs() << endl ;
      _sh->Template.getMomentum(Z, LvI) ;
      _sh->GeodesicDiffeoEvolution(LvI) ;
      _sh->kernel(LvI, vI) ;
      return (vI.scalProd(LvI) + LDDMMenergy(_sh->_psi, _sh->Template, _sh->Target)/ ((_sh->param.sigma)*(_sh->param.sigma)))/Z.d.length ;
    }

    double endOfIteration(Tangent &Z) {
      cout << "obj fun: " << objectiveFun(Z) << endl ;
      _sh->Z0.copy(Z) ;
      _sh->Print() ;
      return -1 ;
    }

    void startOfProcedure(Tangent &Z) {
      Vector foo ;
      int nbFreeVar=0 ;
      _sh->Template.gradient().norm(foo) ;
      for (unsigned int k=0; k<foo.d.length;k++)
	if (foo[k] > 0.01)
	  nbFreeVar++ ;
      _sh->kernelNormalization = nbFreeVar ; 
      cout << "Number of momentum variables: " << nbFreeVar << " out of " << foo.d.length << endl ;
    }

    void endOfProcedure(Tangent &Z) {
      _sh->Z0.copy(Z) ;
      _sh->Print() ;
    }

  private:
    Shooting *_sh ;
  };

  class scalProd2 {
  public:
    ~scalProd2(){} ;
    scalProd2(Shooting *sh){_sh = sh;}
    double operator()(Tangent &Z, Tangent &ZZ){
      VectorMap foo, foo2 ;
      foo.copy(_sh->Template.normal()) ;
      foo *= Z ;
      _sh->kernel(foo, foo2) ;
      foo.copy(_sh->Template.normal()) ;
      foo *= ZZ ;
      return foo.scalProd(foo2) + 1e-4*Z.sumProd(ZZ);;
    }
    private:
      Shooting *_sh ;
  } ;

  void gradientImageMatching(Tangent & ZZ) {
    Tangent Z ;

    _epsBound = param.epsMax ;
    conjGrad<Tangent, Tangent, scalProd, optimShooting> cg ;
    scalProd scp; //scp(this) ;
    
    optimShooting opt(this) ; //opt( Template,  Target,  param.sigma) ;
    cg(opt, scp, ZZ, Z, param.nb_iter, .01, param.epsMax, param.minVarEn, param.gs, param.verb) ;
    Z0.copy(Z) ;
  }

  void gradientImageMatching() {Z0.al(Template.domain()); Z0.zero(); gradientImageMatching(Z0);}


  ~Shooting(){}
  void initialPrint(char * path) ;
  void initialPrint() {  initialPrint(param.outDir) ;}
  void Print(char * path) ;
  void Print() {  Print(param.outDir) ;}

};


/** class implementing symmetric LDDMM with respect to the initial momentum
 */
class ShootingSYM: public ImageEvolution
{
 public:
/*   using ImageMatching::Template ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::kernel ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::savedTemplate ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::Target ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::savedTarget ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::param ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::savedParam; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::mask ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::init_fft ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::makeMask ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::inverseKernel ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::kernelNorm ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::_psi ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::_phi ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::big_adjointStar; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::GeodesicDiffeoEvolution ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::VarGeodesicDiffeoEvolution ; */
/*   using MatchingBase<deformableImage, ImageAffineEnergy>::alpha; */

  ShootingSYM(param_matching &par) {init(par);}
  ShootingSYM(char *file, int argc, char** argv) {init(file, argc, argv);}
  ShootingSYM(char *file, int k) {init(file, k);}

  VectorMap Lv0 ;

  ShootingSYM(){} ;


  void affineReg(){
    ImageAffineEnergy enerAff ;
    Template.domain().putm(enerAff.MIN) ;
    initializeAffineReg(enerAff) ;
    ImageMatching::affineReg(enerAff) ;
    finalizeAffineTransform(gamma, enerAff.MIN) ;    
  }

  class optimShootingSYM: public optimFunBase<VectorMap, VectorMap>{
  public:
    virtual ~optimShootingSYM(){};
    optimShootingSYM(ShootingSYM *sh) {
      _sh = sh ;
    }

    double computeGrad(VectorMap &LvI, VectorMap& grad){
      VectorMap b, vI, foo ;
      double ng ;
      SYMgradient(_sh->_phi, _sh->_psi, _sh->Template, _sh->Target, b, _sh->param.spaceRes) ;
      b /= (_sh->param.sigma)*(_sh->param.sigma) ;
      _sh->kernel(LvI, vI) ;
      _sh->DualVarGeodesicDiffeoEvolutionDiscrete(LvI, b, foo) ;
      _sh->inverseKernel(foo, grad) ;
      grad += LvI ;
      ng = _sh->kernelNorm(grad) ;
      return ng ;
    }

    double objectiveFun(VectorMap &Lv) {
      VectorMap vI ;
      _sh->GeodesicDiffeoEvolution(Lv) ;
      _sh->kernel(Lv, vI) ;
      return (vI.scalProd(Lv) + SYMenergy(_sh->_psi, _sh->Template, _sh->Target, _sh->param.spaceRes)
	      / ((_sh->param.sigma)*(_sh->param.sigma)))/Lv.d.length ;
    }

    double endOfIteration(VectorMap &Lv) {
      _sh->Lv0.copy(Lv) ;
      _sh->Print() ;
      return -1 ;
    }

    void endOfProcedure(VectorMap &Lv) {
      _sh->Lv0.copy(Lv) ;
      _sh->Print() ;
    }
  private:
    ShootingSYM *_sh ;
  };
    
  class scalProd {
  public:
    scalProd(){};
    scalProd(ShootingSYM &sh) { _sh = &sh ;}
    double operator()(VectorMap &Z, VectorMap &ZZ){
      VectorMap foo; 
      _sh->kernel(Z, foo); 
      return foo.scalProd(ZZ);
    }
  private:
    ShootingSYM *_sh ;
  } ;

  void gradientImageMatching(VectorMap & Lv1) {
    VectorMap Lv ;

    conjGrad<VectorMap, VectorMap, scalProd, optimShootingSYM> cg ;
    scalProd scp(*this) ;
    
    optimShootingSYM opt(this) ; //opt( Template,  Target,  param.sigma) ;
    cg(opt, scp, Lv1, Lv, param.nb_iter, 0.001, param.epsMax, param.minVarEn, param.gs, param.verb) ;
  }

  void gradientImageMatching() {Lv0.al(Template.domain()); Lv0.zero(); gradientImageMatching(Lv0);}



  ~ShootingSYM(){}

 
  void initialPrint(char * path) ;
  void initialPrint() {  initialPrint(param.outDir) ;}
  void Print(char * path) ;
  void Print() {  Print(param.outDir) ;}

} ;



/** class implementing symmetric LDDMM with volume cost
 */
class ShootingSEG: public ImageEvolution
{
public:

  ShootingSEG(param_matching &par) {init(par);defaultParam();}
  ShootingSEG(char *file, int argc, char** argv) {init(file, argc, argv);defaultParam();}
  ShootingSEG(char *file, int k) {init(file, k);defaultParam() ;}

  void defaultParam() {lambdaSEG1 = 1 ;lambdaSEG2 = 0;}
  double _epsBound ;


  typedef deformableImage::Tangent Tangent ;

  // Template, initial image momentum, 
  Tangent Z0;
  Vector preSeg ;
  VectorMap Lv0 ;

  //misc parameters
  double lambdaSEG1 ;
  double lambdaSEG2 ;

  ShootingSEG(){} ;

  template <class AE>
  void initializeAffineRegSeg(AE & enerAff) {
    if (param.type_group != param_matching::ID) {
      Vector Tp, Tg, Ps, tmp ;
      std::vector<_real> sz, x0 ;
      Ivector MIN ;
      Domain D ;
      cout << "preprocess" << endl ;
      getPreprocessedImagesAffine(Tp, Tg, D) ;
      preSeg.crop(D, Ps) ;
      D.putm(MIN) ;
      
      int N = param.dim.size() ;
      x0.resize(N) ;
      sz.resize(N) ;
      for (int i=0; i<N; i++) {
	x0[i] = (Tp.d.getM(i) + Tp.d.getm(i))/2.0 ;
	sz[i] = (Tp.d.getM(i) - Tp.d.getm(i)+1) ;
      }
      enerAff.init(Tp, Tg, Ps, sz, x0, MIN, param.lambda) ;
    }
  }

  void getAuxiliaryParameters(){
    if (param.foundAuxiliaryParam) {
      lambdaSEG1 = param.auxParam[0] ;
      lambdaSEG2 = param.auxParam[1] ;
    }
  }

  void loadAuxiliaryFiles(){
    // preSeg = lambdaSEG1 - lambdaSEG2* (isForbidden)
    if (param.foundAuxiliaryFile) {
    // a binary file marking forbidden regions
      preSeg.get_image(param.auxFiles[0].c_str(), param.ndim) ;
      for (unsigned int k=0; k<preSeg.size(); k++) 
	if (preSeg[k] > .5)
	  preSeg[k] = lambdaSEG1 + lambdaSEG2 ;
	else 
	  preSeg[k] = lambdaSEG1 ;
    }
    else {
      int nb = 0 ;
      preSeg.copy(Target.img()) ;
      for (unsigned int k=0; k<Target.img().size(); k++) 
	if (Target.img()[k] > 1.5){
	  preSeg[k] = lambdaSEG1 + lambdaSEG2 ; 
	  Target.img()[k] = 0 ;
	  nb ++ ;
	}
	else 
	  preSeg[k] = lambdaSEG1 ;
      //      cout << "found " << nb << " bone voxels" << endl ;
    }
  }

  void affineReg(){
    //    cout << "starting affine" << endl ;
    ImageSegmentationEnergy enerAff ;
    Template.domain().putm(enerAff.MIN) ;
    //    cout << "init affine" << endl ;
    initializeAffineRegSeg(enerAff) ;
    //    cout << "running affine" << endl ;
    ImageMatching::affineReg(enerAff) ;
    //    cout << "finalizing affine" << endl ;
    finalizeAffineTransform(gamma, enerAff.MIN) ;    
    if (!param.applyAffineToTemplate) {
      _real mat[DIM_MAX][DIM_MAX+1] ;
      Vector tmp ;
      for (int i=0; i<param.ndim; i++)
	for (int j=0; j<=param.ndim; j++)
	  mat[i][j] = affTrans(i, j) ;
      ::affineInterp(preSeg, tmp, mat) ;
      preSeg.copy(tmp) ;
    }
    //    cout << "done affine" << endl ;
  }

  class optimShootingSEG: public optimFunBase<Tangent, Tangent>{
  public:
    //    using PointSetEvolution<KERNEL>::geodesicPointSetEvolution ;
    virtual ~optimShootingSEG(){};
    optimShootingSEG(ShootingSEG *sh) {
      _sh = sh ;
      //      sigma = lm->param.sigma ;
      //      scp.init(lm->Template) ;
    }

    double epsBound() {
      return _sh->_epsBound ;
    }

    double computeGrad(Tangent &Z, Tangent& gradZ){
      VectorMap grad, b, b2, vI, LvI ;
      Vector foog ;
      double ng, c0, c1 ;
      LDDMMgradientInPsi(_sh->_psi, _sh->Template, _sh->Target, b, _sh->param.spaceRes) ;
      b *= _sh->param.lambda ;
      SEGgradientInPsi(_sh->_psi, _sh->Template, _sh->preSeg, b2, _sh->param.spaceRes) ;
      //      b2 *= _sh->lambdaSEG1 ;
      b += b2 ;
      _sh->Template.getMomentum(Z, LvI) ;
      _sh->kernel(LvI, vI) ;
      _sh->DualVarGeodesicDiffeoEvolutionDiscrete(LvI, b, grad) ;
      c0 = vI.maxNorm() ;
      grad += vI ;
      _sh->Template.normal().scalProd(grad, foog) ;
      foog *=-1 ;
      //    cout << "projection" << endl;
      gradZ.copy(foog) ;
      ng = gradZ.norm2() ;
      _sh->Template.getMomentum(gradZ, LvI) ;
      _sh->kernel(LvI, vI) ;
      c1 = vI.maxNorm() ;
      _sh->_epsBound = (1+c0)/(1+c1) ;
      //      _sh->imageTangentProjection(_sh->Template, foog, gradZ) ;
      //      ng = gradZ.norm2() ;
      return ng ;
    }

    double objectiveFun(Tangent &Z) {
      VectorMap LvI, vI ;
      _sh->Template.getMomentum(Z, LvI) ;
      _sh->GeodesicDiffeoEvolution(LvI) ;
      _sh->kernel(LvI, vI) ;
      double a = _sh->param.lambda ;
      //      double b = _sh->lambdaSEG1 ;
      return (vI.scalProd(LvI) + a*LDDMMenergy(_sh->_psi, _sh->Template, _sh->Target) 
	      + SEGenergy(_sh->_psi, _sh->Template, _sh->preSeg))/Z.d.length ;
    }

    double endOfIteration(Tangent &Z) {
      _sh->Z0.copy(Z) ;
      _sh->Print() ;
      return -1 ;
    }

    void startOfProcedure(Tangent &Z) {
      Vector foo ;
      int nbFreeVar=0 ;
      _sh->Template.gradient().norm(foo) ;
      for (unsigned int k=0; k<foo.d.length;k++)
      	if (foo[k] > 0.01)
      	  nbFreeVar++ ;
      _sh->kernelNormalization = nbFreeVar ; 
	   //      _sh->kernelNormalization = 1.0 ; 
      cout << "Number of momentum variables: " << nbFreeVar << " out of " << foo.d.length << endl ;
    }

    void endOfProcedure(Tangent &Z) {
      _sh->Z0.copy(Z) ;
      _sh->Print() ;
    }

    //    scalProdLandmarks<KERNEL> scp ;
  private:
    ShootingSEG *_sh ;
  };

  class scalProd {
  public:
    scalProd(){};
    double operator()(Tangent &Z, Tangent &ZZ){
      return Z.sumProd(ZZ);
    }
  } ;


  void gradientImageMatching(Tangent & ZZ) {
    Tangent Z ;

    _epsBound = param.epsMax ;
    conjGrad<Tangent, Tangent, scalProd, optimShootingSEG> cg ;
    scalProd scp ;
    
    optimShootingSEG opt(this) ; //opt( Template,  Target,  param.sigma) ;
    cg(opt, scp, ZZ, Z, param.nb_iter, 0.001, param.epsMax, param.minVarEn, param.gs, param.verb) ;
    Z0.copy(Z) ;
  }

  void gradientImageMatching() {Z0.al(Template.domain()); Z0.zero(); gradientImageMatching(Z0);}

  ~ShootingSEG(){}

 
  void initialPrint(char * path) ;
  void initialPrint() {  initialPrint(param.outDir) ;}
  void Print(char * path) ;
  void Print() {  Print(param.outDir) ;}
};


#endif
