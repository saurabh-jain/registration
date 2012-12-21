/**
   velocity.h
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
#ifndef _VELOCITY_
#define _VELOCITY_
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
class Velocity: public ImageEvolution
{
public:
  Velocity(param_matching &par) {init(par);}
  Velocity(char *file, int argc, char** argv) {init(file, argc, argv);}
  Velocity(char *file, int k) {init(file, k);}

  // Template, initial image momentum, 
  TimeVectorMap Lv0 ;
  int _T ;
  double _epsBound ;

  //int nbFreeVar ;

  Velocity(){} ;
  ~Velocity(){} ;

  void affineReg(){
    //    cout << "in Affine Reg" << endl ;
    ImageAffineEnergy enerAff ;
    Template.domain().putm(enerAff.MIN) ;
    initializeAffineReg(enerAff) ;
    ImageMatching::affineReg(enerAff) ;
    finalizeAffineTransform(gamma, enerAff.MIN) ;    
  }

  class scalProd {
  public:
    scalProd(){};
    double operator()(TimeVectorMap &Z, TimeVectorMap &ZZ){
      double res=0;
      for (unsigned int t=0; t<Z.size(); t++)
	res += Z[t].scalProd(ZZ[t]);
      return res ;
    }
  } ;

  /*
  class scalProd2 {
  public:
    ~scalProd2(){} ;
    scalProd2(Velocity *sh){_sh = sh;}
    double operator(TimeVectorMap &Z, TimeVectorMap &ZZ){
      double res=0; 
      VectorMap foo; 
      for (int k=0; k<(int) Z.size(); k++) {
	kernel(Z[k], foo); 
	res += foo.scalProd(ZZ[k]);
      }
      return res ;
    }
    private:
      Velocity *_sh ;
  } ;
  */

 
  class optimVelocity: public optimFunBase<TimeVectorMap, TimeVectorMap>{
  public:
    virtual ~optimVelocity(){};
    optimVelocity(Velocity *sh) {
      _sh = sh ;
    }

    double epsBound() {
      return _sh->_epsBound ;
    }

    double computeGrad(TimeVectorMap &Lv, TimeVectorMap& grad){
      double ng2 ;
      double dt = 1.0/_sh->_T ;
      VectorMap foo, foo2, id, vt ; 
      VectorMap pp ;
      TimeVectorMap psi ;
      double c0, c1, epsTmp ;
      std::vector<VectorMap > Dpsi ;
      
      grad.resize(_sh->_T, _sh->Template.domain()) ;
      psi.resize(_sh->_T+1, _sh->Template.domain()) ;
      _sh->initFlow(Lv.d) ;
      psi[0].copy(_sh->_psi) ;
      for (unsigned int t=0; t<Lv.size(); t++) {
	_sh->updateFlow(Lv[t], dt) ;
	psi[t+1].copy(_sh->_psi) ;
      }

      LDDMMgradientInPsi(_sh->_psi, _sh->Template, _sh->Target, pp, _sh->param.spaceRes) ;
      pp /= (_sh->param.sigma)*(_sh->param.sigma) ;
      //pp.copy(b) ;

      id.idMesh(Lv.d) ;
      ng2 = 0 ;
      _sh->_epsBound = _sh->param.epsMax ;
      for (int t = _sh->_T-1; t>=0; t--) {
	_sh->kernel(Lv[t], vt) ;
	c0 = vt.maxNorm() ;
	addProduct(id, -dt, vt, foo) ;
	grad[t].copy(Lv[t]) ;
	foo.multilinInterpGradient(psi[t], Dpsi) ;
	matrixTProduct(Dpsi, pp, foo2) ; 
	grad[t] -= foo2 ;
	for (unsigned int i=0; i<pp.size(); i++)
	  foo.multilinInterpDual(pp[i], pp[i]) ;
	_sh->kernel(grad[t], foo) ;
	ng2 += grad[t].scalProd(foo) ;
	c1 = foo.maxNorm() ;
	epsTmp = (1+c0)/(1+c1) ;
	if (epsTmp < _sh->_epsBound)
	  _sh->_epsBound = epsTmp ; 
      }
      return ng2 ;
    }

    double computeGradOLD(TimeVectorMap &Lv, TimeVectorMap& grad){
      double ng2 ;
      double dt = 1.0/_sh->_T ;
      VectorMap b, foo, chi, id ; 
      
      grad.resize(_sh->_T, _sh->Template.domain()) ;
      LDDMMgradient(_sh->_psi, _sh->Template, _sh->Target, b, _sh->param.spaceRes) ;
      // _sh->initFlow(Lv.d) ;
      b /= (_sh->param.sigma)*(_sh->param.sigma) ;
      //  cout << "norm of b " << b.norm2() << endl; 
      grad[Lv.size()-1].copy(b) ;
      id.idMesh(Lv.d) ;
      chi.copy(id) ;
      for(int t=Lv.size()-2; t>=0; t--) {
	_sh->kernel(Lv[t], foo) ;
	foo *= dt ;
	foo += id ;
	foo.multilinInterp(chi, chi) ;
	_sh->big_adjointStar(b, chi, grad[t]) ;
      }

      ng2 = 0 ;
      for(unsigned int t=0; t<Lv.size(); t++) {
	grad[t] += Lv[t] ;
	_sh->kernel(grad[t], foo) ;
	ng2 += grad[t].scalProd(foo) ;
      }
      return ng2 ;
    }

    double objectiveFun(TimeVectorMap &Lv) {
      double ener ;
      double dt = 1.0/_sh->_T ;
      //  cout << "norm of Lv " << Lv.norm2() << endl ;
      _sh->initFlow(Lv.d) ;
      ener = 0 ;
      for(unsigned int t=0; t<Lv.size(); t++)
	ener += _sh->updateFlow(Lv[t], dt) ;
      //  cout << "def ener " << ener << endl ;
      ener += LDDMMenergy(_sh->_psi, _sh->Template, _sh->Target)/ ((_sh->param.sigma)*(_sh->param.sigma));
      return ener/Lv.d.length ;
    }


    double endOfIteration(TimeVectorMap &Lv) {
      cout << "obj fun: " << objectiveFun(Lv) << endl ;
      _sh->Lv0.copy(Lv) ;
      _sh->Print() ;
      return -1 ;
    }

    void startOfProcedure(TimeVectorMap &Lv) {
      Vector foo ;
      int nbFreeVar=0 ;
      _sh->Template.gradient().norm(foo) ;
      for (unsigned int k=0; k<foo.d.length;k++)
	if (foo[k] > 0.01)
	  nbFreeVar++ ;
      _sh->kernelNormalization = nbFreeVar ; 
      cout << "Number of momentum variables: " << nbFreeVar << " out of " << foo.d.length << endl ;
    }

    void endOfProcedure(TimeVectorMap &Lv) {
      _sh->Lv0.copy(Lv) ;
      _sh->Print() ;
    }

  private:
    Velocity *_sh ;
  };


  void matching(TimeVectorMap & Lw) {
    TimeVectorMap Lv ;

    _epsBound = param.epsMax ;
    conjGrad<TimeVectorMap, TimeVectorMap, scalProd, optimVelocity> cg ;
    scalProd scp; //scp(this) ;
    
    optimVelocity opt(this) ; //opt( Template,  Target,  param.sigma) ;
    cg(opt, scp, Lw, Lv, param.nb_iter, .01, param.epsMax, param.minVarEn, param.gs, param.verb) ;
    Lv0.copy(Lv) ;
  }

  void matching() {    _T = param.time_disc ;    Lv0.zeros(_T, Template.domain()); matching(Lv0);}


  void initialPrint(char * path) ;
  void initialPrint() {  initialPrint(param.outDir) ;}
  void Print(char * path) ;
  void Print() {  Print(param.outDir) ;}

};



#endif
