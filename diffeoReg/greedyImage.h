/**
   greedyImage.h
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

#ifndef _GREEDYIMAGE_
#define _GREEDYIMAGE_
#include "ImageEvolution.h"
#include "ImageMatchingAffine.h"
#include "optimF.h"


/** 
    Class implementing greedy image matching
*/
/** class implementing symmetric LDDMM with respect to the initial momentum
 */
class GreedyImage: public ImageEvolution
{
 public:

  GreedyImage(param_matching &par) {init(par);}
  GreedyImage(char *file, int argc, char** argv) {init(file, argc, argv);}
  GreedyImage(char *file, int k) {init(file, k);}

  GreedyImage(){} ;


  void affineReg(){
    ImageAffineEnergy enerAff ;
    Template.domain().putm(enerAff.MIN) ;
    initializeAffineReg(enerAff) ;
    ImageMatching::affineReg(enerAff) ;
    finalizeAffineTransform(gamma, enerAff.MIN) ;    
  }

  class optimGreedyImage: public optimFunBase<VectorMap, VectorMap>{
  public:
    VectorMap _psiTmp ;
    double totalEnergy ;
    double totalTime ;
    double incTime ;
    double defTotalEnergy ;
    double incEnergy ;
    double oldTotalEnergy ;
    virtual ~optimGreedyImage(){};
    optimGreedyImage(GreedyImage *sh) {
      _sh = sh ;
    }

    double computeGrad(VectorMap &phi, VectorMap& grad){
      VectorMap b, vI ;
      Vector I, DI ;
      (_sh->_psi).multilinInterp(_sh->Template.img(), I) ;
      DI.copy(I) ;
      DI -= _sh->Target.img() ;
      gradient(I, b, _sh->param.spaceRes) ;
      b *= DI ;
      b*= -1 ;
      _sh->kernel(b, grad) ;
      return grad.norm2() ;
    }

    double update(VectorMap &phi, VectorMap& grad, double eps, VectorMap &res) {
      VectorMap foo, foo2, id ;
      cout << "update;  epsilon = " << eps << endl ;
      id.idMesh(phi.d) ;
      foo.copy(grad) ;
      foo *= -eps ;
      foo += id ;
      phi.multilinInterp(foo, res) ;
      foo2.copy(grad) ;
      foo2 *= eps ;
      foo2 += id ;
      foo2. multilinInterp(_sh->_psi, foo) ; 
      incEnergy = eps * _sh->kernelNorm(grad) / phi.d.length ;
      incTime = eps ;

      if (res.inverseMap(foo, _psiTmp, _sh->param.spaceRes) ) {
	double ener = objectiveFun(res) ; 
	return ener;
      }
      else {
	cout << "Negative Jacobian" << endl ;
	return 1e20 ;
      }
    }

    double objectiveFun(VectorMap &phi) {
      Vector DI ;
      _psiTmp.multilinInterp(_sh->Template.img(), DI) ;
      DI -= _sh->Target.img() ;
      return DI.norm2() /  DI.d.length ;
    }

    double endOfIteration(VectorMap &phi) {
      defTotalEnergy += incEnergy ;
      totalTime += incTime ;
      totalEnergy = defTotalEnergy/totalTime + objectiveFun(phi) /(_sh->param.sigma * _sh->param.sigma) ;
      cout << "total energy: " << totalEnergy << " (old: " << oldTotalEnergy <<")" << endl ;
      _sh->_phi.copy(phi) ;
      _sh->_psi.copy(_psiTmp) ;
      //      phi.inverseMap(_sh->_psi, _sh->param.spaceRes) ;
      _sh->Print() ;
      return -1 ;
    }

    void startOfProcedure(VectorMap &phi0) {
      phi0.inverseMap(_psiTmp, _sh->param.spaceRes) ;
      _sh->_psi.copy(_psiTmp) ;
      defTotalEnergy = 0 ;
      totalTime = 0 ;
      oldTotalEnergy = objectiveFun(phi0) /(_sh->param.sigma * _sh->param.sigma) ;
    }

    bool stopProcedure() {
      if (totalEnergy > oldTotalEnergy)
	return true ;
      else {
	oldTotalEnergy = totalEnergy ;
	return false ;
      }
    }


    void endOfProcedure(VectorMap &phi) {
      _sh->_phi.copy(phi) ;
      _sh->_psi.copy(_psiTmp) ;
      //      phi.inverseMap(_sh->_psi, _sh->_psi, _sh->param.spaceRes) ;
      _sh->Print() ;
    }
  private:
    GreedyImage *_sh ;
  };
    
  class scalProd {
  public:
    scalProd(){};
    scalProd(GreedyImage &sh) { _sh = &sh ;}
    double operator()(VectorMap &Z, VectorMap &ZZ){
      return Z.scalProd(ZZ);
    }
  private:
    GreedyImage *_sh ;
  } ;

  void initialPrint(char* path) {
    if (!param.printFiles)
      return ;
    char file[256] ;

    sprintf(file, "%s/template", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    Template.img().write_image(file) ;
    sprintf(file, "%s/binaryTemplate", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    Template.img().write(file) ;
    sprintf(file, "%s/target", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    Target.img().write_image(file) ;
    sprintf(file, "%s/binaryTarget", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    Target.img().write(file) ;

    sprintf(file, "%s/kernel", path) ;
    _kern.write_imagesc(file) ; 
  }



  void Print(char* path) {
    if (!param.printFiles)
      return ;
    char file[256] ;
    Vector I1, I2 ;

    VectorMap dI0, LvI, vtry2, Lvtry2 ;
    _psi.multilinInterp(Template.img(), I1) ;
    sprintf(file, "%s/deformedTemplate", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    I1.write_image(file) ;

    _phi.multilinInterp(Target.img(), I1) ;
    sprintf(file, "%s/deformedTarget", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    I1.write_image(file) ;

    Vector dphi ;

    _phi.logJacobian(dphi, param.spaceRes) ;
    sprintf(file, "%s/jacobian", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    dphi.write(file) ;
    dphi.writeZeroCentered(file) ;

    _phi.displacement(dphi) ;
    sprintf(file, "%s/absoluteDisplacement", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    dphi.write_imagesc(file) ;

    sprintf(file, "%s/template2targetMap", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    _phi.write(file) ;
    
    sprintf(file, "%s/target2templateMap", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    _psi.write(file) ;
  }

  void gradientImageMatching(VectorMap & phi0) {
    VectorMap phi ;
    
    conjGrad<VectorMap, VectorMap, scalProd, optimGreedyImage> cg ;
    scalProd scp(*this) ;
    
    optimGreedyImage opt(this) ; //opt( Template,  Target,  param.sigma) ;
    cg(opt, scp, phi0, phi, param.nb_iter, 0.001, param.epsMax, param.minVarEn, param.gs, param.verb) ;
  }

  void gradientImageMatching() {VectorMap phi; phi.idMesh(Template.domain()); gradientImageMatching(phi);}



  ~GreedyImage(){}

 
  void initialPrint() {  initialPrint(param.outDir) ;}
  void Print() {  Print(param.outDir) ;}

} ;




#endif
