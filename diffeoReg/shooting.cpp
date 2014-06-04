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

#include "shooting.h"

_real LDDMMenergy(const VectorMap& psi, const deformableImage& I0, const deformableImage& I1)
{
  Vector I ;

  psi.multilinInterp(I0.img(), I) ;
  return I.dist2(I1.img()) ;
}

/**
Computes the Eulerian differential of the LDDMM data attachment term considered as a function of the inverse deformation
*/
void LDDMMgradientInverse(const VectorMap& psi, const deformableImage& I0, const deformableImage& I1, VectorMap &b, vector<_real> &resol)
{
  Vector DI ; 
  Vector I, jj, foo ; 
  VectorMap foom ;
  std::vector<VectorMap> Dpsi ;

  psi.multilinInterp(I0.img(), I) ;
  DI.copy(I1.img()) ;
  DI -= I ;
  gradient(I, b, resol) ;
  b *= DI ;
}

void LDDMMgradientInPsi(const VectorMap& psi, const deformableImage& I0, const deformableImage& I1, VectorMap &b, vector<_real> &resol)
{
  Vector DI ; 
  Vector I, jj ;
  //VectorMap foo ;

  psi.multilinInterp(I0.img(), DI) ;
  //cout << "interp" << endl ;
  //  DI.copy(I) ;
  DI -= I1.img() ;

  //gradient(I0.img(), foo, resol) ;
  //psi.multilinInterp(I0.gradient(), b) ;
    psi.multilinInterpGradient(I0.img(), b) ;
  //  b.copy(I.gradient()) ;
  b *= DI ;
}




/**
Computes the Eulerian differential of the LDDMM data attachment term considered as a function of the direct deformation
*/
void LDDMMgradient(const VectorMap& psi, const deformableImage& I0, const deformableImage& I1, VectorMap &b, vector<_real> &resol)
{
  Vector DI ; 
  Vector I, jj, foo0 ;
  VectorMap foo, foo1, foo2 ;
  //  double TINY = 1e-10 ;

  //  cout << I0.img().norm2() << endl ;
  //  cout << psi.norm2() << endl ; 
  psi.multilinInterp(I0.img(), I) ;
  //cout << "interp" << endl ;
  DI.copy(I) ;
  DI -= I1.img() ;
  DI *= -1 ;

  b.zeros(DI.d) ;

  
  psi.multilinInterp(I0.gradient(), foo) ;
  std::vector<VectorMap> Dpsi ;
  psi.differential(Dpsi, resol) ;
  for (unsigned int i=0; i<foo.size(); i++)
    for (unsigned int j=0; j<foo.size(); j++) {
      foo0.copy(foo[j]);
      foo0 *= Dpsi[j][i] ;
      b[i] +=  foo0 ;
    }
  /* 

  gradientPlus(I, foo2, resol) ;
  gradientMinus(I, foo1, resol) ;
  //  cout << foo2.norm2() << " " << foo1.norm2() << endl ;
  //  cout << DI.norm2() << endl ;

  for (unsigned int i=0; i<DI.d.length; i++)
    if (DI[i] > TINY)
      for (unsigned int j=0; j<b.size(); j++) {
	if (foo1[j][i] > TINY) {
	  if (foo2[j][i] < -TINY) 
	    b[j][i] = 0 ;
	  else if (foo2[j][i] > TINY)
 	    b[j][i] = DI[i] * foo2[j][i] ;
	}
	else if (foo2[j][i] < - TINY){
	  if (foo2[j][i] < -TINY) 
 	    b[j][i] = DI[i] * foo1[j][i] ;
	  else if (foo2[j][i] > TINY){
	    if (foo2[j][i] < foo1[j][i])
	      b[j][i] = DI[i] * foo2[j][i] ;
	    else
	      b[j][i] = DI[i] * foo1[j][i] ;
	    //	    b[j][i] = 0 ;
	  }
	}
      }
    else if (DI[i] < - TINY)
      for (unsigned int j=0; j<b.size(); j++) {
	if (foo1[j][i] < -TINY) {
	  if (foo2[j][i] > TINY) 
	    b[j][i] = 0 ;
	  else if (foo2[j][i] < - TINY)
 	    b[j][i] = DI[i] * foo1[j][i] ;
	}
	else if (foo1[j][i] > TINY) {
	  if (foo2[j][i] > TINY) 
 	    b[j][i] = DI[i] * foo2[j][i] ;
	  else if (foo2[j][i] < - TINY) {
	    if (foo2[j][i] > foo1[j][i])
	      b[j][i] = DI[i] * foo1[j][i] ;
	    else
	      b[j][i] = DI[i] * foo2[j][i] ;
	    //	    b[j][i] = 0 ;
	  }
	}
      }
  */
  //  gradient(I, b, resol) ;
  //  b.copy(I.gradient()) ;
  b *= DI ;
}


/*

double Shooting::computeGrad(Tangent &Z, Tangent& gradZ, void *par){
  VectorMap grad, b, vI, LvI ;
  Vector foog ;
  double ng ;
  LDDMMgradientInPsi(_psi, Template, Target, b, ((param_matching*) par)->spaceRes) ;
  b /= ((param_matching*) par)->sigma*((param_matching*) par)->sigma ;
  Template.getMomentum(Z, LvI) ;
  kernel(LvI, vI) ;
  DualVarGeodesicDiffeoEvolutionDiscrete(LvI, b, grad) ;
  grad += vI ;
  Template.normal().scalProd(grad, foog) ;
  foog *=-1 ;
  //    cout << "projection" << endl;
  imageTangentProjection(Template, foog, gradZ) ;
  ng = gradZ.norm2() ;
  return ng ;
}

double Shooting::objectiveFun(Tangent &Z, void * par) 
{
  VectorMap LvI, vI ;
  Template.getMomentum(Z, LvI) ;
  GeodesicDiffeoEvolution(LvI) ;
  kernel(LvI, vI) ;
  return (vI.scalProd(LvI) + LDDMMenergy(_psi, Template, Target)/ (((param_matching*) par)->sigma*((param_matching*) par)->sigma))/Z.d.length ;
}

double Shooting::endOfIteration(Tangent &Z, void * par)
{
  Z0.copy(Z) ;
  Print() ;
  return -1 ;
}

void Shooting::endOfProcedure(Tangent &Z)
{
  Z0.copy(Z) ;
  Print() ;
}
*/




void Shooting::Print(char* path)
{
  if (!param.printFiles) {
    //    cout << "not saving" << endl ;
    return ;
  }
  char file[256] ;
  Vector I1, I2 ;

  VectorMap dI0, LvI, vtry2, Lvtry2 ;
  if (param.gradInZ0) 
    Template.getMomentum(Z0, LvI) ;
  else
    LvI.copy(Lv0) ;
  //  GeodesicDiffeoEvolution(LvI, Lvtry2, 1) ;

  if (param.saveMovie) {
    TimeVectorMap phi, psi ;
    GeodesicDiffeoFlow(LvI, 10, phi, psi) ;
    for (unsigned int t=1; t<=10; t++) {
      psi[t].multilinInterp(Template.img(), I1) ;
      sprintf(file, "%s/movie%03d", path, t) ;
      I1.write_image(file) ;
    }
  }

  _psi.multilinInterp(Template.img(), I1) ;
  sprintf(file, "%s/deformedTemplate", path) ;
  if (param.verb > 1)
    cout << "writing " << file << endl ;
  I1.write_image(file) ;

  _phi.multilinInterp(Target.img(), I1) ;
  sprintf(file, "%s/deformedTarget", path) ;
  if (param.verb > 1)
    cout << "writing " << file << endl ;
  I1.write_image(file) ;

  sprintf(file, "%s/initialScalarMomentum", path) ;
  if (param.verb > 1)
    cout << "writing " << file << endl ;
  Z0.write(file) ;
    
  sprintf(file, "%s/scaledScalarMomentum", path) ;
  if (param.verb > 1)
    cout << "writing " << file << endl ;
  Z0.writeZeroCentered(file) ;

  sprintf(file, "%s/initialMomentum", path) ;
  if (param.verb > 1)
    cout << "writing " << file << endl ;
  LvI.write(file) ;




  Vector dphi ;

  _phi.logJacobian(dphi, param.spaceRes) ;
  sprintf(file, "%s/jacobian", path) ;
  if (param.verb > 1)
    cout << "writing " << file << endl ;
  dphi.write(file) ;
  dphi.writeZeroCentered(file) ;

  _phi.displacement(dphi) ;
  sprintf(file, "%s/absoluteDisplacement", path) ;
  if (param.verb > 1)
    cout << "writing " << file << endl ;
  dphi.write_imagesc(file) ;

  sprintf(file, "%s/template2targetMap", path) ;
  if (param.verb > 1)
    cout << "writing " << file << endl ;
  _phi.write(file) ;

  sprintf(file, "%s/target2templateMap", path) ;
  if (param.verb > 1)
    cout << "writing " << file << endl ;
  _psi.write(file) ;

  //  VectorMap psi ;
  //  _phi.inverseMap(psi, param.spaceRes) ;

}

void Shooting::initialPrint(char* path)
{
  if (!param.printFiles)
    return ;
  char file[256] ;

  sprintf(file, "%s/template", path) ;
  if (param.verb > 1)
    cout << "writing " << file << " " << Template.img().maxAbs() << endl ;
  Template.img().write_image(file) ;
  sprintf(file, "%s/binaryTemplate", path) ;
  if (param.verb > 1)
    cout << "writing " << file << endl ;
  Template.img().write(file) ;
  sprintf(file, "%s/target", path) ;
  if (param.verb > 1)
    cout << "writing " << file << " " << Target.img().maxAbs() << endl ;
  Target.img().write_image(file) ;
  sprintf(file, "%s/binaryTarget", path) ;
  if (param.verb > 1)
    cout << "writing " << file << endl ;
  Target.img().write(file) ;

  if (_kern.initialized() == false) {
    _kern.setParam(param) ;
    _kern.initFFT(imageDim, FFTW_MEASURE) ;
  }
  sprintf(file, "%s/kernel", path) ;
  _kern.kern.write_imagesc(file) ;

  if (_imageKern.initialized() == true) {
    //    cout << "saving image Kernel" << endl ;
    sprintf(file, "%s/imageKernel", path) ;
    _imageKern.kern.write_imagesc(file) ;
  }
  
}

