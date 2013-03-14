/**
   shotingSym.cpp
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

_real SYMenergy(const VectorMap& psi, const deformableImage& I0, const deformableImage& I1, vector<_real> &resol)
{
  Vector I, jj, foo ;

  psi.jacobian(jj, resol) ;
  jj.execFunction(jj, sqrt) ;
  psi.multilinInterp(I0.img(), I) ;
  foo.copy(I) ;
  foo -= I1.img() ;
  //  foo = (Vector) ( I - I1.img()) ;
  foo *= foo ;
  foo *= jj ;
  return foo.sum() ;
}

// gradient of the data attchment term
void SYMgradientInverse(const VectorMap& phi, const deformableImage& I0, const deformableImage& I1, VectorMap &b, vector<_real> &resol)
{
  VectorMap  dI0, foom;
  Vector DI, I, jj , foo;

  phi.multilinInterp(I1.img(), I) ;
  //cout << "interp" << endl ;
  DI.copy(I0.img()) ;
  DI -= I ;

  phi.jacobian(jj, resol) ;
  jj.execFunction(jj, sqrt) ;
  // cout << "jacobian " << jj.sum() << endl ;
  DI *= jj ;    
  b.copy(I0.gradient()) ;
  b *= DI ;

  DI.copy(I0.img()) ;
  DI -= I ;
  DI *= DI ;
  DI *= jj ;
  gradient(DI, foom, resol) ;
  foom /= 4 ;
  b -= foom ;
}


void SYMgradient(const VectorMap &phi, const VectorMap& psi, const deformableImage& I0, const deformableImage& I1, VectorMap &b, vector<_real> &resol)
{
  Vector DI,foo1 ; 
  Vector I, jj ;
  VectorMap foom, foo2 ;

  psi.multilinInterp(I0.img(), DI) ;
  //cout << "interp" << endl ;
  DI -= I1.img() ;
  //  DI *= -1 ;

  psi.multilinInterpGradient(I0.img(), b) ;
  //  b.copy(I.gradient()) ;
  b *= DI ;
  // b *= -1 ;

  psi.jacobian(jj, resol) ;
  jj.execFunction(jj, sqrt) ;


  DI *= DI ;
  DI /= jj ;
  phi.multilinInterp(DI, foo1) ;
  gradient(foo1, foom, resol) ;
  psi.multilinInterp(foom, foo2) ;
  foo2 /= 4 ;
  foo2 *= jj ;
  b -= foo2 ;
  b *= jj ;
}

/*
double ShootingSYM::computeGrad(VectorMap &LvI, VectorMap& grad, void *par){
  VectorMap b, vI, foo ;
  double ng ;
  SYMgradient(_phi, _psi, Template, Target, b, ((param_matching*) par)->spaceRes) ;
  b /= ((param_matching*) par)->sigma*((param_matching*) par)->sigma ;
  kernel(LvI, vI) ;
  DualVarGeodesicDiffeoEvolutionDiscrete(LvI, b, foo) ;
  inverseKernel(foo, grad) ;
  grad += LvI ;
  ng = kernelNorm(grad) ;
  return ng ;
}

double ShootingSYM::objectiveFun(VectorMap &Lv, void * par) 
{
  VectorMap vI ;
  GeodesicDiffeoEvolution(Lv) ;
  kernel(Lv, vI) ;
  return (vI.scalProd(Lv) + SYMenergy(_psi, Template, Target, ((param_matching*) par)->spaceRes)/ (((param_matching*) par)->sigma*((param_matching*) par)->sigma))/Lv.d.length ;
}

double ShootingSYM::endOfIteration(VectorMap &Lv, void * par)
{
  Lv0.copy(Lv) ;
  Print() ;
  return -1 ;
}

void ShootingSYM::endOfProcedure(VectorMap &Lv)
{
  Lv0.copy(Lv) ;
  Print() ;
}
*/




void ShootingSYM::Print(char* path)
{
  if (!param.printFiles)
    return ;
  char file[256] ;

  Vector I1 ;
  //  GeodesicDiffeoEvolution(LvI, Lvtry2, 1) ;

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

  sprintf(file, "%s/initialMomentum", path) ;
  if (param.verb)
    cout << "writing " << file << endl ;
  Lv0.write(file) ;


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

void ShootingSYM::initialPrint(char* path)
{
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
  _kern.kern.write_imagesc(file) ;
  
}

