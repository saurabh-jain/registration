/**
   shootingSEG.cpp
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

_real SEGenergy(const VectorMap& psi, const deformableImage& I0, const Vector &ps)
{
  Vector I ;

  psi.multilinInterp(I0.img(), I) ;
  I *= I ;
  I *= ps ;
  return I.sum() ;
}

void SEGgradientInPsi(const VectorMap& psi, const deformableImage& I0, const Vector &ps, VectorMap &b, vector<_real> &resol)
{
  Vector I ;
  psi.multilinInterp(I0.img(), I) ;
  psi.multilinInterpGradient(I0.img(), b) ;
  b *= I ;
  b*= ps ;
}




void ShootingSEG::Print(char* path)
{
  if (!param.printFiles)
    return ;
  char file[256] ;
  Vector I1, I2 ;

  VectorMap dI0, LvI, vtry2, Lvtry2 ;
  if (param.gradInZ0) 
    Template.getMomentum(Z0, LvI) ;
  else
    LvI.copy(Lv0) ;
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

  sprintf(file, "%s/initialScalarMomentum", path) ;
  if (param.verb)
    cout << "writing " << file << endl ;
  Z0.write(file) ;
    
  sprintf(file, "%s/scaledScalarMomentum", path) ;
  if (param.verb)
    cout << "writing " << file << endl ;
  Z0.writeZeroCentered(file) ;

  sprintf(file, "%s/initialMomentum", path) ;
  if (param.verb)
    cout << "writing " << file << endl ;
  LvI.write(file) ;




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

void ShootingSEG::initialPrint(char* path)
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

  sprintf(file, "%s/Preseg", path) ;
  cout << "Target " << Target.img().max() << " " << Target.img().min() << endl ;
  cout << "Preseg " << preSeg.max() << " " << preSeg.min() << endl ;
  if (param.verb)
    cout << "writing " << file << endl ;
  preSeg.write_imagesc(file) ;
  cout << "end Preseg Print" << endl ; 

  if (_kern.initialized() == false) {
    _kern.setParam(param) ;
    _kern.initFFT(imageDim, FFTW_MEASURE) ;
  }
  sprintf(file, "%s/kernel", path) ;
  _kern.kern.write_imagesc(file) ;
  cout << "end Initial Print" << endl ; 
  
}

