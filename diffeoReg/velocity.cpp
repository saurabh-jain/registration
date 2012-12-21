/**
   velocity.cpp
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
#include "velocity.h"

void LDDMMTimegradient(const VectorMap& phi, const VectorMap &psi, const deformableImage& I0, const deformableImage& I1, VectorMap &b, vector<_real> &resol)
{
  Vector DI ; 
  Vector I, jj, foo ; 
  VectorMap foom ;
  std::vector<VectorMap> Dpsi ;

  phi.multilinInterp(I0.img(), I) ;
  psi.multilinInterp(I1.img(), DI) ;
  DI -= I ;

  phi.jacobian(jj, resol) ;
  DI *= jj ;    
  gradient(I, b, resol) ;
  b *= DI ;
}



void Velocity::Print(char* path)
{
  char file[256] ;
  Vector I1, I2 ;
  VectorMap Lwc, psi ;

  psi.copy(_psi) ;
  initFlow(Lv0.d) ;
  for(unsigned int t=0; t<Lv0.size(); t++) {
    updateFlow(Lv0[t], 1.0/_T) ;
    if (param.saveMovie) {
      _psi.multilinInterp(Template.img(), I1) ;
      sprintf(file, "%s/movie%03d", path, t+1) ;
      I1.write_image(file) ;
    }
  }



  _psi.multilinInterp(Template.img(), I1) ;
  sprintf(file, "%s/deformedTemplate", path) ;
  if (param.verb)
    cout << "writing " << file << endl ;
  I1.write_image(file) ;
  _phi.multilinInterp(I1, I2) ;
  /*
  sprintf(file, "%s/test", path) ;
  if (param.verb)
    cout << "writing " << file << endl ;
  I2.write_image(file) ;
  */

  _phi.multilinInterp(Target.img(), I1) ;
  sprintf(file, "%s/deformedTarget", path) ;
  if (param.verb)
    cout << "writing " << file << endl ;
  I1.write_image(file) ;

  sprintf(file, "%s/initialMomentum", path) ;
  if (param.verb)
    cout << "writing " << file << endl ;
  Lv0[0].write(file) ;

  Vector dphi ;
  _phi.logJacobian(dphi, param.spaceRes) ;

  sprintf(file, "%s/jacobian", path) ;
  if (param.verb)
    cout << "writing " << file << endl ;
  dphi.write(file) ;

  GeodesicDiffeoEvolution(Lv0[0]) ;
  _psi.multilinInterp(Template.img(), I1) ;
  sprintf(file, "%s/shootedTemplate", path) ;
  if (param.verb)
    cout << "writing " << file << endl ;
  I1.write_image(file) ;

  if (param.saveProjectedMomentum) {
    //ImageEvolution mo ;
    Vector Z0, Z ;
    VectorMap v0 ;
    kernel(Lv0[0], v0) ;
    v0.scalProd(Template.normal(), Z) ;
    Z *= -1 ;
    //    Template.infinitesimalAction(v0, Z) ;
    cout << "projection " << Z.d.length << endl ;
    imageTangentProjection(Template, Z, Z0) ;
    sprintf(file, "%s/initialScalarMomentum", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    Z0.write(file) ;


    sprintf(file, "%s/scaledScalarMomentum", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    Z0.writeZeroCentered(file) ;



    //  Z0 *= -1 ;
    Template.getMomentum(Z0, Lwc) ;
    GeodesicDiffeoEvolution(Lwc) ;
    _psi.multilinInterp(Template.img(), I1) ;
    sprintf(file, "%s/Z0ShootedTemplate", path) ;
    if (param.verb)
      cout << "writing " << file << endl ;
    I1.write_image(file) ;
  }

  _psi.copy(psi) ;
  sprintf(file, "%s/template2targetMap", path) ;
  if (param.verb)
    cout << "writing " << file << endl ;
  _phi.write(file) ;

  sprintf(file, "%s/target2templateMap", path) ;
  if (param.verb)
    cout << "writing " << file << endl ;
  _psi.write(file) ;

}


void Velocity::initialPrint(char* path)
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
  if (param.verb)
    cout << "writing " << file << endl ;

  if (_kern.initialized() == false) {
    _kern.setParam(param) ;
    _kern.initFFT(imageDim, FFTW_MEASURE) ;
  }
  //  cout << "kern? " << _kern.initialized() << endl ;
  _kern.kern.write_imagesc(file) ; 

  //  cout << "done initial print" << endl;
}
