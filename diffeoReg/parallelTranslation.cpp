/**
   parallelTranslation.cpp
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
#include "ImageEvolution.h"

int main(int argc, char** argv)
{
  if (argc < 2)
    {
      cout << "syntax: parallelTranslation param_file1" << endl ;
      exit(1) ;
    }

  param_matching param ;
  param.read(argv[1]) ;
  param.read(argc, argv) ;
  param.inverseKernelWeight = 1e-3 ;
  param.projectImage = true ;
  param.doNotModifyImages = true ;
  param.nb_semi = 0 ;


  deformableImage I0, I1, I2 ;
  Vector Z0, Z1, Z ;
  VectorMap Lv0, Lw0, Lwc, grad, foo ;
  char path[256] ;
  Vector dphi ;

  cout << "reading data" << endl ;
  Z0.read(param.fileMom1) ;
  sprintf(path, "%s/momentum1", param.outDir) ;
  Z0.write_image(path) ;
  Z1.read(param.fileMom2) ;
  sprintf(path, "%s/momentum2", param.outDir) ;
  Z1.write_image(path) ;

  ImageEvolution mo(param) ;


  cout << "computing momenta" << endl ;
  I0.copy(mo.Template) ;
  I0.computeGradient(mo.param.spaceRes,mo.param.gradientThreshold) ;
  I0.getMomentum(Z0, Lv0) ;
  I0.getMomentum(Z1, Lw0) ;
  sprintf(path, "%s/template", mo.param.outDir) ;
  I0.img().write_image(path) ;
  I2.copy(mo.Target) ;
  sprintf(path, "%s/target", mo.param.outDir) ;
  I2.img().write_image(path) ;
  mo.geodesicImageEvolutionFromVelocity(mo.Template, Lv0, I2, 1) ;
  mo._phi.logJacobian(dphi, mo.param.spaceRes) ;
  sprintf(path, "%s/jacobian1", mo.param.outDir) ;
  dphi.write_imagesc(path) ;
  sprintf(path, "%s/deformedTemplate1", mo.param.outDir) ;
  I2.img().write_image(path) ;
  mo.geodesicImageEvolutionFromVelocity(mo.Template, Lw0, I2, 1) ;
  mo._phi.logJacobian(dphi, mo.param.spaceRes) ;
  sprintf(path, "%s/jacobian2", mo.param.outDir) ;
  dphi.write_imagesc(path) ;
  sprintf(path, "%s/deformedTemplate2", mo.param.outDir) ;
  I2.img().write_image(path) ;


  //  exit(0) ;

  mo.param.epsilonTangentProjection = 1e-6 ;
  cout << "parallel transport " << Z1.maxAbs() << endl ;
//    VectorMap foo ;
//    mo.ParallelTranslation(Lv0, Lw0, Lwc, 1) ;
//    mo.geodesicImageEvolutionFromVelocity(I0, Lv0, I1, foo, 1) ;
//    mo.geodesicImageEvolutionFromVelocity(I1, Lwc, I2, foo, 1) ;
   mo.parallelImageTransport(I0, Z0, Z1, I2, Lwc, Z) ;
  //      mo.parallelImageTransportFromVelocity(I0, Z0, Z1, I2, Lwc, Z) ;
  //  Z *= I0.gradient().norm() ;
  sprintf(path, "%s/parallel", mo.param.outDir) ;
  I2.img().write_image(path) ;
  //  I2.imageTangentProjectionFP(I2, wc, foo, Z) ;
  sprintf(path, "%s/vectorMomentumParallel", mo.param.outDir) ;
  cout << " saving Lwc" << endl ; 
  Lwc.write(path) ;
  sprintf(path, "%s/scalarMomentumParallel", mo.param.outDir) ;
  Z.write(path) ;
  sprintf(path, "%s/scaledMomentumParallel", mo.param.outDir) ;
  Z.writeZeroCentered(path) ;

  mo.Target.computeGradient(mo.param.spaceRes,mo.param.gradientThreshold) ;
  mo.geodesicImageEvolutionFromVelocity(mo.Target, Lwc, I2, 1) ;
  sprintf(path, "%s/parallelTarg", mo.param.outDir) ;
  I2.img().write_image(path) ;
  //  mo.GeodesicDiffeoEvolution(Lwc, foo,  1) ;
  mo._phi.logJacobian(dphi, mo.param.spaceRes) ;
  sprintf(path, "%s/jacobianParallel", mo.param.outDir) ;
  dphi.write(path) ;
  dphi.write_imagesc(path) ;

  // Adjoint start transport
  mo.adjointStarTransport(I0, Lv0, Lw0, I2, Lwc, Z) ;
  cout << "end of adjointstar" << endl ;
  sprintf(path, "%s/adjoint", mo.param.outDir) ;
  I2.img().write_image(path) ;
  mo.geodesicImageEvolutionFromVelocity(mo.Target, Lwc, I2, 1) ;
  sprintf(path, "%s/adjointTarg", mo.param.outDir) ;
  I2.img().write_image(path) ;
  sprintf(path, "%s/scalarMomentumAdjoint", mo.param.outDir) ;
  Z.write(path) ;
  sprintf(path, "%s/scaledMomentumAdjoint", mo.param.outDir) ;
  Z.writeZeroCentered(path) ;
  mo._phi.logJacobian(dphi, mo.param.spaceRes) ;
  sprintf(path, "%s/jacobianAdjoint", mo.param.outDir) ;
  dphi.write(path) ;
  dphi.write_imagesc(path) ;
}


