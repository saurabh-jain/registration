/**
   shoot.h
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
      cout << "syntax: shoot param_file1" << endl ;
      exit(1) ;
    }

  param_matching param ;
  param.read(argv[1]) ;
  param.read(argc, argv) ;
  param.inverseKernelWeight = 0 ;
  param.projectImage = true ;
  param.doNotModifyImages = true ;
  param.nb_semi = 0 ;

  ImageEvolution mo(param) ;
  cout << param.outDir << ";" << mo.param.outDir << endl ;

  deformableImage I0, I2 ;
  Vector Z0 ;
  VectorMap Lv0 ;
  char path[256] ;
  Vector dphi ;


  cout << "reading data" << endl ;
  I0.copy(mo.Template) ;
  I0.computeGradient(mo.param.spaceRes,mo.param.gradientThreshold) ;

  if (mo.param.useVectorMomentum) {
    cout << "reading Vector Momentum" << mo.param.fileInitialMom << endl ;
    Lv0.read(mo.param.fileInitialMom) ;
  }
  else {
    Z0.read(mo.param.fileInitialMom) ;
    I0.getMomentum(Z0, Lv0) ;
  }

  sprintf(path, "%s/template", mo.param.outDir) ;
  I0.img().write_image(path) ;
  mo.geodesicImageEvolutionFromVelocity(mo.Template, Lv0, I2, 1) ;
  mo._phi.logJacobian(dphi, mo.param.spaceRes) ;
  sprintf(path, "%s/jacobian", mo.param.outDir) ;
  dphi.write_imagesc(path) ;
  sprintf(path, "%s/deformedTemplate", mo.param.outDir) ;
  I2.img().write_image(path) ;

  sprintf(path, "%s/template2targetMap", mo.param.outDir) ;
  mo._phi.write(path) ;

  sprintf(path, "%s/target2templateMap", mo.param.outDir) ;
  mo._psi.write(path) ;

}


