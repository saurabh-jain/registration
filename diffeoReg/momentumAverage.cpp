/**
   momentumAverage.cpp
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

#ifdef MEM
int mem ;
#endif

int main(int argc, char** argv)
{
#ifdef MEM
  mem = 0 ;
#endif

  if (argc != 2)
    {
      cout << "syntax: momentumAverage  param_file1" << endl ;
      exit(1) ;
    }

  ImageEvolution mo ;
  char path[256] ;
  param_matching p0 ;
  deformableImage I0, I2, oldTemplate, foo ;
  Vector Z0, Z, dphi ;
  VectorMap Lv, foom, Lv0, v ;

  p0.read(argv[1]) ;
  p0.read(argc, argv) ;
  p0.verb = 1 ;

  mo.param.copy(p0) ;
  mo.Load() ;
  mo.param.doNotModifyTemplate = true ;

  Z.read(mo.param.dataMom[0].c_str()) ;
  for (unsigned int i=1; i<mo.param.dataMom.size(); i++) {
    Z0.read(mo.param.dataMom[i].c_str()) ;
    Z +=  Z0 ;
  }

  Z /= mo.param.dataMom.size(); 

  I0.get_image(p0.fileTemp, p0.dim.size()) ;
  I0.computeGradient(mo.param.spaceRes,mo.param.gradientThreshold) ;
  I0.getMomentum(Z, Lv) ;
  mo.kernel(Lv, v) ;
  mo.geodesicImageEvolutionFromVelocity(I0, Lv, I2, 1.0) ;
  sprintf(path, "%s/average", mo.param.outDir) ;
  I2.img().write_image(path) ;
  sprintf(path, "%s/averageMomentum", mo.param.outDir) ;
  Z.write(path) ;
  sprintf(path, "%s/scaledAverageMomentum", mo.param.outDir) ;
  Z.writeZeroCentered(path) ;
  mo._phi.logJacobian(dphi, mo.param.spaceRes) ;
  sprintf(path, "%s/jacobianAverage", mo.param.outDir) ;
  dphi.write(path) ;
  Lv*=20 ;
  mo.geodesicImageEvolutionFromVelocity(I0, Lv, I2, 1.0) ;
  sprintf(path, "%s/average2", mo.param.outDir) ;
  I2.img().write_image(path) ;
}


