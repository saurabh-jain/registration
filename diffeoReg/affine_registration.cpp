/**
   affineRegistration.cpp
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


int main(int argc, char** argv)
{
  Shooting mo, m1 ;

  if (argc != 5 && argc != 6)
    {
      cout << "syntax: affine_registration im1 im2 dirOut affine_type [flipDimension]" << endl ;
      cout << "affinetype is either:  none, rotation, similitude, special, general" << endl ;
      exit(1) ;
    }

  mo.param.foundTarget = 1 ;
  mo.param.defaultArray(3) ;
  sprintf(mo.param.fileTemp, "%s", argv[1]) ;
  sprintf(mo.param.fileTarg, "%s", argv[2]) ;
  mo.param.doDefor = false ;
  mo.param.scaleScalars = true ;
  mo.param.scaleThreshold = 100 ;
  if (argc == 6) {
    mo.param.flipTarget = true ;
    mo.param.flipDim = atoi(argv[5]) ;
  }
  mo.param.nb_iterAff = 10 ;
  mo.param.sigmaGauss = -1 ;

  std::map<string, int>::iterator IA ;

  IA = mo.param.affMap.find(argv[4]) ;
  if (IA == mo.param.affMap.end()) {
    cerr << "unknown affine key" << endl ;
    exit(1) ;
  }
  else
    mo.param.type_group = IA -> second ;


  mo.Load() ;
  mo.param.verb= 1 ;

  ImageAffineEnergy eneraff ;  
  mo.affineReg() ;

  char path[256] ;
  sprintf(path, "%s/registeredImage", argv[3]) ;
  mo.Target.img().write_image(path) ;
  sprintf(path, "%s/affine_transformation.dat", argv[3]) ;
  ofstream ofs(path) ;
  ofs << mo.affTrans << endl ;
  ofs.close() ;
}


