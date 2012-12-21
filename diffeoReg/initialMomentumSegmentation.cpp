/**
   initialMomentumSegmentation.cpp
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

  if (argc < 2)
    {
      cout << "syntax: initialMomentumMatchingSEG param_file1" << endl ;
      exit(1) ;
    }

  ShootingSEG mo(argv[1], argc, argv) ;


  cout << mo.param.lambda << " " << mo.param.sigma << " " << mo.lambdaSEG1 << " " << mo.lambdaSEG2 << endl ;
  mo.initialPrint() ;  
  Vector Z, Z0 ;
  if (mo.param.foundInitialMomentum) {
    Z0.read(mo.param.fileInitialMom) ;
  }
  else {
    Z0.al(mo.Template.domain()); 
    Z0.zero();
  }
  mo.param.gradientThreshold = .1 ;

  
  mo.gradientImageMatching(Z0) ;
  mo.Print() ;
}


