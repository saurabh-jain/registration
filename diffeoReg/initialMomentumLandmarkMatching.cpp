/**
   initialMomentumLandmarkMatching.cpp
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

#include "kernels.h"
#include "landmarkMatching.h"


int main(int argc, char** argv)
{
  if (argc < 2)
    {
      cout << "syntax: initialMomentumMatching param_file1" << endl ;
      exit(1) ;
    }

  LandmarkMatching<GaussKernel> mo(argv[1], argc, argv) ;

  mo.initKernel() ;

  mo.initialPrint() ;  
  PointSet Z, Z0 ;
  if (mo.param.foundInitialMomentum) {
    Z0.get_points(mo.param.fileInitialMom, mo.param.ndim) ;
  }
  else {
    Z0.zeros(mo.Template.size(), mo.Template.dim()); 
  }

  //  mo.downscale(Z0) ;
  //  int nbiter = mo.param.nb_iter ;
  //  mo.param.nb_iter = 15 ;
  //  mo.gradientImageMatching(Z0) ;
  //  mo.upscale(mo.Z0) ;
  //  mo.param.nb_iter = nbiter ;
  //  mo.gradDesc(Z0, Z, mo.param.nb_iter, 0.001, mo.param.epsMax, mo.param.minVarEn, mo.param.gs, mo.param.verb, (void *) &(mo.param)) ;
    mo.doMetamorphosis() ;
    if (mo.meta == false)
      cout << "not meta!!!!!" << endl ;
  mo.gradientLandmarkMatching(Z0) ;
  //  mo.Print() ;
  
}


