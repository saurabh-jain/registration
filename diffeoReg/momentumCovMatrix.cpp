/**
   momentumCovMatric.cpp
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

  if (argc < 2)
    {
      cout << "syntax: momentumCovmatrix param_file" << endl ;
      exit(1) ;
    }

  param_matching param ;
  param.read(argv[1]) ;
  param.read(argc, argv) ;
  param.doNotModifyImages = true ;
  //  param.useVectorMomentum = true ;

  ImageEvolution mo(param) ;

  deformableImage I0 ;
  Vector Z0, Z1 ;
  VectorMap Lv0, Lv1, v1, avg ;
  Matrix cov ;


  cov.resize(mo.param.dataMom.size(), mo.param.dataMom.size()) ;
  if (mo.param.useVectorMomentum) {
    for (unsigned int k=0; k<mo.param.dataMom.size(); k++) {
      cout << "Momentum " << k << ":" << flush ;
      Lv0.read(mo.param.dataMom[k].c_str()) ;
      //      Lv0 -= avg ;
      for (unsigned int l=k; l<mo.param.dataMom.size(); l++) {
	Lv1.read(mo.param.dataMom[l].c_str()) ;
	cout << " " << l  << flush ;
	//	Lv1 -= avg ;
	mo.kernel(Lv1, v1) ;
	cov(k,l) = Lv0.scalProd(v1) ;
	cov(l,k) = cov(k,l) ;
      }
      cout << endl ;
    }
  }
  else {
    I0.copy(mo.Template) ;
    I0.computeGradient(mo.param.spaceRes,mo.param.gradientThreshold) ;
    Z0.read(mo.param.dataMom[0].c_str()) ;
    for (unsigned int k=0; k<mo.param.dataMom.size(); k++) {
      Z0.read(mo.param.dataMom[k].c_str()) ;
      I0.getMomentum(Z0, Lv0) ;
      for (unsigned int l=k; l<mo.param.dataMom.size(); l++) {
	Z0.read(mo.param.dataMom[l].c_str()) ;
	I0.getMomentum(Z0, Lv1) ;
	mo.kernel(Lv1, v1) ;
	cov(k,l) = Lv0.scalProd(v1) ;
	cov(l,k) = cov(k,l) ;
      }
    }
  }

  if (mo.param.foundResult) {
    ofstream ofs ;
    ofs.open(mo.param.fileResult) ;
    if (ofs.fail()) {
      cerr << "Unable to open " << mo.param.fileResult << endl ;
      cov.Print() ;
      exit(1) ;
    }
    else {
      for (unsigned int i=0; i<cov.nRows(); i++) {
	for (unsigned int j=0; j<cov.nColumns(); j++)
	  ofs << cov(i, j) << " ";
	ofs << endl ;
      }
      ofs.close() ;
    }
  }
  else
    cov.Print() ;


}


