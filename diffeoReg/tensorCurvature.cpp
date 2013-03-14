/**
   tensorCurvature.cpp
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
#include "tensorField.h"

int main(int argc, char** argv)
{
  if (argc != 3)
    {
      cerr << "Usage: tensorCurvature InputFile OutputFile" << endl ;
      exit(1) ;
    }

  VectorMap eig1 ;
  SymmetricTensorField DTI ;
  vector<double> resol(3) ;
  Vector res, fa, trc, scurv ;
  char path[256] ;
  for (int k=0; k<3; k++) resol[k]  = 1 ;


  cout << "reading file" << endl ;
  DTI.readTensorField(argv[1]) ;
  cout << DTI.d << endl ;
  //  cout << sizeof(SymmetricTensor) * DTI.d.length << endl ;

  //  VectorMap id ;
  //   Ivector I(3), J(3);
  
  //   I[0] = -50; I[1] = -50; I[2] = -50 ;
  //   J[0] = 50; J[1] = 50; J[2] = 50 ;
  //   Domain D(I,J) ;
  //   id.idMap(D)  ;

  DTI.computeTrace(trc) ;
  sprintf(path, "%s_tr", argv[2]) ;
  trc.write_imagesc(path) ;
  cout << "Inverting tensor " << endl;
  DTI.inverseTensor() ;
  sprintf(path, "%s_det0", argv[2]) ;
  DTI.determinant().write(path) ;
//   DTI.inverseTensor(pow(DTI.determinant().max(.01),.33)/100) ;
//   //  DTI.pruneOutliers() ;
//   sprintf(path, "%s_det", argv[2]) ;
//   DTI.determinant().write_imagesc(path) ;
//   DTI.determinant().write(path) ;
  
  cout <<"censoring data" << endl ;
  DTI.censorTensor(0) ;
  sprintf(path, "%s_mask", argv[2]) ;
  //  DTI.Mask.write_imagesc(path) ;
  DTI.Mask.write(path) ;

  cout << "Computing Fractional Anisotropy" << endl ;
  DTI.computeFractionalAnisotropy(fa) ;
  cout << "average anisotropy: " << fa.mean() << endl ;
  sprintf(path, "%s_fa", argv[2]) ;
  //  fa.censor(DTI.Mask, .005) ;
  fa.write_imagesc(path) ;
  fa.write(path) ;

  cout << "Computing principal eigenvector" << endl ;
  DTI.computeFirstEigenvector(eig1) ;
  sprintf(path, "%s_eig", argv[2]) ;
  eig1.write(path) ;
  //  DTI.normalizeByTrace() ;  

  DTI.smoothTensor(2, .9) ;
  //  DTI.censorTensor(1) ;
  DTI.swapMetric() ;
  cout << "Computing symbols" << endl ;
  DTI.computeChristoffelSymbols(resol) ;
  
  cout << "computing curvature" << endl ;
  DTI.computeScalarCurvature(scurv, resol) ;

  sprintf(path, "%s_curv", argv[2]) ;
  scurv *= -1 ;
  res.copy(scurv) ;
  trc += 1e-8;
  res /= trc ;
  scurv.censor(DTI.Mask, .05) ;
  scurv.writeZeroCentered(path) ;
  scurv.write(path) ;
  sprintf(path, "%s_curvtr", argv[2]) ;
  res.censor(DTI.Mask, .05) ;
  res.writeZeroCentered(path) ;
  res.write(path) ;
  sprintf(path, "%s_curvfa", argv[2]) ;
  res *= fa ;
  res.censor(DTI.Mask, .05) ;
  res.writeZeroCentered(path) ;

  SymmetricTensorField Ric ;
  cout << "computing Ricci curvature" << endl ;
  DTI.computeRicciCurvature(Ric, resol) ;
  Ric.computeFirstEigenvector(eig1) ;
  sprintf(path, "%s_eigRicci", argv[2]) ;
  eig1.write(path) ;
}
