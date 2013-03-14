/**
   procrusteanAverage.cpp
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
#include <omp.h>

int main(int argc, char** argv)
{
  if (argc < 2)
    {
      cout << "syntax: procrusteanAverage param_file1" << endl ;
      exit(1) ;
    }


  char path[256] ;
  param_matching p0 ;
  Shooting mo, mo1 ;
  deformableImage I2, oldTemplate, foo ;
  Vector Z0, Z ;
  VectorMap Lv, foom, Lv0 ;
  bool translate_momenta = true ;
  double nmOld ;

  p0.read(argv[1]) ;
  p0.read(argc, argv) ;
  p0.verb = 0 ;
  p0.printFiles = 0 ;

#ifdef _PARALLEL_
  omp_set_num_threads(p0.nb_threads) ;
#endif

  sprintf(path, "%s/procrusteanTemplate", p0.outDir) ;
  int i ;

  for (unsigned int iter=0; iter < 100; iter++) {
    cout << endl << endl << endl << endl << "ITERATION " << iter << endl ;
#pragma omp parallel  private(i,mo1,Z0,path,nmOld,Lv) shared(p0,iter)
#pragma omp for
    for (i=0; i<(int) p0.dataSet.size(); i++) {
      Shooting mo1 ;
      mo1.param.copy(p0) ;

      if (iter ==0 && !mo1.param.cont) {
	if (! p0.foundTemplate)
	  strcpy(mo1.param.fileTemp, p0.dataSet[0].c_str()) ;
	mo1.param.nb_iter *= 5 ;
      }
      else {
	mo1.param.doNotModifyTemplate = true ;
	mo1.param.readBinaryTemplate = true ;
	sprintf(path, "%s/procrusteanTemplate", mo1.param.outDir) ;
	strcpy(mo1.param.fileTemp, path) ;
      }
      mo1.param.foundTarget = 1 ;
      mo1.param.foundTemplate = 1 ;
      strcpy(mo1.param.fileTarg, p0.dataSet[i].c_str()) ;
#pragma omp critical
      mo1.Load() ;
      mo1.Template.computeGradient(p0.spaceRes,p0.gradientThreshold);
      // Should initialize the matching with the translated momentum of the previous matching...
      cout << "comparing " << mo1.param.fileTemp << " and " << mo1.param.fileTarg << endl ;
      if (iter > 0) {
	sprintf(path, "%s/momentum%d", p0.outDir, i) ;
	Z0.read(path) ;
	mo1.Template.getMomentum(Z0, Lv) ;
	nmOld =mo1.kernelNorm(Lv) ;
	mo1.gradientImageMatching(Z0) ;
      }
      else {
	nmOld = 0 ;
	mo1.gradientImageMatching() ;
      }
      sprintf(path, "%s/momentum%d", p0.outDir, i) ;
      mo1.Z0.write(path) ;
      mo1.Template.getMomentum(mo1.Z0, Lv) ;
      cout << "L2 norm of the momentum (" << i << ") " <<  nmOld << " " << mo1.kernelNorm(Lv) << endl ;
    }
    
    
    mo.param.copy(p0) ;
    if (iter ==0 && !mo.param.cont) {
      if (! p0.foundTemplate)
	strcpy(mo.param.fileTemp, p0.dataSet[0].c_str()) ;
    }
    else {
      mo.param.doNotModifyTemplate = true ;
      mo.param.readBinaryTemplate = true ;
      sprintf(path, "%s/procrusteanTemplate", mo.param.outDir) ;
      strcpy(mo.param.fileTemp, path) ;
    }
    mo.param.foundTemplate = 1 ;
    mo.param.foundTarget = 0 ;
    mo.Load() ;
    mo.Template.computeGradient(p0.spaceRes,p0.gradientThreshold);
    double n1 = 0 ;
    sprintf(path, "%s/momentum0", p0.outDir) ;
    Z.read(path) ;
    mo.Template.getMomentum(Z, Lv) ;
    n1 = mo.kernelNorm(Lv) ;
    for (i=1; i<(int) p0.dataSet.size(); i++) {
      sprintf(path, "%s/momentum%d", p0.outDir, i) ;
      Z0.read(path) ;
      Z +=  Z0 ;
      mo.Template.getMomentum(Z0, Lv) ;
      n1 += mo.kernelNorm(Lv) ;
    }


    Z /= p0.dataSet.size();
    n1 /= p0.dataSet.size() ; 
    mo.Template.getMomentum(Z, Lv) ;

    cout << "relative L2 norm of the average momentum " << mo.kernelNorm(Lv)/n1 << endl ;
    cout << "sum of square distances " << n1 << endl ;

    mo.geodesicImageEvolutionFromVelocity(mo.Template, Lv, I2,  1) ;
    sprintf(path, "%s/procrusteanTemplate", mo.param.outDir) ;
    oldTemplate.copy(mo.Template) ;
    oldTemplate.computeGradient(p0.spaceRes,p0.gradientThreshold) ;
    mo.Template.copy(I2) ;
    mo.Template.computeGradient(p0.spaceRes,p0.gradientThreshold) ;
    cout << "writing " << path << endl ;
    mo.Template.img().write_image(path) ;
    mo.Template.img().write(path) ;
    if (translate_momenta) {
	mo.adjointStarTransport(oldTemplate, Lv, Lv, I2, foom, Z) ;
#pragma omp parallel  private(i,Z0,Lv0,foom,path) shared(Z,mo,I2,oldTemplate,p0,Lv,iter)
#pragma omp for
      for (i=0; i<(int)p0.dataSet.size(); i++) {
	sprintf(path, "%s/momentum%d", p0.outDir, i) ;
	Z0.read(path) ;
	oldTemplate.getMomentum(Z0, Lv0) ;
	cout << "transporting the momenta" << endl ;
	mo.adjointStarTransport(oldTemplate, Lv, Lv0, I2, foom, Z0) ;
	Z0 -= Z ;
	Z0 *= -1  ;
	sprintf(path, "%s/momentum%d", p0.outDir, i) ;
	mo.Template.getMomentum(Z0, Lv0) ;
	mo.geodesicImageEvolutionFromVelocity(mo.Template, Lv0, I2,  1) ;
	sprintf(path, "%s/testMomentum%d", p0.outDir, i) ;
	I2.img().write_image(path) ;
      }
    }
  }
}


