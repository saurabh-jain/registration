/**
   landmarkMatching.cpp
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

_real LDDMMLandmarkEnergy(const PointSet& p0, const PointSet& p1)
{
  double res =0, u ;
  //  cout << "energy: " << p0.size() << " " << p0.dim() << endl;
  for(unsigned int k=0; k<p0.size(); k++)
    for(int kk=0; kk<p0.dim(); kk++) {
      //      cout << p0[k][kk] << " " << p1[k][kk] << endl ;
      u = (p0[k][kk]-p1[k][kk]) ;
      res +=  u*u ;
    }
  return res ;
}

/**
Computes the Eulerian differential of the LDDMM data attachment term considered as a function of the inverse deformation
*/
void LDDMMLandmarkGradient(const PointSet& p0, const PointSet & p1, Tangent &b)
{
  b.al(p0.size(), p0.dim()) ;
  for (unsigned int k=0; k<p0.size(); k++)
    for(int kk=0; kk<p0.dim(); kk++)
      b[k][kk] = (p0[k][kk] - p1[k][kk])/p0.size() ;
}







