/**
   pointSetFunctors.h
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
#ifndef _PSETFUNCTORS
#define _PSETFUNCTORS
#include "pointSet.h"

void  listToPointSets (const double *y, int sz, int dm, PointSet &p, Tangent &a) ;
void  listToPointSetsExtended (const double *y, int sz, int dm, int sz2, PointSet &p, Tangent &a, PointSet &q) ;
void  listToVarPointSets (const double *y, int sz, int dm, PointSet &p, Tangent &a, Tangent &dp, Tangent &da) ;
void  listToDualVarPointSets (const double *y, int sz, int dm, Tangent &dp, Tangent &da) ;
void  listToDiffPointSets (const double *y, int sz, int dm, PointSet &p, Tangent &a, MatSet &zp, MatSet &za) ;
void  pointSetsToList (const PointSet &p, const Tangent &a, double *y) ;
void  pointSetsExtendedToList (const PointSet &p, const Tangent &a, const PointSet & q, double* y) ;
void  varPointSetsToList (const PointSet &p, const Tangent &a, const Tangent &dp, const Tangent &da, double *y) ;
void  dualVarPointSetsToList (const Tangent &dp, const Tangent &da, double *y) ;
void  diffPointSetsToList (const PointSet &p, const Tangent &a, const MatSet &zp, const MatSet &za, double *y) ;

/*template <class KERNEL> struct landmarkUpdate {
  landmarkUpdate(){size=0; dim=0;}
  landmarkUpdate(int s, int d){size = s; dim = d;}
  int size ;
  int dim ;
  void operator()(const Doub x, VecDoub_I &y, VecDoub_O &dydx) {
    PointSet p ;
    Tangent a;
    listToPointSets(y, size, dim, p,a) ;
    int j=0 ;
    for (int k=0; k<size; k++) 
      for (int kk=0; kk<dim; kk++) {
	dydx[j] = 0 ;
	for (int l=0; l<size; l++)
	  dydx[j] += KERNEL(p[k], p[l]) *a[l][kk] ;
	j++ ;
      }

    double u ;
    for (int k=0; k<size; k++) 
      for (int kk=0; kk<dim; kk++) {
	dydx[j] = 0 ;
	for (int l=0; l<size; l++) {
	  u = 0 ;
	  for(int ll=0; ll<size; ll++)
	    u += a[k][ll]*a[l][ll] ;
	  dydx[j] -= 2*KERNEL::diff(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u ;
	}
	j++ ;
      }
  }
  };*/



#endif
