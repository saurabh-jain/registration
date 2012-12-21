/**
   PointSetVectorTransforms.cpp
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

#include "pointSet.h"
//#include <nr3.h>

void  listToPointSets (const double *y, int sz, int dm, PointSet &p, Tangent &a) {
  p.al(sz,dm) ;
  a.al(sz,dm) ;
  int j=0 ;
  
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      p[k][kk] = y[j++] ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      a[k][kk] = y[j++] ;
}

void  listToPointSetsExtended (const double *y, int sz, int dm, int sz2, PointSet &p, Tangent &a, PointSet &q) {
  p.al(sz,dm) ;
  a.al(sz,dm) ;
  q.al(sz2,dm) ;
  int j=0 ;
  
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      p[k][kk] = y[j++] ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      a[k][kk] = y[j++] ;
  for(int k=0; k<sz2; k++)
    for (int kk=0; kk<dm; kk++)
      q[k][kk] = y[j++] ;
}

void  listToVarPointSets (const double *y, int sz, int dm, PointSet &p, Tangent &a, Tangent &dp, Tangent &da) {
  p.al(sz,dm) ;
  a.al(sz,dm) ;
  dp.al(sz,dm) ;
  da.al(sz,dm) ;
  int j=0 ;
  
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      p[k][kk] = y[j++] ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      a[k][kk] = y[j++] ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      dp[k][kk] = y[j++] ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      da[k][kk] = y[j++] ;
}

void  listToDualVarPointSets (const double *y, int sz, int dm, Tangent &dp, Tangent &da) {
  dp.al(sz,dm) ;
  da.al(sz,dm) ;
  int j=0 ;
  
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      dp[k][kk] = y[j++] ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      da[k][kk] = y[j++] ;
}

void  listToDiffPointSets (const double *y, int sz, int dm, PointSet &p, Tangent &a, MatSet &zp, MatSet &za) {
  p.al(sz,dm) ;
  a.al(sz,dm) ;
  zp.resize(sz,dm) ;
  za.resize(sz,dm) ;
  int j=0 ;
  
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      p[k][kk] = y[j++] ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      a[k][kk] = y[j++] ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      for(int l=0; l<sz; l++)
	for(int ll=0; ll<dm; ll++)
	  zp(k,kk,l,ll) = y[j++] ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      for(int l=0; l<sz; l++)
	for(int ll=0; ll<dm; ll++)
	  za(k,kk,l,ll) = y[j++] ;
  
}

void  pointSetsToList (const PointSet &p, const Tangent &a, double* y) {
  int sz = p.size() ;
  int dm = p.dim() ;
  //  y.resize(2*sz*dm) ;
  int j=0 ;
  
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      y[j++] = p[k][kk]  ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      y[j++] = a[k][kk]  ;
}

void  pointSetsExtendedToList (const PointSet &p, const Tangent &a, const PointSet &q, double *y) {
  int sz = p.size() ;
  int sz2 = q.size() ;
  int dm = p.dim() ;
  //  y.resize((2*sz+sz2)*dm) ;
  int j=0 ;
  
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      y[j++] = p[k][kk]  ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      y[j++] = a[k][kk]  ;
  for(int k=0; k<sz2; k++)
    for (int kk=0; kk<dm; kk++)
      y[j++] = q[k][kk]  ;
}

void  varPointSetsToList (const PointSet &p, const Tangent &a, const Tangent &dp, const Tangent &da, double *y) {
  int sz = p.size() ;
  int dm = p.dim() ;
  //  y.resize(4*sz*dm) ;
  int j=0 ;
  
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      y[j++] = p[k][kk]  ;
  for(int k=0; k<sz; k++)
      for (int kk=0; kk<dm; kk++)
	y[j++] = a[k][kk]  ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      y[j++] = dp[k][kk]  ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      y[j++] = da[k][kk]  ;
}

void  dualVarPointSetsToList (const Tangent &dp, const Tangent &da, double *y) {
  int sz = dp.size() ;
  int dm = dp.dim() ;
  //  y.resize(2*sz*dm) ;
  int j=0 ;
    for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      y[j++] = dp[k][kk]  ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      y[j++] = da[k][kk]  ;
}

void  diffPointSetsToList (const PointSet &p, const Tangent &a, const MatSet &zp, const MatSet &za, double *y) {
  int sz = p.size() ;
  int dm = p.dim() ;
  //  y.resize(2*sz*dm + 2*sz*sz*dm*dm) ;
  int j=0 ;
  
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      y[j++] = p[k][kk]  ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      y[j++] = a[k][kk]  ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      for(int l=0; l<sz; l++)
	for(int ll=0; ll<dm; ll++)
	  y[j++] = zp(k,kk,l,ll)  ;
  for(int k=0; k<sz; k++)
    for (int kk=0; kk<dm; kk++)
      for(int l=0; l<sz; l++)
	for(int ll=0; ll<dm; ll++)
	  y[j++] = za(k,kk,l,ll) ;
}

