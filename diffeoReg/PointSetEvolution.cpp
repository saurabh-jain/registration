/**
   PointSetEvolution.cpp
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

#include "PointSetEvolution.h"


template <class KERNEL> struct landmarkUpdate {
  landmarkUpdate(){size=0; dim=0;}
  landmarkUpdate(int s, int d){size = s; dim = d;}
  int size ;
  int dim ;
  void operator()(const Doub x, VecDoub_I &y, VectDoub_0 &dydx) {
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

    for (int k=0; k<size; k++) 
      for (int kk=0; kk<dim; kk++) {
	dydx[j] = 0 ;
	for (int l=0; l<size; l++) {
	  u = 0 ;
	  for(int ll=0; ll<size; ll++)
	    u += a[k][ll]*a[l][ll] ;
	  dydx[j] -= 2*KERNEL.diff(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u ;
	}
	j++ ;
      }
  }
};

template <class KERNEL> struct varLandmarkUpdate {
  varLandmarkUpdate(){size=0; dim=0;}
  varLandmarkUpdate(int s, int d){size = s; dim = d;}
  int size ;
  int dim ;
  void operator()(const Doub x, VecDoub_I &y, VectDoub_0 &dydx) {
    PointSet p ;
    Tangent a, dp, da;
    double u, uu ;
    listToVarPointSets(y, size, dim, p,a, dp, da) ;
    int j=0 ;
    for (int k=0; k<size; k++) 
      for (int kk=0; kk<dim; kk++) {
	dydx[j] = 0 ;
	for (int l=0; l<size; l++)
	  dydx[j] += KERNEL(p[k], p[l]) *a[l][kk] ;
	j++ ;
      }

    for (int k=0; k<size; k++) 
      for (int kk=0; kk<dim; kk++) {
	dydx[j] = 0 ;
	for (int l=0; l<size; l++) {
	  u = 0 ;
	  for(int ll=0; ll<size; ll++)
	    u += a[k][ll]*a[l][ll] ;
	  dydx[j] -= 2*KERNEL.diff(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u ;
	}
	j++ ;
      }

    for (int k=0; k<size; k++) 
      for (int kk=0; kk<dim; kk++) {
	dydx[j] = 0 ;
	for (int l=0; l<size; l++) {
	  dydx[j] += KERNEL(p[k], p[l]) *da[l][kk] ;
	  u = 0 ;
	  for (ll=0; ll<dim; ll++)
	    u+= (p[k][ll]-p[l][ll])*(dp[k][ll]-dp[l][ll]) ;
	  dydx[j] += KERNEL.diff(p[k], p[l]) *u * a[l][kk] ;
	}
	j++ ;
      }

    for (int k=0; k<size; k++) 
      for (int kk=0; kk<dim; kk++) {
	dydx[j] = 0 ;
	for (int l=0; l<size; l++) {
	  u = 0 ;
	  for(int ll=0; ll<dim; ll++)
	    u += a[k][ll]*da[l][ll] + da[k][ll]*a[l][ll] ;
	  dydx[j] -= 2*KERNEL.diff(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u ;
	  u = 0 ;
	  for(int ll=0; ll<dim; ll++)
	    u += a[k][ll]*a[l][ll] ;
	  dydx[j] -= 2*KERNEL.diff(p[k], p[l]) *(dp[k][kk]-dp[l][kk]) * u ;
	  uu = 0 ;
	  for (ll=0; ll<dim; ll++)
	    uu+= (p[k][ll]-p[l][ll])*(dp[k][ll]-dp[l][ll]) ;
	  dydx[j] -= 4*KERNEL.diff2(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u * uu ;
	}
	j++ ;
      }
  }
} ;

template <class KERNEL> struct diffLandmarkUpdate {
  diffLandmarkUpdate(){size=0; dim=0;}
  diffLandmarkUpdate(int s, int d){size = s; dim = d;}
  int size ;
  int dim ;
  void operator()(const Doub x, VecDoub_I &y, VectDoub_0 &dydx) {
    MatSet za(size, dim), zp(size, dim), dza(size, dim),dzp(size, dim) ;
    PointSet p ;
    Tangent a, dp, da;
    double u, u1, u2, uu ;
    listToDiffPointSets(y, size, dim, p,a, zp, za) ;
    int j=0 ;
    for (int k=0; k<size; k++) 
      for (int kk=0; kk<dim; kk++) {
	dydx[j] = 0 ;
	for (int l=0; l<size; l++)
	  dydx[j] += KERNEL(p[k], p[l]) *a[l][kk] ;
	j++ ;
      }

    for (int k=0; k<size; k++) 
      for (int kk=0; kk<dim; kk++) {
	dydx[j] = 0 ;
	for (int l=0; l<size; l++) {
	  u = 0 ;
	  for(int ll=0; ll<size; ll++)
	    u += a[k][ll]*a[l][ll] ;
	  dydx[j] -= 2*KERNEL.diff(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u ;
	}
	j++ ;
      }

    for(int q=0; q<size; q++)
      for(int qq=0; qq<dim; qq++) {
	for (int k=0; k<size; k++) 
	  for (int kk=0; kk<dim; kk++) {
	    dydx[j] = 0 ;
	    for (int l=0; l<size; l++) {
	      dzp(k,kk,q,qq) += KERNEL(p[k], p[l]) *za(l,kk,q,qq) ;
	      u = 0 ;
	      for (ll=0; ll<dim; ll++)
		u+= (p[k][ll]-p[l][ll])*(zp(k,ll,q,qq)- zp(l,ll,q,qq)) ;
	      dzp(k,kk,q,qq) += KERNEL.diff(p[k], p[l]) *u * a[l][kk] ;
	    }
	  }

	for (int k=0; k<size; k++) 
	  for (int kk=0; kk<dim; kk++) {
	    dydx[j] = 0 ;
	    for (int l=0; l<size; l++) {
	      u = 0 ;
	      for(int ll=0; ll<dim; ll++)
		u += a[k][ll]*za(l,ll,q,qq) + za(k,ll,q,qq)*a[l][ll] ;
	      dza(k,kk,q,qq) -= 2*KERNEL.diff(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u ;
	      u = 0 ;
	      for(int ll=0; ll<dim; ll++)
		u += a[k][ll]*a[l][ll] ;
	      dza(k,kk,q,qq) -= 2*KERNEL.diff(p[k], p[l]) *(zp(k,kk,q,qq) - zp(l,kk,q,qq)) * u ;
	      uu = 0 ;
	      for (ll=0; ll<dim; ll++)
		uu+= (p[k][ll]-p[l][ll])*(zp(k,ll,q,qq)-zp(l,ll,q,qq)) ;
	      dza(k,kk,q,qq) -= 4*KERNEL.diff2(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u * uu ;
	    }
	  }
      }
 
    for (int k=0; k<size;k++)
      for(int kk=0; kk<dim;kk++)
	for (int l=0; l<size;l++)
	  for(int ll=0; ll<dim;ll++)
	    dydx[j++] = dzp(k,kk,l,ll) ;

    for (int k=0; k<size;k++)
      for(int kk=0; kk<dim;kk++)
	for (int l=0; l<size;l++)
	  for(int ll=0; ll<dim;ll++)
	    dydx[j++] = dza(k,kk,l,ll) ;
  }
};

template<class KERNEL>
void PointSetEvolution::geodesicPointSetEvolution(const PointSet& p0, const Tangent &a0,  PointSet &p1, Tangent &a1, _real delta) {
  VecDoub_I y0 ;
  double x1 = 0, x2 = delta, atol = 1e-3, rtol = atol, h1 = delta/10, hmin = 1e-6 ;
  landmarkUpdate<KERNEL> lup(p0.size(), p0.dim()) ;
  pointSetsToList(p0, a0, y0) ;
  Output out ;
  Odeint<StepperDopr5<landmarkUpdate> > ode(y0, x1, x2, atol, rtol, h1, hmin,out,lup) ;
  ode.integrate() ;
  listToPointSets(y0, p0.size(), p0.dim(), p1, a1) ;
}

template<class KERNEL>
void PointSetEvolution::varGeodesicPointSetEvolution(const PointSet& p0, const Tangent &a0,  const Tangent &da0, Tangent &dp1, Tangent &da1, PointSet &p1, _real delta) {
  VecDoub_I y0 ;
  double x1 = 0, x2 = delta, atol = 1e-3, rtol = atol, h1 = delta/10, hmin = 1e-6 ;
  varLandmarkUpdate<KERNEL> lup(p0.size(), p0.dim()) ;
  Tangent a1, dp0 ;
  dp0.al(p0.size(), p0.dim()) ;
  for(int k=0; k<p0.size(); k++)
    for (int kk=0; kk<p0.dim(); kk++)
      dp0[k][kk] = 0 ;
  varPointSetsToList(p0, a0, dp0, da0, y0) ;
  Output out ;
  Odeint<StepperDopr5<varLandmarkUpdate> > ode(y0, x1, x2, atol, rtol, h1, hmin,out,lup) ;
  ode.integrate() ;
  listToVarPointSets(y0, p0.size(), p0.dim(), p1, a1, dp1, da1) ;
}

template<class KERNEL>
void PointSetEvolution::diffGeodesicPointSetEvolution(const PointSet& p0, const Tangent &a0,  MatSet &zp, MatSet &za, PointSet &p1, _real delta) {
  VecDoub_I y0 ;
  double x1 = 0, x2 = delta, atol = 1e-3, rtol = atol, h1 = delta/10, hmin = 1e-6 ;
  DiffLandmarkUpdate<KERNEL> lup(p0.size(), p0.dim()) ;
  Tangent a1  ;
  zp.zeros(p0.size(), p0.dim()) ;
  za.eye(p0.size(), p0.dim()) ;
  varPointSetsToList(p0, a0, zp, za, y0) ;
  Output out ;
  Odeint<StepperDopr5<varLandmarkUpdate> > ode(y0, x1, x2, atol, rtol, h1, hmin,out,lup) ;
  ode.integrate() ;
  listToDiffPointSets(y0, p0.size(), p0.dim(), p1, a1, zp, za) ;
}

