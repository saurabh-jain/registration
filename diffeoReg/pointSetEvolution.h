/**
   pointSetEvolution.h
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
#ifndef _PSEVOL_
#define _PSEVOL_
#include "pointSetMatching.h"
#include "pointSetFunctors.h"
#include "optimF.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv.h>

/**
Output for diffeq
*/
class DiffEqOutput: public std::vector<std::vector<double> >
{
public:
  DiffEqOutput(){_n=0; _dim =0;}
  void al(int n, int dim){
    _n = n ;
    _dim = dim ;
    resize(n+1) ;
    for (int k=0; k<=n; k++)
      (*this)[k].resize(dim + 1) ;
  }
  void al(int dim){ al(_n, dim) ;}
  int dim(){ return _dim ;}
  int counts(){return _n;}
  void setCounts(int n){ _n = n;}
  double getTime(int k) {return (*this)[k][0] ;} 
  void setTime(int k, double t) {(*this)[k][0] = t;} 
  double getState(int k, int l) {return (*this)[k][l+1] ;} 
  void setState(int k, int l, double t) {(*this)[k][l+1] = t;}
  void setState(int k, const double *y){
    for(int l=0; l<_dim; l++)
      (*this)[k][l+1] = y[l] ;
  }
private:
  int _dim ;
  int _n;
} ; 

/**
Class providing diffeomorphic evolution functions in image space
*/
template <class KERNEL>
class PointSetEvolution: public PointSetMatching
{
 public:
  using PointSetMatching::param ;
  PointSetEvolution(param_matching &par) : PointSetMatching(par){};
  PointSetEvolution(char *file, int argc, char** argv) : PointSetMatching(file, argc, argv){};
  PointSetEvolution(char *file, int k) : PointSetMatching(file, k){};
  // Template, initial image momentum, 

  void initKernel() {
    _K.setWidth(param.sigmaKernel) ;
  }
    
  PointSetEvolution(){meta = false;} 

  bool meta ;
  void doMetamorphosis(){meta=true;}


  class _kernel {
  public:
    _kernel(PointSetEvolution *pse, const PointSet &p){ 
      _pse = pse ; 
      _p.copy(p);}
    void operator()(const Tangent &a, Tangent &b) const{
      _pse->kernel(_p, a, b) ;}
  private:
    PointSet _p;
    PointSetEvolution *_pse ;
  };

  void makeKernelMatrix(const PointSet &p, Matrix &M) const {
    M.resize(p.size(), p.size()) ;
    for(int k=0; k<p.size(); k++) {
      M(k,k) = _K(p[k], p[k]) ;
      for(int l=k+1; l<=p.size(); l++) {
	M(k,l) = _K(p[k], p[l]) ;
	M(l,k) = M(k,l) ;
      }
    }
  }

  class _kernelMat {
  public:
    _kernelMat(PointSetEvolution &pse, const PointSet &p){ pse.makeKernelMatrix(p, _M) ;}
    void operator()(const Tangent &a, Tangent &b) const {
      b.zeros(a.size(), a.dim()) ;
      for(int k=0; k<a.size(); k++)
	for(int l=0; l<a.size(); l++)
	  for(int kk=0; kk<a.dim(); kk++)
	    b[k][kk] += _M(k,l)*a[l][kk]  ;
    }
  private:
    Matrix _M;
  };

  void kernel(const PointSet &p, const Tangent &a, Tangent &Ka) const 
  {
    double u ;
    Ka.zeros(a.size(), a.dim()) ;
    for(unsigned int k=0; k<p.size(); k++) 
      for(unsigned int l=0; l<p.size(); l++) {
	u = _K(p[k], p[l]) ;
	for(int kk=0; kk<p.dim(); kk++) 
	  Ka[k][kk] +=  u * a[l][kk] ;
      }
  }

  void inverseKernel(const PointSet &p, const Tangent &Ka, Tangent &a)
  {
    _kernel K(this, p) ;
    PointSetScp pscp ;
    linearCG<Tangent,PointSetScp, _kernel> invK ;
    Tangent a0 ;
    a0.zeros(Ka.size(), Ka.dim()) ;
    invK(a0, a, Ka, 100, .001, 0, K,pscp) ;
  } 

  void inverseKernelMat(const PointSet&p, const Tangent &Ka, Tangent &a) const 
  {
    _kernelMat K(*this, p) ;
    PointSetScp pscp ;
    //    cout << "inverse kernel mat" << endl ;
    linearCG<Tangent,PointSetScp,_kernelMat> invK ;
    Tangent a0 ;
    a0.zeros(Ka.size(), Ka.dim()) ;
    invK(a0, a, Ka, 100, .001, 0, K,pscp) ;
  } 


  struct landmarkUpdate {
    PointSetEvolution *_pse ;
    landmarkUpdate(PointSetEvolution *pse){
      //      cout << "in l. up." << endl ;
      _pse = pse; 
      size = _pse->Template.size(); 
      dim = _pse->Template.dim(); 
      meta = _pse->meta ;
      //      cout << "set width" << endl ;
      _pse->_K.setWidth(_pse->param.sigmaKernel);
      if (meta)
	sigma = _pse->param.sigma ;
      else 
	sigma = 0 ;
    }
    landmarkUpdate(int s, int d, double sig){size = s; dim = d; sigma=sig;}
    landmarkUpdate(){size=0; dim=0; sigma=0; meta = false;}
    int size ;
    int dim ;
    double sigma ;
    bool meta ;
    void operator()(const double x, const double *y, double *dydx) {
      PointSet p ;
      Tangent a;
      listToPointSets(y, size, dim, p,a) ;
      int j=0 ;
      //      dydx.resize(y.size()) ;
      for (int k=0; k<size; k++) 
	for (int kk=0; kk<dim; kk++) {
	  dydx[j] = sigma * a[k][kk] ;
	  for (int l=0; l<size; l++)
	    dydx[j] += _pse->_K(p[k], p[l]) *a[l][kk] ;
	  j++ ;
	}
      
      double u ;
      for (int k=0; k<size; k++) 
	for (int kk=0; kk<dim; kk++) {
	  dydx[j] = 0 ;
	  for (int l=0; l<size; l++) {
	    u = 0 ;
	    for(int ll=0; ll<dim; ll++)
	      u += a[k][ll]*a[l][ll] ;
	    dydx[j] -= 2*_pse->_K.diff(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u ;
	  }
	  j++ ;
	}
    }
  };

  static int landmarkUpdate_gslCall(double t, const double *y, double *f, void *param) {
    landmarkUpdate* foo = (landmarkUpdate *) param ;
    (*foo)(t, y, f) ;
      return GSL_SUCCESS ;
  }

  struct landmarkUpdateExtended {
    PointSetEvolution *_pse ;
    landmarkUpdateExtended(PointSetEvolution *pse, int sz2){
      //      cout << "in l. up." << endl ;
      _pse = pse; 
      size = _pse->Template.size(); 
      dim = _pse->Template.dim(); 
      size2 = sz2 ;
      //      cout << "set width" << endl ;
      _pse->_K.setWidth(_pse->param.sigmaKernel);
    }
    landmarkUpdateExtended(int s, int d){size = s; dim = d;}
    landmarkUpdateExtended(){size=0; dim=0; size2 = 0 ;}
    int size ;
    int size2 ;
    int dim ;
    void operator()(const double x,  const double *y, double *dydx) {
      PointSet p, q ;
      Tangent a;
      listToPointSetsExtended(y, size, dim, size2, p, a, q) ;
      int j=0 ;
      //      dydx.resize(y.size()) ;
      for (int k=0; k<size; k++) 
	for (int kk=0; kk<dim; kk++) {
	  dydx[j] = 0 ;
	  for (int l=0; l<size; l++)
	    dydx[j] += _pse->_K(p[k], p[l]) *a[l][kk] ;
	  j++ ;
	}
      
      double u ;
      for (int k=0; k<size; k++) 
	for (int kk=0; kk<dim; kk++) {
	  dydx[j] = 0 ;
	  for (int l=0; l<size; l++) {
	    u = 0 ;
	    for(int ll=0; ll<dim; ll++)
	      u += a[k][ll]*a[l][ll] ;
	    dydx[j] -= 2*_pse->_K.diff(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u ;
	  }
	  j++ ;
	}
      for (int k=0; k<q.size(); k++) 
	for (int kk=0; kk<dim; kk++) {
	  dydx[j] = 0 ;
	  for (int l=0; l<size; l++)
	    dydx[j] += _pse->_K(q[k], p[l]) *a[l][kk] ;
	  j++ ;
	}
      
    }
  };

  static int landmarkUpdateExtended_gslCall(double t, const double *y, double *f, void *param) {
    landmarkUpdateExtended* foo = (landmarkUpdateExtended *) param ;
    (*foo)(t, y, f) ;
      return GSL_SUCCESS ;
  }

  void geodesicPointSetEvolution(const PointSet& p0, const Tangent &a0,  PointSet &p1, Tangent &a1, _real delta) {
    DiffEqOutput out ;
    geodesicPointSetEvolution(p0, a0,  p1, a1, delta, out) ;
  }

  void geodesicPointSetEvolution(const PointSet& p0, const Tangent &a0,  PointSet &p1, Tangent &a1, _real delta, DiffEqOutput &out) {
    landmarkUpdate lup(this) ; //(p0.size(), p0.dim(), param.sigmaKernel) ;
    //    cout << "list" << endl; 
    int dimension = 2*lup.size*lup.dim ;
    double * y = new double[dimension];
    pointSetsToList(p0, a0, y) ;
    const gsl_odeiv_step_type * T  = gsl_odeiv_step_rk4;
    gsl_odeiv_step * s = gsl_odeiv_step_alloc (T, dimension);
    gsl_odeiv_control * c = gsl_odeiv_control_y_new (1e-6, 0.0);
    gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc (dimension);
    int (*fptr)(double, const double*, double *, void*) = &PointSetEvolution<KERNEL>::landmarkUpdate_gslCall ;
    gsl_odeiv_system sys = {fptr, NULL, dimension, &lup};
    double t=0.0; 
    double h=1e-6 ;
    int Nsave = out.counts() ;
    if (Nsave > 0) {
      out.al(dimension) ;
      out.setTime(0, 0) ;
      out.setState(0, y) ;
      double delta0 = 0 ;
      for(int kk=1; kk<=Nsave; kk++) {
	delta0 += delta / Nsave ;
	while (t<delta)
	  gsl_odeiv_evolve_apply (e, c, s, &sys, &t, delta, &h, y);
	out.setTime(kk, t) ;
	out.setState(kk, y) ;
      }
    }
    else {
      //      cout << "No Output" << endl ;
	while (t<delta)
	  gsl_odeiv_evolve_apply (e, c, s, &sys, &t, delta, &h, y);
    }    
     
    listToPointSets(y, p0.size(), p0.dim(), p1, a1) ;
    gsl_odeiv_evolve_free (e);
    gsl_odeiv_control_free (c);
    gsl_odeiv_step_free (s);

    delete[] y ;
  }
 

  void geodesicPointSetEvolution(const PointSet& p0, const Tangent &a0,  const PointSet &q0, PointSet &p1, Tangent &a1, PointSet &q1, _real delta, DiffEqOutput &out) {
    landmarkUpdateExtended lup(this, q0.size()) ; //(p0.size(), p0.dim(), param.sigmaKernel) ;
    //    cout << "list" << endl; 
    int dimension = (2*p0.size()+q0.size())*p0.dim() ;
    double * y = new double[dimension];
    pointSetsExtendedToList(p0, a0, q0, y) ;
    const gsl_odeiv_step_type * T  = gsl_odeiv_step_rk4;
    gsl_odeiv_step * s = gsl_odeiv_step_alloc (T, dimension);
    gsl_odeiv_control * c = gsl_odeiv_control_y_new (1e-6, 0.0);
    gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc (dimension);
    int (*fptr)(double, const double*, double *, void*) = &PointSetEvolution<KERNEL>::landmarkUpdateExtended_gslCall ;
    gsl_odeiv_system sys = {fptr, NULL, dimension, &lup};
    double t=0.0; 
    double h=1e-6 ;
    int Nsave = out.counts() ;
    if (Nsave > 0) {
      out.al(dimension) ;
      out.setTime(0, 0) ;
      out.setState(0, y) ;
      double delta0 = 0 ;
      for(int kk=1; kk<=Nsave; kk++) {
	delta0 += delta / Nsave ;
	while (t<delta0)
	  gsl_odeiv_evolve_apply (e, c, s, &sys, &t, delta, &h, y);
	out.setTime(kk, t) ;
	out.setState(kk, y) ;
      }
    }
    else {
	while (t<delta)
	  gsl_odeiv_evolve_apply (e, c, s, &sys, &t, delta, &h, y);
    }    
    listToPointSetsExtended(y, p0.size(), p0.dim(), q0.size(), p1, a1, q1) ;
    gsl_odeiv_evolve_free (e);
    gsl_odeiv_control_free (c);
    gsl_odeiv_step_free (s);

    delete[] y ;
  }
  
  struct varLandmarkUpdate {
    PointSetEvolution *_pse ;
    varLandmarkUpdate(PointSetEvolution *pse){
      _pse = pse; 
      size = _pse->Template.size(); 
      dim = _pse->Template.dim(); 
      _pse->_K.setWidth(_pse->param.sigmaKernel);
    }
    varLandmarkUpdate(){size=0; dim=0;}
    varLandmarkUpdate(int s, int d){size = s; dim = d;}
    //    varLandmarkUpdate(int s, int d, double sigma){size = s; dim = d; _pse->_K.setWidth(sigma);}
    int size ;
    int dim ;
    void operator()(const double x, const double* y, double* dydx) {
      PointSet p ;
      Tangent a, dp, da;
      double u, uu ;
      //      dydx.resize(y.size()) ;
      listToVarPointSets(y, size, dim, p,a, dp, da) ;
      int j=0 ;
      for (int k=0; k<size; k++) 
	for (int kk=0; kk<dim; kk++) {
	  dydx[j] = 0 ;
	  for (int l=0; l<size; l++)
	    dydx[j] += _pse->_K(p[k], p[l]) *a[l][kk] ;
	  j++ ;
	}
      
      for (int k=0; k<size; k++) 
	for (int kk=0; kk<dim; kk++) {
	  dydx[j] = 0 ;
	  for (int l=0; l<size; l++) {
	    u = 0 ;
	    for(int ll=0; ll<dim; ll++)
	      u += a[k][ll]*a[l][ll] ;
	    dydx[j] -= 2*_pse->_K.diff(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u ;
	  }
	  j++ ;
	}
      
      for (int k=0; k<size; k++) 
	for (int kk=0; kk<dim; kk++) {
	  dydx[j] = 0 ;
	  for (int l=0; l<size; l++) {
	    dydx[j] += _pse->_K(p[k], p[l]) *da[l][kk] ;
	    u = 0 ;
	    for (int ll=0; ll<dim; ll++)
	      u+= (p[k][ll]-p[l][ll])*(dp[k][ll]-dp[l][ll]) ;
	    dydx[j] += 2*_pse->_K.diff(p[k], p[l]) *u * a[l][kk] ;
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
	    dydx[j] -= 2*_pse->_K.diff(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u ;
	  u = 0 ;
	  for(int ll=0; ll<dim; ll++)
	    u += a[k][ll]*a[l][ll] ;
	  dydx[j] -= 2*_pse->_K.diff(p[k], p[l]) *(dp[k][kk]-dp[l][kk]) * u ;
	  uu = 0 ;
	  for (int ll=0; ll<dim; ll++)
	    uu+= (p[k][ll]-p[l][ll])*(dp[k][ll]-dp[l][ll]) ;
	  dydx[j] -= 4*_pse->_K.diff2(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u * uu ;
	}
	j++ ;
	}
    }
  } ;

  static int varLandmarkUpdate_gslCall(double t, const double *y, double *f, void *param) {
    varLandmarkUpdate* foo = (varLandmarkUpdate *) param ;
    (*foo)(t, y, f) ;
      return GSL_SUCCESS ;
  }



  void varGeodesicPointSetEvolution(const PointSet& p0, const Tangent &a0,  const Tangent &da0, Tangent &dp1, Tangent &da1, PointSet &p1, _real delta) {
    Tangent a1, dp0 ;
    dp0.al(p0.size(), p0.dim()) ;
    for(int k=0; k<p0.size(); k++)
      for (int kk=0; kk<p0.dim(); kk++)
	dp0[k][kk] = 0 ;
    varLandmarkUpdate lup(this) ;//p0.size(), p0.dim(), param.sigmaKernel) ;
    int dimension = 4*p0.size()*p0.dim() ;
    double * y = new double[dimension];
    varPointSetsToList(p0, a0, dp0, da0, y) ;
    const gsl_odeiv_step_type * T  = gsl_odeiv_step_rk4;
    gsl_odeiv_step * s = gsl_odeiv_step_alloc (T, dimension);
    gsl_odeiv_control * c = gsl_odeiv_control_y_new (1e-6, 0.0);
    gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc (dimension);
    int (*fptr)(double, const double*, double *, void*) = &PointSetEvolution<KERNEL>::varLandmarkUpdate_gslCall ;
    gsl_odeiv_system sys = {fptr, NULL, dimension, &lup};
    double t=0.0; 
    double h=1e-6 ;
    while (t<delta) {
      gsl_odeiv_evolve_apply (e, c, s, &sys, &t, delta, &h, y);
    }
    listToVarPointSets(y, p0.size(), p0.dim(), p1, a1, dp1, da1) ;
    gsl_odeiv_evolve_free (e);
    gsl_odeiv_control_free (c);
    gsl_odeiv_step_free (s);

    delete[] y ;
  }

  struct dualVarLandmarkUpdate {
    PointSetEvolution *_pse ;
    std::vector<double> _x ;
    std::vector<PointSet> _p ;
    std::vector<Tangent> _a ;
    void interpolateForwardEvolution(double x, PointSet &p, Tangent &a) {
      double z = ((x-_x[0])/(_x[_x.size()-1]-_x[0])) * _a.size() ;
      int u = (int) floor(z) ;
      double r ;
      p.zeros(size, dim) ;
      a.zeros(size, dim) ;
      if (u < 0) {
	a.copy(_a[0]) ;
	p.copy(_p[0]) ;
      }
      else if (u < (int) _a.size()-1) {
	r = z - u ;
	for (int k=0; k<size; k++)
	  for(int kk=0; kk<dim;kk++) {
	    a[k][kk] = (1-r)*_a[u][k][kk] + r * _a[u+1][k][kk] ;
	    p[k][kk] = (1-r)*_p[u][k][kk] + r * _p[u+1][k][kk] ;
	  }
      }
      else {
	a.copy(_a[_a.size()-1]) ;
	p.copy(_p[_a.size()-1]) ;
      }
      //      cout << "Interpolate: " << x << " " << u << endl ;
      //      cout << p << endl ;

    }

    dualVarLandmarkUpdate(PointSetEvolution *pse, DiffEqOutput &out0){
      _pse = pse; 
      size = _pse->Template.size(); 
      dim = _pse->Template.dim(); 
      meta = _pse->meta ;
      _pse->_K.setWidth(_pse->param.sigmaKernel);
      _x.resize(out0.size()) ;
      _p.resize(out0.size()) ;
      _a.resize(out0.size()) ;
      if (meta)
	sigma = _pse->param.sigma ;
      else 
	sigma = 0 ;

      int j ;
      for(unsigned int i=0; i<out0.size(); i++) {
	_x[i] = out0.getTime(i) ;
	//	cout << i << " TTime: " << _x[i] << endl ;
	_p[i].al(size, dim) ;
	_a[i].al(size, dim) ;
	j= 0 ;
	for(int k=0; k< size; k++)
	  for (int kk=0; kk<dim; kk++)
	    _p[i][k][kk] = out0.getState(i, j++) ;  
	for(int k=0; k<size; k++)
	  for (int kk=0; kk<dim; kk++)
	    _a[i][k][kk] = out0.getState(i, j++) ;  
      }

/*       cout << "initializing" << endl ; */
/*       for(int k=0; k<_p.size(); k++) */
/* 	cout << _a[k] << endl << endl ; */
    }

    dualVarLandmarkUpdate(){size=0; dim=0;sigma =0;meta=false;}
    dualVarLandmarkUpdate(int s, int d){size = s; dim = d;sigma=0;meta=false;}
    //    varLandmarkUpdate(int s, int d, double sigma){size = s; dim = d; _pse->_K.setWidth(sigma);}
    int size ;
    int dim ;
    double sigma ;
    bool meta ;
    void operator()(const double x, const double* y, double *dydx) {
      PointSet p ;
      Tangent a, dp, da, zp, za;
      double u, uu ;
      //      cout << "dual var " << x  << endl ;
      //     dydx.resize(y.size()) ;
      zp.al(size, dim) ;
      za.al(size, dim) ;
      //      cout << "interpolate" << endl ;
      interpolateForwardEvolution(_x[_x.size()-1] + _x[0]-x, p, a) ;
      //      cout << "list" << endl ;
      listToDualVarPointSets(y, size, dim, dp, da) ;
      int j=0 ;

      //      cout << "computation" << endl ;
      for (int k=0; k<size; k++) 
	for (int kk=0; kk<dim; kk++) {
	  zp[k][kk] = 0 ;
	  za[k][kk] = sigma * dp[k][kk] ;
	  for (int l=0; l<size; l++) {
	    za[k][kk] += _pse->_K(p[k], p[l]) *dp[l][kk] ;
	    u = 0 ;
	    for (int ll=0; ll<dim; ll++)
	      u+= a[l][ll]*dp[k][ll] + a[k][ll]*dp[l][ll] ;
	    zp[k][kk] += 2*_pse->_K.diff(p[k], p[l]) *u * (p[k][kk]-p[l][kk]) ;
	  }
	  for (int l=0; l<size; l++) {
	    uu = 0 ;
	    for(int ll=0; ll<dim; ll++)
	      uu += (p[k][ll] - p[l][ll])*(da[k][ll] - da[l][ll]) ;
	    za[k][kk] -= 2*_pse->_K.diff(p[k], p[l]) * a[l][kk] * uu ;
	    u = 0 ;
	    for(int ll=0; ll<dim; ll++)
	      u += a[k][ll]*a[l][ll] ;
	    zp[k][kk] -= 2*_pse->_K.diff(p[k], p[l]) *(da[k][kk]-da[l][kk]) * u ;
	    //	    uu = 0 ;
	    //	    for (int ll=0; ll<dim; ll++)
	    //	      uu+= (p[k][ll]-p[l][ll])*(da[k][ll]-da[l][ll]) ;
	    zp[k][kk] -= 4*_pse->_K.diff2(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u * uu ;
	  }
	}
      //      cout << "storing" << endl ;
      for(int k=0; k<size; k++)
	for(int kk=0; kk<dim; kk++)
	  dydx[j++] = zp[k][kk] ;
      for(int k=0; k<size; k++)
	for(int kk=0; kk<dim; kk++)
	  dydx[j++] = za[k][kk] ;
      //      cout << zp << endl << endl << za << endl ;
      //      cout << "done update" << endl ;
    }
  } ;

  static int dualVarLandmarkUpdate_gslCall(double t, const double *y, double *f, void *param) {
    dualVarLandmarkUpdate* foo = (dualVarLandmarkUpdate *) param ;
    (*foo)(t, y, f) ;
      return GSL_SUCCESS ;
  }

  void dualVarGeodesicPointSetEvolution(const PointSet& p0, const Tangent &a0,  const Tangent &dp0, Tangent &dp1, Tangent &da1,  _real delta) {
    landmarkUpdate lup0(this) ; //(p0.size(), p0.dim(), param.sigmaKernel) ;
    int dimension = 2*lup0.size*lup0.dim ;
    double * y = new double[dimension];
    pointSetsToList(p0, a0, y) ;
    const gsl_odeiv_step_type * T  = gsl_odeiv_step_rk4;
    gsl_odeiv_step * s = gsl_odeiv_step_alloc (T, dimension);
    gsl_odeiv_control * c = gsl_odeiv_control_y_new (1e-6, 0.0);
    gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc (dimension);
    gsl_odeiv_system sys = {this->landmarkUpdate_gslCall, NULL, dimension, &lup0};
    double t=0.0; 
    double h=1e-6 ;
    int Nsave =20 ;
    DiffEqOutput out0 ;
    out0.al(Nsave, dimension) ;
    out0.setTime(0, 0) ;
    out0.setState(0, y) ;
    double delta0 = 0;
    for (int kk=1; kk<=Nsave; kk++) {
      delta0 += delta/Nsave ;
      while (t<delta0) {
	gsl_odeiv_evolve_apply (e, c, s, &sys, &t, delta0, &h, y);
      }
      //      cout << kk << " time =" << t << endl ;
      out0.setTime(kk, t) ;
      out0.setState(kk, y) ;
    }
    gsl_odeiv_evolve_free (e);
    gsl_odeiv_control_free (c);
    gsl_odeiv_step_free (s);
    delete[] y ;


    //    cout << "preparing dual" << endl ;
    dualVarLandmarkUpdate lup(this, out0) ;//p0.size(), p0.dim(), param.sigmaKernel) ;
    Tangent a1, da0 ;
    da0.al(p0.size(), p0.dim()) ;
    for(unsigned int k=0; k<p0.size(); k++)
      for (int kk=0; kk<p0.dim(); kk++)
	da0[k][kk] = 0 ;
    dimension = 2*dp0.size()*dp0.dim() ;
    y = new double[dimension];
    dualVarPointSetsToList(dp0, da0, y) ;
    //T  = gsl_odeiv_step_rk8pd;
    s = gsl_odeiv_step_alloc (T, dimension);
    c = gsl_odeiv_control_y_new (1e-6, 0.0);
    e = gsl_odeiv_evolve_alloc (dimension);
    int (*fptr2)(double, const double*, double *, void*) = &PointSetEvolution<KERNEL>::dualVarLandmarkUpdate_gslCall ;
    gsl_odeiv_system sys2 = {fptr2, NULL, dimension, &lup};
    t=0.0; 
    while (t<delta) {
      gsl_odeiv_evolve_apply (e, c, s, &sys2, &t, delta, &h, y);
    }
    listToDualVarPointSets(y, p0.size(), p0.dim(), dp1, da1) ;
    gsl_odeiv_evolve_free (e);
    gsl_odeiv_control_free (c);
    gsl_odeiv_step_free (s);

    delete[] y ;
  }

  struct diffLandmarkUpdate {
    PointSetEvolution *_pse ;
    diffLandmarkUpdate(PointSetEvolution *pse){
      _pse = pse; 
      size = _pse->Template.size(); 
      dim = _pse->Template.dim();
      _pse->_K.setWidth(_pse->param.sigmaKernel);
    }
    diffLandmarkUpdate(){size=0; dim=0;}
    diffLandmarkUpdate(int s, int d){size = s; dim = d;}
    //    diffLandmarkUpdate(int s, int d, double sigma){size = s; dim = d; _pse->_K.setWidth(sigma);}
    int size ;
    int dim ;
    void operator()(const double x, double *y, double *dydx) {
      MatSet za(size, dim), zp(size, dim), dza(size, dim),dzp(size, dim) ;
      PointSet p ;
      Tangent a, dp, da;
      double u, uu ;
      listToDiffPointSets(y, size, dim, p,a, zp, za) ;
      int j=0 ;
      //      dydx.resize(y.size()) ;
      for (int k=0; k<size; k++) 
	for (int kk=0; kk<dim; kk++) {
	  dydx[j] = 0 ;
	  for (int l=0; l<size; l++)
	    dydx[j] += _pse->_K(p[k], p[l]) *a[l][kk] ;
	  j++ ;
	}

      for (int k=0; k<size; k++) 
	for (int kk=0; kk<dim; kk++) {
	  dydx[j] = 0 ;
	  for (int l=0; l<size; l++) {
	    u = 0 ;
	    for(int ll=0; ll<dim; ll++)
	      u += a[k][ll]*a[l][ll] ;
	    dydx[j] -= 2*_pse->_K.diff(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u ;
	  }
	  j++ ;
	}

      for(int q=0; q<size; q++)
	for(int qq=0; qq<dim; qq++) {
	  for (int k=0; k<size; k++) 
	    for (int kk=0; kk<dim; kk++) {
	      for (int l=0; l<size; l++) {
		dzp(k,kk,q,qq) += _pse->_K(p[k], p[l]) *za(l,kk,q,qq) ;
		u = 0 ;
		for (int ll=0; ll<dim; ll++)
		  u+= (p[k][ll]-p[l][ll])*(zp(k,ll,q,qq)- zp(l,ll,q,qq)) ;
		dzp(k,kk,q,qq) += 2*_pse->_K.diff(p[k], p[l]) *u * a[l][kk] ;
	      }
	    }
	  
	  for (int k=0; k<size; k++) 
	    for (int kk=0; kk<dim; kk++) {
	      for (int l=0; l<size; l++) {
		u = 0 ;
		for(int ll=0; ll<dim; ll++)
		  u += a[k][ll]*za(l,ll,q,qq) + za(k,ll,q,qq)*a[l][ll] ;
		dza(k,kk,q,qq) -= 2*_pse->_K.diff(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u ;
		u = 0 ;
		for(int ll=0; ll<dim; ll++)
		  u += a[k][ll]*a[l][ll] ;
		dza(k,kk,q,qq) -= 2*_pse->_K.diff(p[k], p[l]) *(zp(k,kk,q,qq) - zp(l,kk,q,qq)) * u ;
		uu = 0 ;
		for (int ll=0; ll<dim; ll++)
		  uu+= (p[k][ll]-p[l][ll])*(zp(k,ll,q,qq)-zp(l,ll,q,qq)) ;
		dza(k,kk,q,qq) -= 4*_pse->_K.diff2(p[k], p[l]) *(p[k][kk]-p[l][kk]) * u * uu ;
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

  static int diffLandmarkUpdate_gslCall(double t, const double *y, double *f, void *param) {
    diffLandmarkUpdate* foo = (diffLandmarkUpdate *) param ;
    (*foo)(t, y, f) ;
      return GSL_SUCCESS ;
  }


  void diffGeodesicPointSetEvolution(const PointSet& p0, const Tangent &a0,  MatSet &zp, MatSet &za, PointSet &p1, _real delta) {
    diffLandmarkUpdate lup(this); //(p0.size(), p0.dim(), param.sigmaKernel) ;
    Tangent a1  ;
    zp.zeros(p0.size(), p0.dim()) ;
    za.eye(p0.size(), p0.dim()) ;
    //    cout << "list" << endl ;
    int dimension = 2*p0.size()*p0.dim()*(1 + p0.size()*p0.dim()) ;
    double * y = new double[dimension];
    diffPointSetsToList(p0, a0, zp, za, y) ;
    const gsl_odeiv_step_type * T  = gsl_odeiv_step_rk4;
    gsl_odeiv_step * s = gsl_odeiv_step_alloc (T, dimension);
    gsl_odeiv_control * c = gsl_odeiv_control_y_new (1e-6, 0.0);
    gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc (dimension);
    int (*fptr)(double, const double*, double *, void*) = &PointSetEvolution<KERNEL>::diffLandmarkUpdate_gslCall ;
    gsl_odeiv_system sys = {fptr, NULL, dimension, &lup};
    double t=0.0; 
    double h=1e-6 ;
    while (t<delta) {
      gsl_odeiv_evolve_apply (e, c, s, &sys, &t, delta, &h, y);
    }
    listToDiffPointSets(y, p0.size(), p0.dim(), p1, a1, zp, za) ;
    gsl_odeiv_evolve_free (e);
    gsl_odeiv_control_free (c);
    gsl_odeiv_step_free (s);

    delete[] y ;
  }
  void parallelPointSetTransport(const PointSet &p0, const Tangent &a0, const Tangent &b0, 
				 PointSet &p2, Tangent &ac) ;
  ~PointSetEvolution(){}

private:
  KERNEL _K ;
};


#endif
