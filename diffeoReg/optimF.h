/**
   optimF.h
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
#ifndef _GRAD_DESC_
#define _GRAD_DESC_
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>



  /** Linear conjugate gradient
      requires the implementation of linearCGFunction (Linear transformation), scalProd (dot product)
  */
template <class OBJECT_TYPE, class scalProd, class linearOperator> class linearCG {
 public:
  double operator()(const OBJECT_TYPE &x0, OBJECT_TYPE &x, const OBJECT_TYPE &b, int nbStep, double error, bool verb, linearOperator &A, scalProd &scp) {
    OBJECT_TYPE r,p,q, foo ;
    double mu ;
    
    r.copy(b) ;
    A(x0, foo) ;
    r -= foo ;
    
    //conjugate gradient loop
    
    double energy=-1, energy0 = scp(x0, foo)/2 - scp(b,x0) ;
    double energyOld = energy0 ;
    double muOld = 0, alpha, beta, var, var0 ;

    x.copy(x0) ;
    p.copy(r) ;
    muOld = scp(r,r) ;
    mu = muOld ;
    A(p, q) ;
    alpha = mu /(scp(p, q) + 1e-10) ;
    foo.copy(p) ;
    foo *= alpha ;
    x += foo ;
    foo.copy(q) ;
    foo *= alpha ;
    r -= foo ;
    var0 = alpha * mu / 2 ; //scp(p,r) ;
    energy = energyOld - var0 ;
    //    A(x, foo) ;
    //    energyTest = scp(x, foo)/2 - scp(b,x) ;
    energyOld = energy ;
    if (verb)
      cout << " conjugate gradient: initial variation = " << var0 << endl ;

    if (sqrt(mu) > error)    
      for (int ss = 0; ss <nbStep; ss++) {
	mu = scp(r,r) ;
	beta = mu/muOld ;
	p *= beta ;
	p += r ;
	
	A(p, q) ;
	alpha = mu /(scp(p, q) + 1e-10) ;
	foo.copy(p) ;
	foo *= alpha ;
	x += foo ;
	foo.copy(q) ;
	foo *= alpha ;
	r -= foo ;
	
	muOld = mu ;
	var = alpha * mu / 2 ; //scp(p,r) ;
	energy = energyOld - var ;
	//A(x, foo) ;
	//energyTest = scp(x, foo)/2 - scp(b,x) ;
	//var2 = energyOld - energyTest ;
	energyOld = energy ;
	if (verb)
	  cout << "step " << ss << " conjugate gradient: variation = " << var << endl ;

	//	var = alpha * scp(p,r) ;
	//	energy = energyOld - var ;
	
	//	if (verb)
	//  cout << "step " << ss << " conjugate gradient: " << energy0 << " " << energy << " mu=" << mu << " variation =" << var << " alpha*mu = " << alpha*mu << endl ;
	if (sqrt(mu) < error || abs(var/var0) < error )
	  break ;
	energyOld = energy ;
      }
    
    cout << " conjugate gradient: Total variation: " << energy0 - energy << endl ;
    return energy ;
  }
} ;

template<class OBJECT_TYPE, class TAN_TYPE>
class optimFunBase {
public:
  double gradNorm ;
  virtual double computeGrad(OBJECT_TYPE &x, TAN_TYPE &grad) {cout << "computeGradient is not implemented" << endl; exit(1) ;}
  virtual double objectiveFun(OBJECT_TYPE &x) {cout << "objectiveFun is not implemented" << endl; exit(1) ;}
  virtual void startOfProcedure(OBJECT_TYPE &x) {;}
  virtual void endOfProcedure(OBJECT_TYPE &x) {;}
  virtual bool stopProcedure(){ return false;}
  virtual void startOfIteration(OBJECT_TYPE &x) {;}
  virtual double endOfIteration(OBJECT_TYPE &x) {return -1 ;}
  virtual double epsBound(){return 1e100;}
  virtual double update(OBJECT_TYPE &x, TAN_TYPE& grad, double eps, OBJECT_TYPE &res) {
      res.copy(grad) ;
      res *= -eps ;
      res += x ;
      double ener = objectiveFun(res) ; 
      return ener;
  }
  virtual  ~optimFunBase() {};
};




/**
A few optimization components.
*/
template<class OBJECT_TYPE, class TAN_TYPE, class _scalProd, class _optimFun> class conjGrad{
 public:
  double epsMin ;
  double eps ;
  conjGrad(){epsMin = 1e-100 ;  eps = 1 ;}
  //  virtual double linearCGFunction(OBJECT_TYPE &x, void *par){cout << "linearCGFunction is not implemented" << endl; exit(1) ;}
  //  virtual double scalProd(OBJECT_TYPE &x, OBJECT_TYPE &y){cout << "scalProd is not implemented" << endl; exit(1) ;}
  virtual ~conjGrad() {} ;

  double getStep(){return eps ;}

  /** 
      Non linear conjugate gradient
      Requires a definition of objectiveFun (Objective function), computeGrad (gradient of the objective function), scalProd (dot product)
      Optional functions: startOfProcedure,endOfProcedure, startOfIteration, endOfIteration
      *par is send to objectiveFun and computeGrad to transmit additional parameters.
      Control parameters: nb_iter: maximum number of iterations; epsIni: initial step; epsMax: maximum step; minVarEn: minimal relative variation of energy to continue
      gs: flag for golden search; verb: flag for verbose printing
  */
  double operator()(_optimFun &opt, _scalProd &scp, OBJECT_TYPE& x0, OBJECT_TYPE &x, int nb_iter, double epsIni, double epsMax, double minVarEn, bool gs, bool verb) {
    TAN_TYPE grad, oldGrad, foo, savedGrad;
    OBJECT_TYPE xtry, xtry2, xtry3 ;
    double  old_ener, old_ener0, ener=0, ener2, eps2 ;
    _real gs_ener[3], gs_eps[3] ;
    unsigned int CG_RATE = 200 ;
    int smallVarCount = 0 , svcMAX = 2 ;
    bool cg ;
    //    _optimFun opt ;

    eps = epsIni ;
    x.copy(x0) ;
    opt.startOfProcedure(x) ;
    old_ener = opt.objectiveFun(x0) ;
    old_ener0 = old_ener ; 
    if (verb)
      cout << "initial energy = " << old_ener << endl ;

    bool stopLoop ;
    int cg_count = 0;

    for(int iter=0; iter < nb_iter; iter++) {
      opt.startOfIteration(x) ;
      if (verb)
	cout <<"Iteration " << iter+1 << endl ;
      stopLoop = false;

      if (iter > 0)
	oldGrad.copy(grad) ;
      opt.gradNorm = opt.computeGrad(x, grad) ;
      if (verb)
	cout << "Gradient Norm: " << opt.gradNorm << endl ;
      cg = false ;
      if (iter > 0 && cg_count % CG_RATE != 0 && smallVarCount == 0) {
	//	_scalProd scp(x) ;
	double b ;
	foo.copy(grad) ;
	foo -= oldGrad ;
	b = scp(grad, foo)/ scp(oldGrad, oldGrad) ;
	if (b > 0) {
	  oldGrad *= b ;
	  grad -= oldGrad ;
	  cg = true ;
	  if (verb) 
	    cout << "Using CG direction" << endl ;
	}
	else if (verb == true) {
	  cout << "Negative b in CG: " << b << endl ; 
	}
      }
      //      cout << grad << endl ;



      if (eps > epsMax)
	eps = epsMax ;
      eps2 = opt.epsBound() ;
      if (eps > eps2)
	eps = eps2 ; 
      //      if (verb)
      //cout << "eps2 = " << eps2 << endl;
      ener = opt.update(x, grad, eps, xtry) ;
      if (verb) {
	cout << "ener = " << ener << " old_ener = " << old_ener << " eps = " << eps << " " << opt.gradNorm << endl ;
      }
      gs_ener[0] = old_ener ;
      gs_eps[0] = 0 ;

      if (ener > old_ener) {
	// test if correct direction of descent
	double	epsTest = eps * 1e-10 ;
	if (epsTest < epsMin)
	  epsTest = epsMin ;
	ener2 = opt.update(x, grad, epsTest,  xtry2) ;
	if (ener2 > old_ener) {
	  if (cg == false)
	    cout << "Iter " << iter << ": Bad Direction ; ener = " << ener2 << " old_ener = " << old_ener << " eps = " << epsTest << endl ;
	  //	ener2 = opt.update(x, grad, 0,  xtry2) ;
	  //cout << ener2 << endl ;
	  stopLoop = true ;
	}
      }
    
      if (stopLoop) {
	if (cg == false) {
	  break ;
	}
	else
	  cg_count = 0 ;
      }
      else{
	cg_count++ ;
	gs_ener[2] = ener ;
	gs_eps[2] = eps ;
      
	if (ener < old_ener) {
	  bool okLoop = false ;
	  int il = 0 ;
	  do {
	    ener = opt.update(x, grad, 1.5*gs_eps[2], xtry3) ;
	    if (verb) {
	      cout << "Step increase; ener = " << ener << " old_ener = " << old_ener << " eps = " << 1.5*gs_eps[2] << endl ;
	    }
	    if (ener < 0.9999999 * gs_ener[2]){
	      gs_eps[2] *= 1.5 ;
	      gs_ener[2] = ener ;
	      xtry.copy(xtry3) ;
	      okLoop = true ;
	    }
	    else 
	      okLoop = false ;
	  }
	  while (okLoop && (++il < 3) && (gs_eps[2] < epsMax) && (gs_eps[2] < eps2)) ;
/* 	  if (il == 3) */
/* 	    eps = gs_eps[2]/2 ; */
/* 	  else */
	    eps = gs_eps[2] ;
	    ener = gs_ener[2] ;
	   }

	if (!gs && iter>0) {
	  double a0 = 0.0;
	  // b0 = 5000 ;
	  //	  while ((gs_ener[2] > old_ener - a0 * gs_eps[2] * ng2 || gs_ener[2] < old_ener - b0 * gs_eps[2] * ng2) && gs_eps[2] > epsMin)  {
	  while ((gs_ener[2] > old_ener - a0 * gs_eps[2] * opt.gradNorm) && gs_eps[2] > eps*1e-10)  {
	    gs_eps[2] *= .5 ;
	    gs_ener[2] = opt.update(x, grad, gs_eps[2], xtry) ;
	    if (verb) {
	      cout << "ener = " << gs_ener[2] << " modified old_ener = "  << old_ener - a0 * gs_eps[2] * opt.gradNorm 
		   << " " <<  opt.gradNorm  << " eps = " << gs_eps[2] << endl ;
	    }
	  }
	  x.copy(xtry) ;
	  ener = gs_ener[2] ;
	  eps = 1.5*gs_eps[2] ;
	  ener2 = opt.endOfIteration(x) ;
	  if (ener2 > -1e-10)
	    ener = ener2 ;
	}
	else {
	  gs_eps[1] = eps ;
	  gs_ener[1] = ener ;
	  // find intermediate value for golden search
	  int kg=0 ;
	  if (ener > old_ener)
	    do {
	      gs_eps[2] = gs_eps[1] ;
	      gs_ener[2] = gs_ener[1] ;
	      gs_eps[1] *= .5 ;
	      gs_ener[1] = opt.update(x, grad, gs_eps[1], xtry2) ;
	      if (verb) {
		cout << "ener = " << gs_ener[1] << " old_ener = " << old_ener  << " eps = " << gs_eps[1] << endl ;
	      }
	      kg++ ;
	    }
	    while ( (gs_ener[1] >  0.999999 * old_ener || (gs_ener[1] >  0.999999 * gs_ener[2] && kg <= 5)) && gs_eps[1] > eps*1e-10) ;
	 else {
	    xtry2.copy(xtry) ;
	    /*do {
	      gs_enerTry = opt.update(x, grad, 2*gs_eps[2],  xtry3) ;
	      if (verb) {
		cout << "ener = " << gs_enerTry << " old_ener = " << old_ener  << " eps = " << gs_eps[2] << endl ;
	      }
	      if (gs_ener[2] >  0.999999 * gs_enerTry) {
		gs_eps[2] *= 2 ;
		gs_ener[2] = gs_enerTry ;
	      }
	      kg++ ;
	    }
	    while ( (gs_ener[2] >  0.999999 * gs_enerTry && kg <= 5) && gs_eps[2] < epsMax) ;*/
	  }
	
	  if (gs_ener[1] > gs_ener[0] || gs_ener[1] > gs_ener[2]) {
	    // if no intermediate is found: no golden search 
	    if (verb)
	      cout << "No golden search" << endl ;
	    eps = gs_eps[2]/0.75 ;
	    opt.update(x, grad, gs_eps[2],  xtry) ;
	    ener = gs_ener[2] ;
	    x.copy(xtry) ;
	    opt.endOfIteration(x) ;
	  }
	  else { 
	    //golden search
	    if (verb)
	      cout << "golden search" << endl ;
	    xtry.copy(xtry2) ;
	    ener = gs_ener[1] ;
	    _real delta = gs_eps[2] - gs_eps[1];// eps0 = gs_eps[2] ;
	    int bigI = 1 ;
	    if (gs_eps[1] - gs_eps[0] > delta) {
	      delta = gs_eps[1] - gs_eps[0] ;
	      bigI = -1 ;
	    }
	
	    //      cout << gs_eps[0] << " " << gs_eps[1] << " " << gs_eps[2] << " " << bigI << endl ;
	    while (delta > 0.25*gs_eps[1]) {
	      _real epsTry = gs_eps[1] + 0.383 * bigI * delta ;
	      ener2 = opt.update(x, grad, epsTry, xtry2) ;
	      if (verb) {
		cout << "ener = " << ener2 << " old_ener = " << old_ener << " eps = " << epsTry << endl ;
	      }
	      if (ener2 > gs_ener[1]) {
		if (bigI == 1) {
		  gs_ener[2] = ener ;
		  gs_eps[2] = epsTry ;
		}
		else{
		  gs_ener[0] = ener ;
		  gs_eps[0] = epsTry ;
		}
	      }
	      else{
		xtry.copy(xtry2) ;
		ener = ener2 ;
		if (bigI == 1) {
		  gs_ener[0] = gs_ener[1] ;
		  gs_eps[0] = gs_eps[1] ;
		  gs_ener[1] = ener ;
		  gs_eps[1] = epsTry ;
		}
		else {
		  gs_ener[2] = gs_ener[1] ;
		  gs_eps[2] = gs_eps[1] ;
		  gs_ener[1] = ener ;
		  gs_eps[1] = epsTry ;
		}
	      }
	      delta = gs_eps[2] - gs_eps[1];
	      bigI = 1 ;
	      if (gs_eps[1] - gs_eps[0] > delta) {
		delta = gs_eps[1] - gs_eps[0] ;
		bigI = -1 ;
	      }
	      if (verb)
		cout << gs_eps[0] << " " << gs_eps[1] << " " << gs_eps[2] << " " << bigI << endl ;
	    }

	    eps = gs_eps[1] * 2 ;
	    x.copy(xtry) ;
	    ener2 = opt.endOfIteration(x) ;
	    if (ener2 > -1e-10)
	      ener = ener2 ;
	  }
	}
	//	if (ener < 0.01)
	//	  break ;
	if (old_ener - ener < minVarEn * (old_ener0-ener)) {
	  smallVarCount ++ ;
	  //cg = false ;
	  if (smallVarCount > svcMAX)
	    break ; 
	  if (smallVarCount == svcMAX)
	    eps = gs_eps[2] * 100 ; 
	}
	else
	  smallVarCount = 0 ;
	old_ener = ener ;
	if (verb)
	  cout << "ener = " << ener << " " << smallVarCount << endl ;
      }

      if (opt.stopProcedure())
	break ;
      //    cout << "endloop" << endl; 
    }

    opt.endOfProcedure(x) ;
    return ener ;
  }
} ;

#endif
  
