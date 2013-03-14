/**
   optim.h
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





/**
A few optimization components.
*/
template<class OBJECT_TYPE> class Optim{
 public:
  double epsMin ;
  Optim(){epsMin = 1e-7 ;}
  virtual double linearCGFunction(OBJECT_TYPE &x, void *par){cout << "linearCGFunction is not implemented" << endl; exit(1) ;}
  virtual double scalProd(OBJECT_TYPE &x, OBJECT_TYPE &y){cout << "scalProd is not implemented" << endl; exit(1) ;}
  virtual double computeGrad(OBJECT_TYPE &x, OBJECT_TYPE &grad, void *par) {cout << "computeGradient is not implemented" << endl; exit(1) ;}
  virtual double objectiveFun(OBJECT_TYPE &x, void *par) {cout << "objectiveFun is not implemented" << endl; exit(1) ;}
  virtual void startOfProcedure(OBJECT_TYPE &x) {;}
  virtual void endOfProcedure(OBJECT_TYPE &x) {;}
  virtual void startOfIteration(OBJECT_TYPE &x, void *par) {;}
  virtual double endOfIteration(OBJECT_TYPE &x, void *par) {return -1 ;}
  virtual double update(OBJECT_TYPE &x, OBJECT_TYPE& grad, double eps, void *par, OBJECT_TYPE &res) {
      res.copy(grad) ;
      res *= -eps ;
      res += x ;
      return objectiveFun(res, par) ;
  }
  virtual ~Optim() {} ;

  /** Linear conjugate gradient
      requires the implementation of linearCGFunction (Linear transformation), scalProd (dot product)
  */
  double linearCG(OBJECT_TYPE &x0, OBJECT_TYPE &x, OBJECT_TYPE &b, int nbStep, double error, bool verb, void*par) {
    OBJECT_TYPE r,p,q, foo ;
    double mu ;

    r.copy(b) ;
    linearCGFunction(x0, foo, par) ;
    r -= foo ;

    //conjugate gradient loop
    
    double energy, energy0 = scalProd(x0, foo)/2 - scalProd(b,x0) ;
    double energyOld = energy0 ;
    double muOld = 0, alpha, beta ;
    x.copy(x0) ;

    for (int ss = 0; ss <nbStep; ss++) {
      mu = scalProd(r,r) ;
      if (ss == 0)
	p.copy(r) ;
      else {
	beta = mu/muOld ;
	p *= beta ;
	p += r ;
      }

      linearCGFunction(p, q, par) ;
      alpha = mu /(scalProd(p, q) + 1e-10) ;
      foo.copy(p) ;
      foo *= alpha ;
      x += foo ;
      foo.copy(q) ;
      foo *= alpha ;
      r -= foo ;
      
      muOld = mu ;
      energy = energyOld - alpha * scalProd(p,r) ;

      if (verb)
	cout << "step " << ss << " conjugate gradient: " << energy0 << " " << energy << endl ;
      if (abs(energyOld - energy) < error * energyOld)
	break ;
      energyOld = energy ;
    }
    return energy ;
  }



  /** 
      Non linear conjugate gradient
      Requires a definition of objectiveFun (Objective function), computeGrad (gradient of the objective function), scalProd (dot product)
      Optional functions: startOfProcedure,endOfProcedure, startOfIteration, endOfIteration
      *par is send to objectiveFun and computeGrad to transmit additional parameters.
      Control parameters: nb_iter: maximum number of iterations; epsIni: initial step; epsMax: maximum step; minVarEn: minimal relative variation of energy to continue
      gs: flag for golden search; verb: flag for verbose printing
  */
  double gradDesc(OBJECT_TYPE& x0, OBJECT_TYPE &x, int nb_iter, double epsIni, double epsMax, double minVarEn, bool gs, bool verb, void *par) {
    OBJECT_TYPE grad, oldGrad, xtry, xtry2, foo ;
    double ng2, eps = epsIni, old_ener, old_ener0, ener=0, ener2 ;
    _real gs_ener[3], gs_eps[3] ;
    unsigned int CG_RATE = 200 ;
    int smallVarCount = 0 , svcMAX = 2 ;
    bool cg ;

    x.copy(x0) ;
    startOfProcedure(x) ;
    old_ener = objectiveFun(x0, par) ;
    old_ener0 = old_ener ; 
    if (verb)
      cout << "initial energy = " << old_ener << endl ;

/*     if (old_ener < 0.00000000001) { */
/*       endOfProcedure(x) ; */
/*       return(old_ener) ; */
/*     } */

    bool stopLoop ;
    int cg_count = 0;

    for(int iter=0; iter < nb_iter; iter++) {
      startOfIteration(x, par) ;
      if (verb)
	cout <<"Iteration " << iter+1 << endl ;
      stopLoop = false;

      if (iter > 0)
	oldGrad.copy(grad) ;
      ng2 = computeGrad(x, grad, par) ;
      cg = false ;
      if (iter > 0 && cg_count % CG_RATE != 0) {
	double b ;
	foo.copy(grad) ;
	foo -= oldGrad ;
	b = scalProd(grad, foo)/ scalProd(oldGrad, oldGrad) ;
	if (b > 0) {
	  oldGrad *= b ;
	  grad += oldGrad ;
	  cg = true ;
	  cout << "Using CG direction" << endl ;
	}
	else if (verb == true)
	  cout << "Negative b in CG: " << b << endl ; 
      }



      if (eps > epsMax)
	eps = epsMax ;

      ener = update(x, grad, eps, par, xtry) ;
      if (verb) {
	cout << "ener = " << ener << " old_ener = " << old_ener << " eps = " << eps << " " << ng2 << endl ;
      }

      gs_ener[0] = old_ener ;
      gs_eps[0] = 0 ;

      if (ener > old_ener) {
	// test if correct direction of descent
	ener2 = update(x, grad, eps/10000, par, xtry2) ;
	if (verb && ener2 > old_ener) {
	  cout << "Bad Gradient ; ener = " << ener2 << " old_ener = " << old_ener << " eps = " << eps/1000 << endl ;
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
      
	if (iter==0) {
	  bool okLoop = false ;
	  int il = 0 ;
	  do {
	    ener = update(x, grad, 2*gs_eps[2], par, xtry) ;
	    if (verb) {
	      cout << "ener = " << ener << " old_ener = " << old_ener << " eps = " << 2*gs_eps[2] << endl ;
	    }
	    if (ener < 0.9999999 * gs_ener[2]){
	      gs_eps[2] *= 2 ;
	      gs_ener[2] = ener ;
	      okLoop = true ;
	    }
	    else 
	      okLoop = false ;
	  }
	  while (okLoop && (++il < 3)) ;
/* 	  if (il == 3) */
/* 	    eps = gs_eps[2]/2 ; */
/* 	  else */
	    eps = gs_eps[2] ;
	  ener = gs_ener[2] ;
	}

	if (!gs && iter>0) {
	  double a0 = 0.0, b0 = 5000 ;
	  while ((gs_ener[2] > old_ener - a0 * gs_eps[2] * ng2 || gs_ener[2] < old_ener - b0 * gs_eps[2] * ng2) && gs_eps[2] > epsMin)  {
	    gs_eps[2] *= .5 ;
	    gs_ener[2] = update(x, grad, gs_eps[2], par, xtry) ;
	    if (verb) {
	      cout << "ener = " << gs_ener[2] << " modified old_ener = "  << old_ener - a0 * gs_eps[2] * ng2 
		   << " " << old_ener - b0 * gs_eps[2] * ng2  << " eps = " << gs_eps[2] << endl ;
	    }
	  }
	  x.copy(xtry) ;
	  ener = gs_ener[2] ;
	  eps = 1.5*gs_eps[2] ;
	  ener2 = endOfIteration(x,par) ;
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
	      gs_eps[1] *= .5 ;
	      gs_ener[1] = update(x, grad, gs_eps[1], par, xtry2) ;
	      if (verb) {
		cout << "ener = " << gs_ener[1] << " old_ener = " << old_ener  << " eps = " << gs_eps[1] << endl ;
	      }
	      kg++ ;
	    }
	    while ( (gs_ener[1] >  0.999999 * old_ener || (gs_ener[1] >  0.999999 * gs_ener[2] && kg <= 5)) && gs_eps[1] > epsMin) ;
	  else
	    do {
	      gs_eps[2] *= 2 ;
	      gs_ener[2] = update(x, grad, gs_eps[2], par, xtry2) ;
	      if (verb) {
		cout << "ener = " << gs_ener[2] << " old_ener = " << old_ener  << " eps = " << gs_eps[2] << endl ;
	      }
	      kg++ ;
	    }
	    while ( (gs_ener[1] >  0.999999 * gs_ener[2] && kg <= 5) && gs_eps[2] > epsMax) ;
      
	
	  if (gs_ener[1] > gs_ener[0] || gs_ener[1] > gs_ener[2]) {
	    // if no intermediate is found: no golden search 
	    if (verb)
	      cout << "No golden search" << endl ;
	    eps = gs_eps[2]/0.75 ;
	    update(x, grad, gs_eps[2], par, xtry) ;
	    x.copy(xtry) ;
	    endOfIteration(x,par) ;
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
	      ener2 = update(x, grad, epsTry, par, xtry2) ;
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
	    ener2 = endOfIteration(x,par) ;
	    if (ener2 > -1e-10)
	      ener = ener2 ;
	  }
	}
	//	if (ener < 0.01)
	//	  break ;
	if (old_ener - ener < minVarEn * (old_ener0-ener)) {
	  smallVarCount ++ ;
	  if (smallVarCount > svcMAX)
	    break ; 
	  if (smallVarCount == svcMAX)
	    eps = 0.5 ; 
	}
	else
	  smallVarCount = 0 ;
	old_ener = ener ;
	if (verb)
	  cout << "ener = " << ener << endl ;
      }
      //    cout << "endloop" << endl; 
    }

    endOfProcedure(x) ;
    return ener ;
  }
} ;

#endif
  
