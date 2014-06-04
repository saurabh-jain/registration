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
    if (verb)
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
  double operator()(_optimFun &opt, _scalProd &scp, OBJECT_TYPE& x0, OBJECT_TYPE &x, int nb_iter, double epsIni, double epsMax, double minVarEn, bool gs, int verb) {
    TAN_TYPE grad, oldGrad, dir0, oldDir, savedGrad;
    OBJECT_TYPE xtry, xtry2, xtry3 ;
    double  enerTry, enerTry2, ener0, ener=0, ener2, eps2, b, oldGrad2,
      grad2, grad12, gradTry ;
    // _real gs_ener[3], gs_eps[3] ;
    unsigned int CG_RATE = 200 ;
    int smallVarCount = 0 , svcMAX = 2 ;
    bool cg = true ;
    //    _optimFun opt ;

    eps = epsIni ;
    x.copy(x0) ;
    opt.startOfProcedure(x) ;
    ener = opt.objectiveFun(x0) ;
    ener0 = ener ; 
    if (verb>0)
      cout << "initial energy = " << ener << endl ;

    bool stopLoop, noupdate ;
    int cg_count = 0;

    for(int iter=0; iter < nb_iter; iter++) {
      noupdate = false ;
      opt.startOfIteration(x) ;
      if (verb>1)
	cout <<"Iteration " << iter+1 << endl ;
      stopLoop = false;

      opt.gradNorm = opt.computeGrad(x, grad) ;
      if (iter == 0) {
	oldGrad2 = scp(grad, grad) ;
	grad2 = oldGrad2 ;
	if (oldGrad2 < 1e-20)
	  gradTry = 1e-10 ;
	else
	  gradTry = sqrt(oldGrad2) ;
	oldDir.copy(grad) ;
	oldGrad.copy(grad) ;
	dir0.copy(grad) ;
	b=0 ;
      }
      else {
	grad2 = scp(grad, grad) ;
	grad12 = scp(grad, oldGrad) ;
	oldGrad.copy(grad) ;
	if (cg)
	  b = (grad2 -grad12)/oldGrad2 ;
	else
	  b = 0 ;
	if (b<0)
	  b=0 ;
	oldGrad2 = grad2 ;
	gradTry = grad2 + b * grad12 ;
	if (gradTry < 1e-20)
	  gradTry = 1e-10 ;
	else
	  gradTry = sqrt(gradTry) ;
	dir0.copy(oldDir) ;
	dir0 *= b ;
	dir0 += grad ;
	//	addProduct(grad, b, oldDir, dir0) ;
	oldDir.copy(dir0) ;
	oldGrad.copy(grad) ;
      }

      if (verb>1)
	cout << "Gradient Norm: " << opt.gradNorm << endl ;
      // cg = false ;
      // if (iter > 0 && cg_count % CG_RATE != 0 && smallVarCount == 0) {
      // 	//	_scalProd scp(x) ;
      // 	double b ;
      // 	foo.copy(grad) ;
      // 	foo -= oldGrad ;
      // 	b = scp(grad, foo)/ scp(oldGrad, oldGrad) ;
      // 	if (b > 0) {
      // 	  oldGrad *= b ;
      // 	  grad -= oldGrad ;
      // 	  cg = true ;
      // 	  if (verb) 
      // 	    cout << "Using CG direction" << endl ;
      // 	}
      // 	else if (verb == true) {
      // 	  cout << "Negative b in CG: " << b << endl ; 
      // 	}
      // }
      //      cout << grad << endl ;



      if (eps > epsMax)
	eps = epsMax ;
      eps2 = opt.epsBound() ;
      if (eps > eps2)
	eps = eps2 ; 

      enerTry = opt.update(x, dir0, eps, xtry) ;
      if (verb>1) {
	cout << "ener = " << enerTry << " old ener = " << ener << " eps = " << eps << " " << opt.gradNorm << endl ;
      }

      if (enerTry > ener) {
	// test if correct direction of descent
	double	epsTest = 1e-6/gradTry ;
	ener2 = opt.update(x, dir0, epsTest,  xtry2) ;
	if (ener2 > ener) {
	  if (cg == false){
	    cout << "Iter " << iter << ": Bad Direction ; ener = " <<
	    ener2 << " old ener = " << ener << " eps = " << epsTest << endl ;
	    //	ener2 = opt.update(x, grad, 0,  xtry2) ;
	    //cout << ener2 << endl ;
	    stopLoop = true ;
	  }
	  else {
	    cg = false ;
	    noupdate = true ;
	  }
	}
	else {
	  while ((enerTry > ener) && (eps > epsMin)) {
	      eps /= 2 ;
	      enerTry = opt.update(x, dir0, eps, xtry) ;
	    }
	}
      }

      if (~noupdate) {
	bool contt = true ;
	while (contt) {
	  enerTry2 = opt.update(x, dir0, 0.5*eps, xtry2);
	  if (enerTry > enerTry2) {
	    eps /= 2 ;
	    enerTry = enerTry2 ;
	    xtry.copy(xtry2) ;
	  }
	  else
	    contt = false ;
	}
	contt = true ;
	while (contt) {
	  enerTry2 = opt.update(x, dir0, 1.25*eps, xtry2);
	  if (enerTry > enerTry2) {
	    eps *= 1.25 ;
	    enerTry = enerTry2 ;
	    xtry.copy(xtry2) ;
	  }
	  else
	    contt = false ;
	}

 	if (ener - enerTry < minVarEn * (ener0-ener)) {
	  smallVarCount ++ ;
	  //cg = false ;
	  if (smallVarCount > svcMAX)
	    break ; 
	}
	else
	  smallVarCount = 0 ;
	ener = enerTry ;
	x.copy(xtry) ;
	if (verb>1)
	  cout << "ener = " << ener << " " << smallVarCount << endl ;
      }

      if (verb>0)
	cout << "Iter " << iter << " ener = " << ener << " eps = " << eps << " " << opt.gradNorm << endl ;
      if (opt.stopProcedure())
	break ;
      //    cout << "endloop" << endl; 
      opt.endOfIteration(x) ;
    }

    opt.endOfProcedure(x) ;
    return ener ;
  }
} ;

#endif
  
