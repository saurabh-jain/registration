/**
   diffeomorphism.h
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
#ifndef _DIFFEO_
#define _DIFFEO_
#include "vectorKernels.h"
//#include <fftw3.h>
//#include "Vector.h"
#include "matrix.h"
//#include "param_matching.h"
#ifdef MEM
extern mem ;
#endif

/**
   manipulation and equations for diffeomorphisms
*/
class Diffeomorphisms
{
public:
  Ivector imageDim ;
  VectorMap mask ;
  // Deformation and variations
  VectorMap _phi, _psi, var_psi, var_phi, alpha, beta, betaT, alphaT, _fooMap ;
  // Vector fields and their momenta
  VectorMap v0, Lv0, w0, Lw0 ;
  TimeVectorMap LvTime, __phi, __psi ;
  param_matching param ;
  unsigned int _T ;
  VectorKernel _kern ;
  double kernelNormalization ;

  //  Diffeomorphisms(){FFT_INITIALIZED = false;} 
  Diffeomorphisms(){kernelNormalization = 1 ;} 

  void setKernel() {
    _kern.setParam(param) ;
  }

  // Lie group functions
/**
   Lie group adjoint res0 = [v, w]
*/
  void adjoint(const VectorMap &v, const VectorMap &w, VectorMap &res0) {  
    Vector foo ;
    VectorMap  res;
    std::vector<VectorMap> gradv, gradw ;
  
    v.differential(gradv, param.spaceRes) ;
    w.differential(gradw, param.spaceRes) ;

    matrixProduct(gradv, w, res0) ;
    matrixProduct(gradw, v, res) ;
    res0 -= res ;
  }

  void adjoint2(const VectorMap &v, const VectorMap &w, VectorMap &res0) {  
    VectorMap id, foo, foo1, foo2 ;
    VectorMap  res;
    std::vector<VectorMap> gradv ;

    id.idMesh(v.d) ;
    foo1.copy(id) ;
    foo2.copy(id) ;
    foo1 += v ;
    foo2 -= v ;

  
    foo1.differential(gradv, param.spaceRes) ;
    matrixProduct(gradv, w, foo) ;
    foo2.multilinInterp(foo, res) ;
    res0.copy(res) ;
  }

/**
   Lie group dual adjoint
*/
  void adjointStar(const VectorMap &v, const VectorMap &m, VectorMap &res0) {
    Vector foo, foo1, foo2  ;
    std::vector<VectorMap> gradv, gradm;
    VectorMap res ;

    v.differential(gradv, param.spaceRes) ;
    matrixTProduct(gradv, m, res) ;
    for (unsigned int i=0; i<v.size(); i++) {
      for (unsigned int j=0; j<v.size(); j++) {
	foo.copy(v[j]) ;
	foo *= m[i] ;
	foo.shift(j, 1, foo1) ;
	foo.shift(j, -1, foo2) ;
	foo1 -= foo2 ;
	foo1 /= 2*param.spaceRes[j] ;
	res[i] += foo1 ;
      }
    }
    res0.copy(res) ;
  }

  void adjointStar2(const VectorMap &v, const VectorMap &m, VectorMap &res0) {
    VectorMap foo, foo1, foo2, id ;
    std::vector<VectorMap> gradv ;
    VectorMap res ;

    id.idMesh(m.d) ;
    foo1.copy(id) ;
    foo2.copy(id) ;
    foo1 += v ;
    foo2 -= v ;
    foo.al(m.d) ;
    for (unsigned int i=0; i<foo1.size(); i++)
      foo1.multilinInterpDual(m[i], foo[i]) ;
    foo2.differential(gradv, param.spaceRes) ;
    matrixTProduct(gradv, foo, res) ;
    res0.copy(res) ;
  }

/**
   Lie group Adjoint res0 = Ad_phi(v)
*/
  void big_adjoint(const VectorMap &v, const VectorMap &phi, VectorMap &res0) {
    std::vector<VectorMap> dphi ;
    VectorMap Z, res ;
    Vector foo ;

    phi.inverseDifferential(dphi, param.spaceRes) ;
    phi.multilinInterp(v, Z) ;

    res.al(v.d) ;

    for (unsigned int i=0; i<v.size(); i++) {
      res[i].zero() ;
      for (unsigned int j=0; j<v.size(); j++) {
	foo.copy(Z[j]) ;
	foo *= dphi[i][j] ;
	res[i] += foo ;
      }
    }

    res0.copy(res) ;
  }

/** 
    Lie group dual Adjoint
*/
  void big_adjointStar(const VectorMap &m, const VectorMap &phi, VectorMap &res0) {
    std::vector<VectorMap> dphi ;
    Vector jac, foo ;
    VectorMap Z, res ;
  
    phi.differential(dphi, param.spaceRes) ;
    phi.jacobian(jac, param.spaceRes) ;
    phi.multilinInterp(m, Z) ;
    res.al(m.d) ;

    for (unsigned int i=0; i<m.size(); i++) {
      res[i].zero() ;
      for (unsigned int j=0; j<m.size(); j++) {
	foo.copy(Z[j]) ;
	foo *= dphi[j][i] ;
	res[i] += foo ;
      }
      res[i] *= jac ;
    }

    res0.copy(res) ;
  }

  //  void ba_K_baStar(const VectorMap &m, const VectorMap &phi, const VectorMap &psi, VectorMap &res0) ;

/**
Integration of a time dependent vector field
*/
  _real flow(TimeVectorMap &Lv) {
    VectorMap foo, v, id, semi ;
    id.idMesh(Lv.d) ;
    _real res =0 ;

    _real dt = 1.0/(_T) ;

    __phi[0].copy(id) ;
    __psi[0].copy(id) ;
    for (unsigned int t=1; t<=_T; t++) {
      kernel(Lv[t-1], v) ;
      res += Lv[t-1].scalProd(v) ;
      semi.copy(v) ;
      semi *= dt ;
      for (int jj = 0; jj<param.nb_semi; jj++) {
	foo.copy(semi) ;
	foo /= -2 ;
	foo += id ;
	foo.multilinInterp(v, semi) ;
	semi *= dt ;
      }
    
      foo.copy(id) ;
      foo +=  semi ;
      __phi[t-1].multilinInterp(foo, __phi[t]) ;
      foo.copy(id) ;
      foo -=  semi ;
      foo.multilinInterp(__psi[t-1], __psi[t]) ;
    }
    return res / _T ;
  }

  _real flowDual(TimeVectorMap &Lv) {
    VectorMap foo, v, id, semi ;
    id.idMesh(Lv.d) ;
    _real res =0 ;

    _real dt = 1.0/(_T) ;

    __phi[0].copy(id) ;
    __psi[0].copy(id) ;
    for (unsigned int t=1; t<=_T; t++) {
      kernel(Lv[t-1], v) ;
      res += Lv[t-1].scalProd(v) ;
      semi.copy(v) ;
      semi *= dt ;
      for (int jj = 0; jj<param.nb_semi; jj++) {
	foo.copy(semi) ;
	foo /= -2 ;
	foo += id ;
	foo.multilinInterp(v, semi) ;
	semi *= dt ;
      }
    
      foo.copy(id) ;
      foo +=  semi ;
      __psi[t-1].multilinInterpDual(foo, __psi[t]) ;
      foo.copy(id) ;
      foo -=  semi ;
      foo.multilinInterpDual(__phi[t-1], __phi[t]) ;
    }
    return res / _T ;
  }

  void initFlow(const Domain &d) {  _phi.idMesh(d) ; _psi.idMesh(d) ;}
  _real updateFlow(const VectorMap &Lv, _real dt) {
    VectorMap foo, foo2, v, semi, id ;
    _real res ;
    id.idMesh(Lv.d) ;
    kernel(Lv, v) ;
    res = Lv.scalProd(v) ;
    semi.copy(v) ;
    semi *= dt ;
    for (int jj = 0; jj<param.nb_semi; jj++) {
      foo.copy(semi) ;
      foo /= -2 ;
      foo += id ;
      foo.multilinInterp(v, semi) ;
      semi *= dt ;
    }
    
    foo.copy(id) ;
    foo +=  semi ;
    _phi.multilinInterp(foo, _phi) ;
    foo.copy(id) ;
    foo -=  semi ;
    foo.multilinInterp(_psi, foo2) ;
    _psi.copy(foo2) ;
  
    return res*dt ;
  }

  _real updateFlowInverseOnly(const VectorMap &Lv, _real dt) {
    VectorMap foo, foo2, v, semi, id ;
    _real res ;

    id.idMesh(Lv.d) ;
    kernel(Lv, v) ;
    res = Lv.scalProd(v) ;
    semi.copy(v) ;
    semi *= dt ;
    for (int jj = 0; jj<param.nb_semi; jj++) {
      foo.copy(semi) ;
      foo /= -2 ;
      foo += id ;
      foo.multilinInterp(v, semi) ;
      semi *= dt ;
    }
    
    foo.copy(id) ;
    foo -=  semi ;
    foo.multilinInterp(_psi, foo2) ;
    _psi.copy(foo2) ;
    
    return res*dt ;
  }

  //  void GeodesicDiffeoEvolutionInverseOnly(const VectorMap &Lv, VectorMap &Lv2, _real delta) ;
  //  void GeodesicDiffeoEvolutionInverseOnly(){ GeodesicDiffeoEvolutionInverseOnly(Lv0, _fooMap, 1);}
  //  void GeodesicDiffeoEvolutionInverseOnly(const VectorMap &Lv){ GeodesicDiffeoEvolutionInverseOnly(Lv, _fooMap, 1);}

/**
   Solution of the EPdiff equation with initial momentum Lv over time [0,delta];
   Lv2 is the final momentum
   diffeomorphisms and inverse diffeomorphisms are stored in _phi and _psi
*/
  void GeodesicDiffeoEvolution(const VectorMap &Lv,  VectorMap &Lv2, _real delta) {
    VectorMap foo, foo2, Lvt, v, vt, id, semi ;
    //    cout << "epdiff : " << Lv.maxNorm() << endl ;
    kernel(Lv, v);
    _real dt, T, M = delta*v.maxNorm(), norm ;

    T = ceil(param.accuracy*M+1) ;
    Lvt.copy(Lv) ;
    vt.copy(v) ;
    norm  = sqrt(vt.scalProd(Lvt)) ;
    /*
      adjointStar(vt, Lvt, foo) ;
      kernel(foo, foo2) ;
      T = ceil(param.accuracy*foo.scalProd(foo2)/ (norm*norm+1) + 1) ;*/
    if (T > param.Tmax)
      T = param.Tmax ;

    id.idMesh(Lv.d) ;
    //   Integrate along the path                                                 
    dt = delta/T ;
    if (param.verb)
      cout << "Evolution; T =  " << T << " " << norm << endl ; 
    
    for (unsigned int t=0; t<T; t++) {
      copyProduct(dt, vt, semi) ;
      for (int jj = 0; jj<param.nb_semi; jj++) {
	addProduct(id, -.5, semi, foo) ;
	foo.multilinInterp(vt, semi) ;
	semi *= dt ;
      }
  
      if (t>0){
	addProduct(id, 1, semi, foo) ;
	_phi.multilinInterp(foo, foo2) ;
	_phi.copy(foo2) ;
	addProduct(id, -1, semi, foo) ;
	foo.multilinInterp(_psi, foo2) ;
	_psi.copy(foo2) ;
      }
      else {
	addProduct(id,1,semi, _phi) ;
	addProduct(id,-1,semi, _psi) ;
      }

      //addProduct(id, -1, semi, foo) ;
      //big_adjointStar(Lvt, foo, Lvt) ;
      adjointStar2(semi, Lvt, Lvt) ; 
      //    adjointStar(vt, Lvt, foo) ;
      //    kernel(foo, foo2) ;
      //    addProduct(Lvt, -dt, foo) ;
      kernel(Lvt, vt) ;
      _real norm2 = sqrt(vt.scalProd(Lvt)) ;
      // cout << "t = " << t << " --- " << dt * foo.scalProd(foo2)/ (norm2*norm2) << endl ;
    
      if (norm2 > 0.00000001) {
	vt *= norm/norm2 ;
	Lvt *=  norm/norm2 ;
      }
    }
    
    Lv2.copy(Lvt)  ;
  }

  void GeodesicDiffeoFlow(const VectorMap &Lv,  const int nStep, TimeVectorMap &phi, TimeVectorMap &psi) {
    VectorMap foo, foo2, Lvt, v, vt, id, semi ;
    //    cout << "epdiff : " << Lv.maxNorm() << endl ;
    kernel(Lv, v);
    _real dt, T, M = v.maxNorm(), norm ;

    T = ceil(param.accuracy*M+1) ;
    Lvt.copy(Lv) ;
    vt.copy(v) ;
    norm  = sqrt(vt.scalProd(Lvt)) ;
    /*
      adjointStar(vt, Lvt, foo) ;
      kernel(foo, foo2) ;
      T = ceil(param.accuracy*foo.scalProd(foo2)/ (norm*norm+1) + 1) ;*/
    if (T > param.Tmax)
      T = param.Tmax ;
    int rate = (int) ceil(T/nStep) ; 
    T = nStep * rate ;

    id.idMesh(Lv.d) ;
    //   Integrate along the path                                                 
    dt = 1.0/T ;
    if (param.verb)
      cout << "Evolution; T =  " << T << " " << norm << endl ; 
    phi.resize(nStep+1) ;
    psi.resize(nStep+1) ;
    phi[0].copy(id) ;
    
    for (unsigned int t=0; t<T; t++) {
      copyProduct(dt, vt, semi) ;
      for (int jj = 0; jj<param.nb_semi; jj++) {
	addProduct(id, -.5, semi, foo) ;
	foo.multilinInterp(vt, semi) ;
	semi *= dt ;
      }
  
      if (t>0){
	addProduct(id, 1, semi, foo) ;
	_phi.multilinInterp(foo, foo2) ;
	_phi.copy(foo2) ;
	addProduct(id, -1, semi, foo) ;
	foo.multilinInterp(_psi, foo2) ;
	_psi.copy(foo2) ;
      }
      else {
	addProduct(id,1,semi, _phi) ;
	addProduct(id,-1,semi, _psi) ;
      }

      if ((t+1)%rate == 0) {
	phi[(t+1)/rate].copy(_phi) ;
	psi[(t+1)/rate].copy(_psi) ;
      }
      //addProduct(id, -1, semi, foo) ;
      //big_adjointStar(Lvt, foo, Lvt) ;
      adjointStar2(semi, Lvt, Lvt) ; 
      //    adjointStar(vt, Lvt, foo) ;
      //    kernel(foo, foo2) ;
      //    addProduct(Lvt, -dt, foo) ;
      kernel(Lvt, vt) ;
      _real norm2 = sqrt(vt.scalProd(Lvt)) ;
      // cout << "t = " << t << " --- " << dt * foo.scalProd(foo2)/ (norm2*norm2) << endl ;
    
      if (norm2 > 0.00000001) {
	vt *= norm/norm2 ;
	Lvt *=  norm/norm2 ;
      }
    }
  }

  void GeodesicDiffeoEvolution(const VectorMap &Lv){ GeodesicDiffeoEvolution(Lv, _fooMap, 1);}
  void GeodesicDiffeoEvolution(_real delta) {GeodesicDiffeoEvolution(Lv0, _fooMap, delta);};
  void GeodesicDiffeoEvolution(){ GeodesicDiffeoEvolution(Lv0, _fooMap, 1);}


  //  void DualVarGeodesicDiffeoEvolution(const VectorMap &Lv, const VectorMap &Lw, _real delta) ;
  void DualVarGeodesicDiffeoEvolutionDiscrete(const VectorMap &Lv, const VectorMap &Ugrad, VectorMap &Uvar, _real delta) {
    VectorMap foo, foo1, foo2, xi,  eta, id, semi, var_v, v, w, H, u, uu;
    std::vector<VectorMap> dpsi ;

    kernel(Lv, v) ;
    _real norm = sqrt(v.scalProd(Lv)) ;

    _real M = delta*v.maxNorm() ;
    unsigned int T = (int) ceil(param.accuracy*M+1) ;
    /*    adjointStar(v, Lv, foo) ;
	  kernel(foo, foo2) ;
	  T = ceil(param.accuracy*foo.scalProd(foo2)/ (norm*norm+1) + 1) ;*/
    if (T > (unsigned int) param.Tmax)
      T = param.Tmax ;

    if (param.verb)
      cout << "DualVarGeodesicDiffeoEvolution: T = " << T << " M = " << M << endl ;
  
    id.idMesh(Lv.d) ;

    double dt = delta/T ;
    TimeVectorMap vt, Lvt, psit ;

    psit.resize(T+1, v.d) ;
    vt.resize(T+1, v.d) ;
    Lvt.resize(T+1, v.d) ;
    vt[0].copy(v) ;
    Lvt[0].copy(Lv) ;
    psit[0].copy(id) ;


    // forward evolution to compute v and Lv over time
    for (unsigned int t=1; t<=T; t++) {
      addProduct(id, -dt, vt[t-1], foo) ;
      foo.multilinInterp(psit[t-1], psit[t]) ;
      foo.copy(vt[t-1]) ;
      foo *= dt ;
      adjointStar2(foo, Lvt[t-1], Lvt[t]) ;
      // big_adjointStar(Lvt[t-1], foo, Lvt[t]) ;
      //    adjointStar(vt[t-1], Lvt[t-1], foo) ;
      //    addProduct(Lvt[t-1], -dt, foo, Lvt[t]) ;
      
      kernel(Lvt[t], vt[t]) ;
      _real norm2 = sqrt(vt[t].scalProd(Lvt[t])) ;
      //cout << "t: " << t << " " << norm2 << endl ;
      
      if (norm2 > 0.00000001) {
	Lvt[t] *= norm/norm2 ;
	vt[t] *= norm/norm2 ;
      }
    }
    
    u.copy(Ugrad) ;
    //  u *= -1 ;
    uu.zeros(v.d) ;
    

    // backward loop to compute the dual
    for (unsigned int t=T; t>0; t--) {
      // cout << t << endl;
      copyProduct(dt, vt[t], foo1);
	//      adjoint(vt[t], uu, foo) ;
      adjoint2(foo1, uu, foo) ;
      foo *= -1 ;
      //    foo -= foo1 ;
      adjointStar(uu, Lvt[t], foo1) ;
      kernel(foo1, foo1) ;
      foo += foo1 ;
      
      addProduct(id, -dt, vt[t], foo1) ;
      foo2.al(v.d) ;
      for (unsigned int i=0; i<u.size(); i++) 
	foo1.multilinInterpDual(u[i], foo2[i]) ;
      foo1.multilinInterpGradient(psit[t], dpsi) ;
      //psit[t].differential(dpsi, param.spaceRes) ;
      // foo1.zeros(Lv.d) ;
      matrixTProduct(dpsi, u, foo1) ;
      //    for (unsigned int i=0; i<foo1.size(); i++) {
      //      dpsi[i].scalProd(foo2, foo1[i]) ;
      //    }
      kernel(foo1, foo1) ;
      
      foo -= foo1 ;
      foo *= dt ;
      uu += foo ; 
      u.copy(foo2) ;

      // cout << "t=: " << t << " " << u.norm2() << " " << uu.norm2() << endl ;
      //kernel(uu,uu) ;
      
      // addProduct(id, -dt, vt[T-t], foo1) ;
      // for (unsigned int j=0; j<foo1.size(); j++)
      //   foo1.multilinInterpDual(u[j], foo2[j]) ;
      // u.copy(foo2) ;
    }
    //  cout << "endGrad" << endl ; 
    Uvar.copy(uu) ;
  }

  //  void DualVarGeodesicDiffeoEvolutionGlobal(const VectorMap &Lv, const VectorMap &Lw, _real delta) ;
  //  void DualVarGeodesicDiffeoEvolution(const VectorMap &Lv, const VectorMap &Lw){ DualVarGeodesicDiffeoEvolution(Lv, Lw, 1);} 
  //  void DualVarGeodesicDiffeoEvolution(_real delta) {DualVarGeodesicDiffeoEvolution(Lv0, Lw0, delta);}
  //  void DualVarGeodesicDiffeoEvolution(){ DualVarGeodesicDiffeoEvolution(Lv0, Lw0, 1);} 
  void DualVarGeodesicDiffeoEvolutionDiscrete(const VectorMap &Lv, const VectorMap &Lw, VectorMap &Uvar) {
    DualVarGeodesicDiffeoEvolutionDiscrete(Lv, Lw, Uvar, 1) ;}

  //  void VarGeodesicDiffeoEvolutionGlobal(const VectorMap &Lv, const VectorMap &Lw, _real delta) ;
  //void VarGeodesicDiffeoEvolution(const VectorMap &Lv, const VectorMap &Lw, _real delta) ;
  //void VarGeodesicDiffeoEvolution(const VectorMap &Lv, const VectorMap &Lw){VarGeodesicDiffeoEvolution(Lv, Lw, 1);} 
  //void VarGeodesicDiffeoEvolution(_real delta) {VarGeodesicDiffeoEvolution(Lv0, Lw0, delta);}
  //void VarGeodesicDiffeoEvolution(){ VarGeodesicDiffeoEvolution(Lv0, Lw0, 1);} 

  void ParallelTranslation(const VectorMap &Lv, const VectorMap &Lww, VectorMap & Lwc, _real delta) {
    VectorMap foo, Lvt, vt, id, semi, v, w, zz, Lw ;
  
    Lw.copy(Lww) ;
    kernel(Lv, v) ;
    kernel(Lw, w) ;
    _real nw = sqrt(Lw.scalProd(w)) + 1;
    Lw /= nw ;
    w /= nw ;
    
    Lvt.copy(Lv) ;
    vt.copy(v) ;
    id.idMesh(Lv.d) ;

    beta.zeros(v.d) ;

    _real T, dt, M = delta*v.maxNorm() ;

    T = 2*ceil(param.accuracy*M+1) ;  
    if (T > param.Tmax)
      T = param.Tmax ;
    dt = delta/T  ; 
    if (param.verb)
      cout << "ParallelTranslation " << T << endl ;

    if (param.verb)
      cout << "T = " << 0 << " " << vt.scalProd(Lvt) << " " << w.scalProd(Lw) << " " << vt.scalProd(Lw) << endl ;
    for (unsigned int t=0; t<T; t++) {
      semi.copy(vt) ;
      semi *= dt ;
      for (int jj = 0; jj<param.nb_semi; jj++) {
	foo.copy(semi) ;
	foo /= -2 ;
	foo += id ;
	foo.multilinInterp(vt, semi) ;
	semi *= dt ;
      }
  
      if (t>0){
	foo.copy(id) ;
	foo +=  semi ;
	_phi.multilinInterp(foo,  _phi) ;
	foo.copy(id) ;
	foo -=  semi ;
	foo.multilinInterp(_psi, _psi) ;
      }
      else {
	_phi.copy(id) ;
	_phi +=  semi ;
	_psi.copy(id) ;
	_psi -=  semi ;
      }

      adjointStar(w, Lvt, zz) ;
      adjointStar(vt, Lw, foo) ;
      zz += foo ;
      adjoint(vt, w, foo) ;
      inverseKernel(foo, foo) ;
      zz -= foo ;
      zz *= dt/2 ;
      Lw -= zz ;
      kernel(Lw, w) ;
      adjointStar(vt, Lvt, zz) ;
      big_adjointStar(Lv, _psi, Lvt) ;
      kernel(Lvt, vt) ;
      
      if (param.verb)
	cout << "T = " << (t+1)*dt << " " << vt.scalProd(Lvt) << " " << w.scalProd(Lw) << " " << vt.scalProd(Lw) << endl ;
      
    }
    Lw *= nw ;
    Lwc.copy(Lw) ;
  }
  
/** 
    Tries to rescale time to ensure constant velocity norm
    accepts only if energy is reduced
*/
  void geodesicTimeRescale(const TimeVectorMap &Lv, TimeVectorMap &Lvtry) {
    VectorMap Lv2, tmp ;
    double dt = 1.0/Lv.size(), q0 = 0, entot = 0 ;
    std::vector<double> ener, q ;
    ener.resize(Lv.size()) ;
    q.resize(Lv.size()+1) ;
    
    q[0] = 0 ;
    initFlow(Lv.d) ;
    for (unsigned int t=0; t<Lv.size(); t++) {
      ener[t] = updateFlow(Lv[t], dt) + 1e-10;
      entot += ener[t] ;
      ener[t] = sqrt(ener[t]) ;
      q[t+1] = q[t] + ener[t] ;
      //    cout << t << " : " << ener[t] << endl ;
    }
    
    q0 = q[q.size()-1] ;
    for (unsigned t=0; t<q.size(); t++)
      q[t] /= q0 ;
    
    Lvtry.resize(Lv.size(), Lv.d) ;
    
    unsigned int j = 1 ;
    j=1 ; 
    for(unsigned int t=0; t<Lv.size(); t++) {
      while(j < Lv.size() && q[j] < t*dt)
	j++ ;
      VectorMap tmp ;
      if(j== Lv.size() ) {
	Lvtry[t].copy(Lv[j-1]) ;
	Lvtry[t] *= dt * q0/ener[j-1] ;
      }
      else {
	_real r = (q[j] - t*dt)/(q[j] - q[j-1]) ;
	Lvtry[t].copy(Lv[j-1]) ;
	Lvtry[t] *= dt * q0/ener[j-1] ;
	Lvtry[t] *= r ;
	tmp.copy(Lv[j]) ;
	tmp *= dt * q0/ener[j] ;
	tmp *= 1-r ;
	Lvtry[t] += tmp ;
      }
    }
  }

  // kernel functions
/**
   Applies the kernel to v and returns res0
*/
  void kernel(const VectorMap &Lv, VectorMap&res0) {
    VectorMap res ;
    if (_kern.initialized() == false) {
      cout << "initializing kernel " << endl ;
      _kern.setParam(param) ;
      _kern.initFFT(imageDim, FFTW_MEASURE) ;
      //cout << "done" << endl;
    }
    _kern.applyWithMask(Lv, mask, res, param.spaceRes) ;
    res /= kernelNormalization ;
    res0.copy(res) ; 
  }

/**
Inverts the kernel with conjugate gradient
*/
  void inverseKernel(const VectorMap &v0, VectorMap &res0) {
    _real eps0 = param.inverseKernelWeight, eps = eps0 ;
    
    VectorMap X, KX, p, q, r, v, z, foo ;
    X.al(v0.d) ;
    v.copy(v0) ;
    _real nv = sqrt(v.norm2()) ;
    
    if (nv > 0.0000001)
      v /= nv ;

    X.zero() ;
    kernel(X, KX) ;
    foo.copy(X) ;
    foo *= eps ;
    KX += foo ;
    r.copy(v) ;
    r -= KX ;

    unsigned int K = 100 ;
    unsigned int K0 = 10 ;
    _real ener, enerInit = v.norm2(), mu=0, muold, alfa, alfa0, be, ener2 ;
    for(unsigned int i2 = 0; i2<K; i2++) {
      if (i2 == K0)
	eps = eps0 ;
      if (i2 == 0) {
	mu = r.norm2() ;
	p.copy(r) ;
      }
      else {
	muold = mu ;
	mu = r.norm2() ;
	be = mu/muold ;
	p *= be ;
	p += r ;
      }

      kernel(p,q) ;
      foo.copy(p) ;
      foo *= eps ;
      q+= foo;
      alfa0 = p.scalProd(q) ;
      
      if (alfa0 < 1e-30) {
	cout << "negative alpha: iter " << i2 << " " << alfa0 << endl ;
	if (nv > 0.0000001)
	  X *= nv ;
	res0.copy(X) ;
	return ;
      }

      alfa = mu / alfa0 ;
      foo.copy(p) ;
      foo *= alfa ;
      X += foo ;
      foo.copy(q) ;
      foo *= alfa ;
      r -= foo ;
      KX += foo ;
      ener = KX.scalProd(X)/2 - X.scalProd(v) ;
      z.copy(KX) ;
      z -= v ;
      ener2 = z.norm2() ;

      if ((i2 > K0) && (ener2 < 0.000001 || ener2 < 0.0001 * enerInit)) {
	K=i2 ;
	break ;
      }
      if (param.verb)
	cout << "iter " << i2 << ": ener = " << ener2 << " " << ener2/(enerInit + 0.00000000000001) << " " << alfa0 << endl ;
    }
    if (nv > 0.0000001)
      X *= nv ;
    if (param.verb && K == 100)
      cout << "iter " << K << ": ener = " << ener << " " << ener2/(enerInit + 0.00000000000001) << " " << alfa0 << endl ;
    res0.copy(X) ;
  }

  //void inverseKernelAlt(const VectorMap &v, VectorMap&res) ;
  _real kernelNorm(const VectorMap &Lv) {VectorMap v; kernel(Lv, v); return Lv.scalProd(v);}

/**
   Basic function for fftw3 initialization
*/
//  void init_fft(const Ivector &dim, unsigned int TYPEINIT) {
//    _kern.initFFT(dim, TYPEINIT) ;
//  }

//  void init_fft() {
    //    unsigned int TYPEINIT = FFTW_MEASURE ;
  //    _kern.setParam(param) ;
    //  KERNEL.FFT_INITIALIZED = true ;
    //  }
  void makeMask(unsigned int margin, Domain &S) {makeMaskNeumann(margin, S);}

/**
   Creates a censoring mask, null boundary condition
*/
  void makeMaskZero(unsigned int margin, Domain &S) {
    // compute a smooth mask on images
    
    VectorMap Id;
    Vector res ;
    if (param.periodic == 1) {
      mask.ones(S) ;
      return ;
    }
    mask.zeros(S) ;

    Ivector m, M ;
    m.resize(S.n) ;
    M.resize(S.n) ;
    for (unsigned int i=0; i < S.n; i++) {
      M[i] = S.getM(i) - margin ;
      m[i] = S.getm(i) + margin ;
    }
    Domain D(m, M) ;
    Id.idMeshNorm(D) ;
    res.zeros(D) ;
    
    for(unsigned int i=0; i<res.size(); i++) {
      bool notzero ;
      notzero = true ;
      for (unsigned int j=0; j<D.n && notzero ; j++) {
	if (Id[j][i] < 0.01 || 1-Id[j][i] < 0.01) 
	  notzero = false ;
	else
	  res[i] += 1/(Id[j][i]*(1-Id[j][i])) ;
      }
      if (notzero)
	res[i] = exp(-0.01 * res[i]) ;
      else
	res[i] = 0 ;
    }
    
    for (unsigned int j=0; j<D.n ; j++)
      mask[j].subCopy(res) ;
  }

/**
   Creates a censoring mask, von Neuman boundary condition
*/
  void makeMaskNeumann(unsigned int margin, Domain &S) {
    VectorMap Id;
    Vector res ;
    if (param.periodic == 1) {
      mask.ones(S) ;
      return ;
    }
    mask.zeros(S) ;

    Ivector m, M ;
    m.resize(S.n) ;
    M.resize(S.n) ;
    for (unsigned int i=0; i < S.n; i++) {
      M[i] = S.getM(i)  ;
      m[i] = S.getm(i) ;
    }
    
    for (unsigned int j=0; j<S.n ; j++) {
      M[j] = S.getM(j) - margin ;
      m[j] = S.getm(j) + margin ;
      Domain D(m, M) ;
      Id.idMeshNorm(D) ;
      res.zeros(D) ;
      for(unsigned int i=0; i<res.size(); i++) {
	bool notzero ;
	notzero = true ;
	if (Id[j][i] < 0.01 || 1-Id[j][i] < 0.01) 
	  notzero = false ;
	else
	  res[i] = 1/(Id[j][i]*(1-Id[j][i])) ;
	if (notzero)
	  res[i] = exp(-0.01 * res[i]) ;
	else
	  res[i] = 0 ;
      }
      mask[j].subCopy(res) ;
      M[j] = S.getM(j) ;
      m[j] = S.getm(j) ;
    }
  }



};


#endif
