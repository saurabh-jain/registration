/**
   morphingSym.h
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
#ifndef _MORPHING_
#define _MORPHING_
//#include <complex.h>
#include "ImageMatching.h"
#include "ImageMatchingAffine.h"
#include "matchingBase.h"
#include "param_matching.h"
#ifdef MEM
extern mem ;
#endif

/** Main class for image metamorphosis
 */
class Morphing:public ImageMatching
{
 public:
  Vector Z ;
  TimeVector J ;
  TimeVectorMap w ;

  Morphing(){} ;

  Morphing(param_matching &par){ init(par) ;}
  Morphing(char *file, int argc, char** argv) {init(file, argc, argv);}
  Morphing(char *file, int k) {init(file, k);}


  void affineReg(){
    ImageAffineEnergy enerAff ;
    Template.domain().putm(enerAff.MIN) ;
    initializeAffineReg(enerAff) ;
    ImageMatching::affineReg(enerAff) ;
    finalizeAffineTransform(gamma, enerAff.MIN) ;    
  }

/**
   integrates the flow along a time dependent vector field
*/
    void flow(TimeVectorMap &phi) 
    {
      VectorMap Id, tmp ;
      phi.resize(w.size()+1, w.d) ;
      phi[0].idMesh(w.d) ;
      Id.idMesh(w.d) ;

      for(unsigned t=1; t<phi.size(); t++) {
	kernel(w[t-1], tmp) ;
	tmp *= 1.0/w.size() ;
	tmp += Id ;
	phi[t-1].multilinInterp(tmp, phi[t]) ;
      }
    }    
  

    void flow(TimeVectorMap &phi, TimeVectorMap &psi)
    {
      VectorMap Id, tmp, ft ;
      psi.resize(w.size()+1, w.d) ;
      phi.resize(w.size()+1, w.d) ;
      phi[0].idMesh(w.d) ;
      psi[0].idMesh(w.d) ;
      Id.idMesh(w.d) ;

      for(unsigned t=1; t<phi.size(); t++) {
	kernel(w[t-1], ft) ;
	tmp.copy(ft) ;
	tmp *= 1.0/w.size() ;
	tmp += Id ;
	phi[t-1].multilinInterp(tmp, phi[t]) ;
	tmp.copy(ft) ;
	tmp *= -1.0/w.size() ;
	tmp += Id ;
	tmp.multilinInterp(psi[t-1], psi[t]) ;
      }
    }    
  

    /**
       generates template time evolution
    */
    void getTemplate(TimeVector &res) 
    {
      VectorMap Id, tmp, ph ;
      res.resize(w.size()+1, w.d) ;
      res[0].copy(J[0]) ;
      Id.idMesh(w.d) ;
      ph.copy(Id) ;
      
      for(unsigned t=1; t<res.size(); t++) {
	kernel(w[t-1], tmp) ;
	tmp *= 1.0/w.size() ;
	tmp += Id ;
	ph.multilinInterp(tmp, ph) ;
	ph.multilinInterp(J[t], res[t]) ;
      }
    }    
  
    void geodesicShooting(Vector& Z1, deformableImage &It)
    {
      VectorMap vt, foo, foo2, semi, id ;
      Vector Zt, Jt, foov, invJac, jacPhi, jacPsi;
      deformableImage tmpI ;
      It.copy(Template) ;
      tmpI.copy(It) ;
      convImage(It.img(), tmpI.img()) ;
      tmpI.computeGradient(param.spaceRes,param.gradientThreshold) ;
      Jt.copy(It.img()) ;
      Zt.copy(Z1) ;
      foo.copy(tmpI.gradient()) ;
      foo *= Zt ;
      foo *= -1 ;
      kernel(foo, foo2) ;
      kernel(foo2, vt) ;
      
      id.idMesh(Z1.d) ;
      jacPsi.ones(Z1.d) ;
      _phi.copy(id) ;
      _psi.copy(id) ;
      unsigned int T = param.time_disc ;
      _real dt = 1.0 / T ;
      
      int k ;
      for (k=0; k<3; k++) {
	semi.copy(vt) ;
	semi *= -dt/2 ;
	semi += id ;
	semi.multilinInterp(foo, foo2) ;
	kernel(foo2, vt) ;
	vt *= -1 ;
      }
      
      for (unsigned int t=0; t<T; t++) {
	semi.copy(vt) ;
	semi *= dt ;
	for (unsigned int jj = 0; jj <0; jj++) {
	  foo.copy(semi) ;
	  foo /= -2 ;
	  foo += id ;
	  foo.multilinInterp(vt, semi) ;
	  semi *= dt ;
	}
	
	divergence(vt, foov, param.spaceRes) ;
	foov *= -dt ;
	foov += 1 ;
	
	foo.copy(id) ;
	foo +=  semi ;
	_phi.multilinInterp(foo, foo2) ;
	_phi.copy(foo2) ;
	foo.copy(id) ;
	foo -=  semi ;
	foo.multilinInterp(_psi, foo2) ;
	_psi.copy(foo2) ;
	foo.multilinInterp(jacPsi, jacPsi) ;
	jacPsi *= foov ;
	
	_psi.multilinInterp(Z1, Zt) ;
	Zt *= jacPsi ;
	
	_phi.multilinInterp(jacPsi, invJac) ;
	foov.copy(Z1) ;
	foov *= invJac ;
	foov *= dt /param.lambda ;
	Jt += foov ;
	_psi.multilinInterp(Jt, It.img()) ;
	convImage(It.img(), tmpI.img()) ;

	tmpI.computeGradient(param.spaceRes,param.gradientThreshold) ;
	foo.copy(tmpI.gradient()) ;
	foo *= Zt ;
	kernel(foo, vt) ;

	vt *= -1 ;
	
	double nv = sqrt(vt.norm2()) ;
	if (nv > 1e100) {
	  cout << "large vt norm" << vt.norm2() << endl ;
	  cout << "stopping shooting" << endl ;
	  return ;
	}


	for (k=0; k<3; k++) {
	  semi.copy(vt) ;
	  semi *= -dt/2 ;
	  semi += id ;
	  semi.multilinInterp(foo, foo2) ;
	  kernel(foo2, vt) ;
	  vt *= -1 ;
	}
      }
    }


    /**
       save stuff...
    */

    void Print(){
      Print(param.outDir) ;
    }

    void initialPrint(){
      initialPrint(param.outDir) ;
    }

    void initialPrint(char* path)
    {
      char file[256] ;
      sprintf(file, "%s/template", path) ;
      Template.img().write_imagesc(file) ;
      sprintf(file, "%s/target", path) ;
      Target.img().write_imagesc(file) ;
      sprintf(file, "%s/binaryTemplate", path) ;
      Template.img().write(file) ;
    }

    void Print(char* path)
    {
      char file[256] ;
      
      double mx = J[0].maxAbs() ;
      for(unsigned int t=1; t< J.size(); t++) {
	double mx2 = J[t].maxAbs() ;
	if (mx2 > mx)
	  mx = mx2 ;
      }


      cout << "Printing Images" << endl ;
      for(unsigned int t=0; t< J.size(); t++) {
	Vector foo ;
	sprintf(file, "%s/image%03d", path, t) ;
	foo.copy(J[t]) ;
	foo *=255/(mx +0.0001) ;
	foo.write_image(file) ;
      }
      if (param.doDefor == 0)
	return ;
      

      VectorMap y, Id ;
      //  Vector Z ;
      
      Id.idMesh(w.d) ;
      kernel(w[0], y) ;
      y *= 1.0/w.size() ;
      y += Id ;
      
      y.multilinInterp(J[1], Z) ;
      Z -= J[0] ;
      Z *= w.size() * param.lambda ;
      
      cout << "Printing momentum" << endl ;
      sprintf(file, "%s/initialScalarMomentumMeta", path) ;
      Z.write(file) ;
      
      Vector dphi, I11 ;
      /*  cout << "Printing shooted template" << endl ;
	  deformableImage J11 ;
	  //  Z.zeros(Z.d) ;
	  geodesicShooting(Z, J11) ;
	  sprintf(file, "%s/shootedTemplate", path) ;
	  J11.img().write_imagesc(file) ;*/

      _real m = Z.min(), M = Z.max() ;
      
      if (fabs(m) > fabs(M)) {
	Z *= 127 / fabs(m) ;
	Z += 128 ;
      }
      else {
	Z *= 127 / fabs(M) ;
	Z += 128 ;
      }
      sprintf(file, "%s/scaledScalarMomentumMeta", path) ;
      Z.write_image(file) ;
      
    
      TimeVectorMap phi, psi ;
      Vector jacob ;
      TimeVector Tplt ;
      //  getTemplate(Tplt) ;
      //  sprintf(file, "%s/finalTemplate", path) ;
      //  Tplt[Tplt.size()-1].write_imagesc(file) ;
      
      
      cout << "Printing deformed target" << endl ;
      flow(phi, psi) ;
      phi[phi.size()-1].multilinInterp(J[J.size()-1], I11) ;
      if (param.matchDensities) {
	phi[phi.size()-1].jacobian(jacob, param.spaceRes) ;
	I11 *= jacob ;
      }
      sprintf(file, "%s/deformedTarget", path) ;
      I11.write_imagesc(file) ;
      
      cout << "Printing deformed template" << endl ;
      psi[psi.size()-1].multilinInterp(J[0], I11) ;
      if (param.matchDensities) {
	psi[psi.size()-1].jacobian(jacob, param.spaceRes) ;
	I11 *= jacob ;
      }
      sprintf(file, "%s/deformedTemplate", path) ;
      I11.write_imagesc(file) ;

      cout << "Printing matching" << endl ;
      for (unsigned int t=0; t<phi.size(); t++) {
	sprintf(file, "%s/forwardMatching%03d", path, t) ;
	phi[t].write(file) ;
	phi[t].write_grid(file) ;
	sprintf(file, "%s/backwardMatching%03d", path, t) ;
	psi[t].write(file) ;
	psi[t].write_grid(file) ;
      }
      phi[phi.size()-1].logJacobian(dphi, param.spaceRes) ;
      
      m = dphi.min(), M = dphi.max() ;
      
      if (fabs(m) > fabs(M)) {
	dphi *= 127 / fabs(m) ;
	dphi += 128 ;
      }
      else {
	dphi *= 127 / fabs(M) ;
	dphi += 128 ;
      }

      cout << "Printing deformed jacobian" << endl ;
      sprintf(file, "%s/jacobian", path) ;
      dphi.write_image(file) ;

    }




    /**
       Main function for comparing two images
       mainly calls matchingStep
    */
    void morphing()
    {
      TimeVectorMap wIni ;
      
      Domain D(imageDim) ;
      J.zeros(param.time_disc, D) ;
      Z.al(D) ;
      w.resize(param.time_disc, D) ;
      wIni.zeros(w.size(), w.d) ;
      
      int itr= param.nb_iter ;
      bool doInitSmooth = false ;
      if (doInitSmooth) {
	param.nb_iter = 30 ;
	matchingStep(wIni) ;
	
	Template.copy(input1) ;
	Target.copy(input2) ;
	
	param.type_group = param_matching::ID ;
	param.nb_iter = itr ;
      }
      Vector foo ;
      int nbFreeVar=0 ;
      Template.gradient().norm(foo) ;
      for (unsigned int k=0; k<foo.d.length;k++)
	if (foo[k] > 0.01)
	  nbFreeVar++ ;
      Target.gradient().norm(foo) ;
      for (unsigned int k=0; k<foo.d.length;k++)
	if (foo[k] > 0.01)
	  nbFreeVar++ ;
      kernelNormalization = nbFreeVar/2 ; 
      cout << "Number of momentum variables: " << nbFreeVar << " out of " << foo.d.length << endl ;

      matchingStep(w) ;
      
    }


    /**
       Morphing energy: deformation part
    */
    _real enerw()
    {
      _real en = 0 , M = w.length() ;
      for (unsigned int t=0; t<w.size(); t++)
	en += kernelNorm(w[t]) ; 
      en = (1/(param.lambda*M)) * en + energyM() ;
      return en ;
    }


    /**
       Morphing energy: image part
    */
    _real energyM() 
    {
      VectorMap Id, y1, y2, ft ;
      Vector It1, It2, divf1, divf2 ;
      Id.idMesh(w.d) ;
      _real dt = 1.0/w.size() ;
      _real en =0 ;


      for (unsigned int t=0; t<w.size()  ; t++) {
	kernel(w[t], ft) ;
	y1.copy(ft) ;
	y1 *= dt ;
	y1 += Id ;
	y2.copy(ft) ;
	y2 *= -dt ;
	y2 += Id ;
	y1.multilinInterp(J[t+1], It1) ;
	y2.multilinInterp(J[t], It2) ;
	if (param.matchDensities) {
	  divergence(ft, divf1, param.spaceRes) ;
	  divf2.copy(divf1) ;
	  divf1 *= dt ;
	  divf2 *=-dt ;
	  divf1 += 1 ;
	  divf2 += 1 ;
	  It1 *= divf1 ;
	  It2 *= divf2 ;
	}

	It1 -= J[t] ;
	It2 -= J[t+1] ;

	It1 /= dt ;
	It2 /= dt ;
	en += (It1.norm2() + It2.norm2())/2;
      }  
      en /= w.length() ;

      return en ;
    }


    /**
       Conjugate gradient step (deformation)
    */
    void gradientStepC() 
    {
      VectorMap y1, y2, y1Try, y2Try, Id, ft, foo ;
      Id.idMesh(w.d) ;
      _real dt = 1.0/w.size(), step = 5 ;
      VectorMap r, oldr, kr, oldkr, tmp, wtry, foom, GIt  ;
      Vector q0, absG, foo1, foo2, divf, divf1, divf2, divf1Try, divf2Try ;
      VectorMap GI, GI1, GI2 ;
      Vector It, It1, It2, tmpI, It1Try, It2Try ;

      int iter  ;
      int iterMaxC = param.nbCGMeta ;

      _real energy, energy0 ;
      
      for (unsigned int t=0; t<w.size(); t++) {
	kernel(w[t], ft) ; 
	if (param.matchDensities){
	  divergence(ft, divf1, param.spaceRes) ;
	  divf2.copy(divf1) ;
	  divf1 *= dt ;
	  divf1 += 1 ;
	  divf2 *= -dt ;
	  divf2 += 1 ;
	}

	y1.copy(ft) ;
	y1 *= dt ;
	y1 += Id ;
	y2.copy(ft) ;
	y2 *= -dt ;
	y2 += Id ;
    
	y1.multilinInterp(J[t+1], It1) ;
	if (param.matchDensities){
	  foo1.copy(It1) ;
	  It1 *= divf1 ;
	}
	y2.multilinInterp(J[t], It2) ;
	if (param.matchDensities) {
	  foo2.copy(It2) ;
	  It2 *= divf2 ;
	}
	It1 -= J[t] ;
	It2 -= J[t+1] ;
    
	It1 /= dt ;
	It2 /= -dt ;
	energy0 = (kernelNorm(w[t]) + (param.lambda/2) * (It1.norm2()+It2.norm2()))/w.length() ;    

	iter = 0 ;
	step = 5 ;
	bool skipCG = 0 ;
	int failedStep = -1 ;
	while (iter < iterMaxC) {
	  if (iter > 0) {
	    oldr.copy(r) ;
	    oldkr.copy(kr) ;
	  }

	  y1.multilinInterpGradient(J[t+1], GI1) ;
	  if (param.matchDensities)
	    GI1 *= divf1 ;
	  GI1 *= It1 ;
	  
	  y2.multilinInterpGradient(J[t], GI2) ;
	  if (param.matchDensities)
	    GI2 *= divf2 ;
	  GI2 *= It2 ;
	  GI.copy(GI1) ;
	  GI += GI2 ;
    
	  tmp.copy(GI) ;
      
	  if (param.matchDensities) {
	    foo1 *= It1 ;
	    gradient(foo1, GI1, param.spaceRes) ;
	    tmp -= GI1 ;
	    foo2 *= It2 ;
	    gradient(foo2, GI2, param.spaceRes) ;
	    tmp -= GI2 ;
	  }
    
	  r.copy(tmp) ;
	  r *= param.lambda/2 ;
	  r += w[t] ;
	  kernel(r, kr) ;

	  if (iter > 0 && skipCG == 0) {
	    double b ;
	    foo.copy(r) ;
	    foo -= oldr ;
	    b = kr.scalProd(foo)/ oldr.scalProd(oldkr) ;
	    if (b > 0) {
	      oldr *= b ;
	      r += oldr ;
	    }
	    else {
	      skipCG = 1 ;
	      if (param.verb == true)
		cout << "Negative b in CG: " << b << endl ;
	    } 
	  }
  
 
	  energy = 2*energy0 + 1 ;
	  while(energy > energy0 *(1 - 1e-10 + param.tolGrad) && step > 1e-10) {
	    wtry.copy(r) ;
	    wtry *= -step ;
	    wtry += w[t] ;
	    kernel(wtry, ft);
	    y2Try.copy(ft) ;
	    y2Try *= -dt ;
	    y2Try += Id ;
	    y1Try.copy(ft) ;
	    y1Try *= dt ;
	    y1Try += Id ;
	    y1Try.multilinInterp(J[t+1], It1Try) ;
	    y2Try.multilinInterp(J[t], It2Try) ;
	    foo1.copy(It1Try) ;
	    foo2.copy(It2Try) ;
	    if (param.matchDensities) {
	      divergence(ft, divf1Try, param.spaceRes) ;
	      divf2Try.copy(divf1Try) ;
	      divf1Try *= dt ;
	      divf1Try += 1 ;
	      divf2Try *= -dt ;
	      divf2Try += 1 ;
	      It1Try *= divf1Try ;
	      It2Try *= divf2Try ;
	    }
	    
	    It1Try -= J[t] ;
	    It2Try -= J[t+1] ;
	    
	    It1Try /= dt ;
	    It2Try /= -dt ;
	    
	    energy = (kernelNorm(wtry) + (param.lambda/2) * (It1Try.norm2()+It2Try.norm2()))/w.length() ;
	
	    if (energy > (1 - 1e-10 + param.tolGrad) * energy0) {
	      step = step * 0.5 ;
	    }
	  }
	  
	  if (energy > (1-1e-10)*energy0 || step < 1e-10 ) {
	    if (skipCG == 1) {
	      failedStep = iter ;
	      iter = iterMaxC ;
	    }
	    else
	      skipCG = 1 ;
	  }
	  else {
	    iter = iter + 1;
	    skipCG = 0 ;
	  }
	  
	  if (energy < (1 - 1e-10+ param.tolGrad)*energy0) {
	    w[t].copy(wtry) ;
	    y1.copy(y1Try) ;
	    y2.copy(y2Try) ;
	    It1.copy(It1Try) ;
	    It2.copy(It2Try) ;
	    if (param.matchDensities) {
	      divf1.copy(divf1Try) ;
	      divf2.copy(divf2Try) ;
	    }
	    energy0 = energy ;
	    step *=2 ;
	  }
	  if (param.verb && iter==iterMaxC) {
	    cout << " V gradient: " << iter << " " << step << " " << energy0 << " " << r.scalProd(kr)  ;
	    if (failedStep >= 0)
	      cout << " (descent stuck at " << failedStep << ")" <<  endl ;
	    else
	      cout << endl ;
	  }
	}
      }
    }


    /**
       Optimal image interpolation along the flow
    */
    void interpolate() 
    {
      // interpolated image
      J.zeros(w.size() + 1, w.d) ;
      
      
      Vector tmp ;
      tmp.copy(Template.img()) ;
      tmp -= Target.img() ;
      
      for (unsigned int t=0; t< J.size(); t++) { 
	_real r = (_real) t/w.size() ;
	Vector tmp ;
	tmp.copy(Template.img()) ;
	tmp *= 1-r ;
	J[t].copy(Target.img()) ;
	J[t] *= r ;
	J[t] += tmp ;
      }
      
      smoothIConj(200, 0.001) ;
    }
    
    

    /**
       matching two images
    */
    _real matchingStep(TimeVectorMap &wIni) 
    {
      Vector tmp ;
      w.copy(wIni) ;
      
      // initialization
      J.zeros(w.size()+1, w.d) ;
      w.zeros(w.size(), w.d) ;
      Z.zeros(w.d) ;
      
      VectorMap Id ;
      Id.idMesh(w.d) ;
      interpolate() ;
      
      if (param.doDefor == 0)
	return 0 ;
      _real enOld = enerw(), en = 0 ;
      _real test = 0, tol0 = param.tolGrad ;
      int it = 1 ;
      
      while (test <  1 && it < param.nb_iter) {
	cout << "iteration " << it << endl ;
	Print() ;
	
	for (int cnt=0; cnt < 1; cnt++) {
	  gradientStepC() ;
	  cout << endl ;
	} 
	
	cout << "energy after gradient: " << enerw() << endl ; ; 
	smoothIConj(20, -1) ;
	cout << "energy after CG: " << enerw() << endl ; ; 
	
	
	if (it >0 && it % 1 == 0) 
	  timeChange() ; 
	
	en = enerw() ;
	
	cout << "energy before and after iteration: " <<  enOld << " " << en << endl ; ; 
	if (enOld-en < 0.00001*en)
	  test = test + 0.25 ;
	else
	  test = 0 ;
	
	enOld = en ;
	
	if (param.tolGrad > 0.00000001)
	  param.tolGrad = tol0 / (1 + 10*it) ;
	it = it+1 ;
      }
      
      return en ;
    }
    

    /**
       Image interpolation along v by conjugate gradient
    */
    void smoothIConj(int nbStep, _real error)
    {
      VectorMap y1, y2, y1old, y2old, ft ;
      VectorMap Id ;
      Vector divf1, divf1old, divf2, divf2old ;
      
      Id.idMesh(w.d) ;
      _real dt = 1.0/w.size() ;
      int T = J.size()-1 ;
      
      if (error < 0)
	error = 0.000001 ;
      
      
      // Compute b
      Vector tmp1, tmp2, tmp3 ;
      TimeVector p, r, q, x ;
      TimeVectorMap f ;
      _real mu ;
      p.zeros(J.size(), J.d) ;
      q.zeros(J.size(), J.d) ;
      r.zeros(J.size(), J.d) ;
      
      f.resize(w.size(), w.d) ;
      for (unsigned int t=0; t<w.size(); t++)
	kernel(w[t], f[t]) ; 
      y1.copy(f[T-1]) ;
      y1 *= dt ;
      y1 += Id ;
      y2.copy(f[T-1]) ;
      y2 *= -dt ;
      y2 += Id ;
      y1.multilinInterp(J[T], tmp1);
      tmp2.copy(J[T]) ;
      if (param.matchDensities) {
	divergence(f[T-1],divf1, param.spaceRes) ;
	divf2.copy(divf1) ;
	divf1 *= dt ;
	divf1 += 1 ;
	divf2 *= -dt ;
	divf2 += 1 ;
	tmp1 *= divf1 ;
	tmp2 *= divf2 ;
      }
      r[T-1].copy(tmp1) ;
      y2.multilinInterpDual(tmp2, tmp1);
      r[T-1] += tmp1 ;
      
      y1old.copy(f[0]) ;
      y1old *= dt ;
      y1old += Id ;
      y2old.copy(f[0]) ;
      y2old *= -dt ;
      y2old += Id ;
      y2old.multilinInterp(J[0], tmp1) ;
      tmp2.copy(J[0]) ;
      if (param.matchDensities) {
	divergence(f[0],divf1old, param.spaceRes) ;
	divf2old.copy(divf1old) ;
	divf1old *= dt ;
	divf1old += 1 ;
	divf2old *= -dt ;
	divf2old += 1 ;
	tmp1 *= divf2old ;
	tmp2 *= divf1old ;
      }
      r[1].copy(tmp1) ;
      y1old.multilinInterpDual(tmp2, tmp1) ;
      r[1] += tmp1 ;
      

      //conjugate gradient loop
      
      for (int t=1; t<T; t++) {
	y1.copy(f[t]) ;
	y1 *= dt ;
	y1 += Id ;
	y2.copy(f[t]) ;
	y2 *= -dt ;
	y2 += Id ;
	if (param.matchDensities) {
	  divergence(f[t],divf1, param.spaceRes) ;
	  divf2.copy(divf1) ;
	  divf1 *= dt ;
	  divf1 += 1 ;
	  divf2 *= -dt ;
	  divf2 += 1 ;
	}
	
	tmp1.copy(J[t]) ;
	tmp1 *= 2 ;  
	y2.multilinInterp(J[t], tmp2) ;
	if (param.matchDensities) {
	  tmp2 *= divf2 ;
	  tmp2 *= divf2 ;
	}
	y2.multilinInterpDual(tmp2, tmp3) ;
	tmp1 += tmp3 ; 
	
	y1old.multilinInterp(J[t], tmp2) ;
	if (param.matchDensities) {
	  tmp2 *= divf1old ;
	  tmp2 *= divf1old ;
	}
	y1old.multilinInterpDual(tmp2, tmp3) ;
	tmp1 += tmp3 ;
	
	if (t < T-1) {
	  y1.multilinInterp(J[t+1], tmp2) ;
	  if (param.matchDensities) {
	    tmp2 *= divf1 ;
	  }
	  tmp1 -= tmp2 ;
	  
	  tmp2.copy(J[t+1]) ;
	  if (param.matchDensities) {
	    tmp2 *= divf2 ;
	  }
	  y2.multilinInterpDual(tmp2, tmp3) ;
	  tmp1 -= tmp3 ;
	}
	
	if (t > 1) {
	  y2old.multilinInterp(J[t-1], tmp2) ;
	  if (param.matchDensities) {
	    tmp2 *= divf2old ;
	  }
	  tmp1 -= tmp2 ;
	  tmp2.copy(J[t-1]) ;
	  if (param.matchDensities) {
	    tmp2 *= divf1old ;
	  }
	  y1old.multilinInterpDual(tmp2, tmp3) ;
	  tmp1 -= tmp3 ;
	}
	
	r[t] -= tmp1 ;
	y1old.copy(y1) ;
	divf1old.copy(divf1) ;
	y2old.copy(y2) ;
	divf2old.copy(divf2) ;
      }
      

      _real energy, energy0 = energyM() ;
      _real energyOld = energy0 ;
      _real muOld = 0, alpha, beta ;
      for (int ss = 0; ss <nbStep; ss++) {
	mu = r.norm2(1, T-1) ;
	if (ss == 0)
	  p.copy(r) ;
	else {
	  beta = mu/muOld ;
	  p *= beta ;
	  p += r ;
	}
	
	y1old.copy(f[0]) ;
	y1old *= dt ;
	y1old += Id ;
	y2old.copy(f[0]) ;
	y2old *= -dt ;
	y2old += Id ;
	
	if (param.matchDensities) {
	  divergence(f[0],divf1old, param.spaceRes) ;
	  divf2old.copy(divf1old) ;
	  divf1old *= dt ;
	  divf1old += 1 ;
	  divf2old *= -dt ;
	  divf2old += 1 ;
	}
	
	for (int t=1; t<T; t++) {
	  y1.copy(f[t]) ;
	  y1 *= dt ;
	  y1 += Id ;
	  y2.copy(f[t]) ;
	  y2 *= -dt ;
	  y2 += Id ;
	  if (param.matchDensities) {
	    divergence(f[t],divf1, param.spaceRes) ;
	    divf2.copy(divf1) ;
	    divf1 *= dt ;
	    divf1 += 1 ;
	    divf2 *= -dt ;
	    divf2 += 1 ;
	  }
	  
	  tmp1.copy(p[t]) ;
	  tmp1 *= 2 ;  
	  y2.multilinInterp(p[t], tmp2) ;
	  if (param.matchDensities) {
	    tmp2 *= divf2 ;
	    tmp2 *= divf2 ;
	  }
	  y2.multilinInterpDual(tmp2, tmp3) ;
	  tmp1 += tmp3 ; 
	  
	  y1old.multilinInterp(p[t], tmp2) ;
	  if (param.matchDensities) {
	    tmp2 *= divf1old ;
	    tmp2 *= divf1old ;
	  }
	  y1old.multilinInterpDual(tmp2, tmp3) ;
	  tmp1 += tmp3 ;
	  
	  if (t < T-1) {
	    y1.multilinInterp(p[t+1], tmp2) ;
	    if (param.matchDensities) {
	      tmp2 *= divf1 ;
	    }
	    tmp1 -= tmp2 ;
	    
	    tmp2.copy(p[t+1]) ;
	    if (param.matchDensities) {
	      tmp2 *= divf2 ;
	    }
	    y2.multilinInterpDual(tmp2, tmp3) ;
	    tmp1 -= tmp3 ;
	  }
	  
	  if (t > 1) {
	    y2old.multilinInterp(p[t-1], tmp2) ;
	    if (param.matchDensities) {
	      tmp2 *= divf2old ;
	    }
	    tmp1 -= tmp2 ;
	    tmp2.copy(p[t-1]) ;
	    if (param.matchDensities) {
	      tmp2 *= divf1old ;
	    }
	    y1old.multilinInterpDual(tmp2, tmp3) ;
	    tmp1 -= tmp3 ;
	  }
	  
	  
	  q[t].copy(tmp1) ;
	  
	  y1old.copy(y1) ;
	  divf1old.copy(divf1) ;
	  y2old.copy(y2) ;
	  divf2old.copy(divf2) ;
	}
	
	alpha = mu /(p.sumScalProd(q, 1, T-1) + 1e-10) ;
	for (int t=1; t<T; t++) {
	  tmp1.copy(p[t]) ;
	  tmp1 *= alpha ;
	  J[t] += tmp1 ;
	  tmp1.copy(q[t]) ;
	  tmp1 *= alpha ;
	  r[t] -= tmp1 ;
	}
	muOld = mu ;
	energy = energyM() ;
	
	if (param.verb)
	  cout << "step " << ss << " conjugate gradient: " << energy0 << " " << energy << endl ;
	if (abs(energyOld - energy) < error * energyOld)
	  break ;
	energyOld = energy ;
      }
    }
    

    /** 
	Tries to rescale time to ensure constant velocity norm
	accepts only if energy is reduced
    */
    void timeChange()
    {
      VectorMap wt, ft  ;
      Vector It1, It2, divf1, divf2 ;
      VectorMap y1, y2, Id ;
      TimeVectorMap wtry, w2 ;
      TimeVector Itry, J2 ;
      _real dt = 1.0/w.size(), step = 1, q0 = 0, entot = 0 ;
      std::vector<_real> ener, q ;
      //  f.zeros(w.size(), w.d) ;
      Id.idMesh(w.d) ;
      ener.resize(w.size()) ;
      q.resize(w.size()+1) ;
      
      q[0] = 0 ;
      for (unsigned int t=0; t<w.size(); t++) {
	kernel(w[t], ft) ; 
	y1.copy(ft) ;
	y1 *= dt ;
	y1 += Id ;
	y2.copy(ft) ;
	y2 *= -dt ;
	y2 += Id ;
	
	y1.multilinInterp(J[t+1], It1) ;
	y2.multilinInterp(J[t], It2) ;
	if (param.matchDensities) {
	  divergence(ft, divf1, param.spaceRes) ;
	  divf2.copy(divf1) ;
	  divf1 *= dt ;
	  divf1 += 1 ;
	  It1 *= divf1 ;
	  divf2 *= -dt ;
	  divf2 += 1 ;
	  It2 *= divf2 ;
	}
	
	It1 -= J[t] ;
	It2 -= J[t+1] ;
	It1 /= dt ;
	It2 /= dt ;
	
	ener[t] = (kernelNorm(w[t])/param.lambda + (It1.norm2() + It2.norm2())/2)/w.length() ;
	entot += ener[t] ;
	ener[t] = sqrt(ener[t]) ;
	q[t+1] = q[t] + ener[t] ;
      }
      
      q0 = q[q.size()-1] ;
      for (unsigned t=0; t<q.size(); t++)
	q[t] /= q0 ;
      
      wtry.resize(w.size(), w.d) ;
      Itry.resize(J.size(), J.d) ;
      
      Itry[0].copy(J[0]) ;
      Itry[J.size()-1].copy(J[J.size()-1]) ;
      
      unsigned int j = 1 ;
      for(unsigned int t=1; t<J.size()-1; t++) {
	while(j < J.size() && q[j] < t*dt)
	  j++ ;
	if (j<J.size()) {
	  Vector tmp ;
	  _real r = (q[j] - t*dt)/(q[j] - q[j-1]) ;
	  Itry[t].copy(J[j-1]) ;
	  Itry[t] *= r ;
	  tmp.copy(J[j]) ;
	  tmp *= 1-r ;
	  Itry[t] += tmp ;
	}
      }
      
      j=1 ; 
      for(unsigned int t=0; t<J.size()-1; t++) {
	while(j < J.size()-1 && q[j] < t*dt)
	  j++ ;
	VectorMap tmp ;
	if(j== J.size() -1) {
	  wtry[t].copy(w[j-1]) ;
	  wtry[t] *= dt * q0/ener[j-1] ;
	}
	else {
	  _real r = (q[j] - t*dt)/(q[j] - q[j-1]) ;
	  wtry[t].copy(w[j-1]) ;
	  wtry[t] *= dt * q0/ener[j-1] ;
	  wtry[t] *= r ;
	  tmp.copy(w[j]) ;
	  tmp *= dt * q0/ener[j] ;
	  tmp *= 1-r ;
	  wtry[t] += tmp ;
	}
      }
      
      _real ener2 ;
      do {
	ener2 = 0 ;
	J2.copy(Itry) ;
	J2 -= J ;
	J2 *= step ;
	J2 += J ;
	w2.copy(wtry) ;    
	w2 -= w ;
	w2 *= step ;
	w2 += w ;
	for (unsigned int t=0; t<w.size(); t++) {
	  VectorMap wt  ;
	  Vector It, divf ;
	  
	  wt.copy(w2[t]) ;
	  kernel(wt, ft) ; 
	  
	  y1.copy(ft) ;
	  y1 *= dt ;
	  y1 += Id ;
	  y2.copy(ft) ;
	  y2 *= -dt ;
	  y2 += Id ;
	  
	  y1.multilinInterp(J2[t+1], It1) ;
	  y2.multilinInterp(J2[t], It2) ;
	  if (param.matchDensities) {
	    divergence(ft, divf1, param.spaceRes) ;
	    divf2.copy(divf1) ;
	    divf1 *= dt ;
	    divf1 += 1 ;
	    It1 *= divf1 ;
	    divf2 *= -dt ;
	    divf2 += 1 ;
	    It2 *= divf2 ;
	  }
	  
	  It1 -= J2[t] ;
	  It2 -= J2[t+1] ;
	  It1 /= dt ;
	  It2 /= dt ;
	  
	  ener2 += (kernelNorm(wt)/param.lambda + 0.5*  (It1.norm2() + It2.norm2()))/w.length() ;
	}
	
	if (ener2 > entot)
	  step *= 0.5 ;
      } while(ener2 > entot && step > 0.1) ;
      
      if (ener2 < entot) {
	w.copy(w2) ;
	J.copy(J2) ;
	if (param.verb)
	  cout << "updating w and J " << endl ;
      }
    }
    
    
    ~Morphing(){
      /*     fftw_destroy_plan(_ptoImage) ; fftw_destroy_plan(_pfromImage) ; fftw_free(_inImage) ; fftw_free(_outImage) ; */
      /*     fftw_destroy_plan(_ptoMap) ; fftw_destroy_plan(_pfromMap) ; fftw_free(_inMap) ; fftw_free(_outMap) ; */
      /*     fftw_free(_fImageKernel) ; fftw_free(_fMapKernel) ; */
    }
    
};

#endif
