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
#include "optimF.h"
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
  std::vector<double> step ;
  double _epsBound ;

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
  
  void flowDual(TimeVectorMap &phi, TimeVectorMap &psi)
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
      tmp *= -1.0/w.size() ;
      tmp += Id ;
      psi[t-1].multilinInterp(tmp, psi[t]) ;
      tmp.copy(ft) ;
      tmp *= 1.0/w.size() ;
      tmp += Id ;
      tmp.multilinInterp(phi[t-1], phi[t]) ;
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
      // for (unsigned int jj = 0; jj <0; jj++) {
      // 	foo.copy(semi) ;
      // 	foo /= -2 ;
      // 	foo += id ;
      // 	foo.multilinInterp(vt, semi) ;
      // 	semi *= dt ;
      // }
	
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

  void geodesicEvolution(const Vector& Z1, TimeVector &It)
  {
    VectorMap vt, id, y1, y2, grd ;
    Vector Zt, foo, foo2;

    Zt.copy(Z1) ;
    unsigned int T = param.time_disc ;
    _real dt = 1.0 / T ;
    It.resize(T+1, Z1.d) ;
    It[0].copy(Template.img()) ;
    id.idMesh(Z1.d) ;
    _phi.copy(id) ;
    _psi.copy(id) ;


    for (unsigned int t=0; t<T; t++) {
      gradient(It[t], grd, param.spaceRes) ;
      vt.copy(grd) ;
      vt *= Zt ;
      vt *= -1 ;
      kernel(vt, vt) ;
      //cout << dt * vt.maxNorm() << endl ;
      addProduct(id, dt, vt, y1) ;
      addProduct(id, -dt, vt, y2) ;
      _phi.multilinInterp(y1, _phi) ;
      y2.multilinInterp(_psi, _psi) ;
      y2.multilinInterp(It[t], foo2) ;
      foo.copy(Zt) ;
      foo *= dt/param.lambda ;
      foo2 += foo ;
      It[t+1].copy(foo2) ;
      y1.multilinInterpDual(Zt, Zt) ;
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
    if (_kern.initialized() == false) {
      _kern.setParam(param) ;
      _kern.initFFT(imageDim, FFTW_MEASURE) ;
    }
    sprintf(file, "%s/kernel", path) ;
    _kern.kern.write_imagesc(file) ;
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

    if (param.verb)
      cout << "Printing Images" << endl ;
    for(unsigned int t=0; t< J.size(); t++) {
      Vector foo ;
      sprintf(file, "%s/image%03d", path, t) ;
      foo.copy(J[t]) ;
      //      cout << foo.sum() << " " << foo.max() << endl ;
      foo *=255/(mx +0.0001) ;
      foo.write_image(file) ;
    }
    if (param.doDefor == 0)
      return ;
      

    VectorMap y, Id ;
    //  Vector Z ;
      
    Id.idMesh(w.d) ;
    kernel(w[0], y) ;
    y /= w.size() ;
    y += Id ;
      
    y.multilinInterp(J[1], Z) ;
    Z -= J[0] ;
    Z *= w.size() * param.lambda ;
      
    if (param.verb)
      cout << "Printing momentum" << endl ;
    sprintf(file, "%s/initialScalarMomentumMeta", path) ;
    Z.write(file) ;
      
    Vector dphi, I11 ;
    /*
    cout << "Printing shooted template" << endl ;
    TimeVector J11 ;
    //  Z.zeros(Z.d) ;
    geodesicEvolution(Z, J11) ;
    sprintf(file, "%s/shootedTemplate", path) ;
    for(unsigned int t=0; t< J11.size(); t++) {
      Vector foo ;
      sprintf(file, "%s/shootedImage%03d", path, t) ;
      foo.copy(J11[t]) ;
      //      foo *=255/(mx +0.0001) ;
      foo.write_imagesc(file) ;
    }
    */
    //    J11.img().write_imagesc(file) ;

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
      
    if (param.verb)     
      cout << "Printing deformed target" << endl ;
    //flowDual(psi, phi) ;
    if (param.matchDensities) {
      flowDual(phi, psi) ;
      psi[psi.size()-1].multilinInterpDual(J[J.size()-1], I11) ;
    }
    else {
      flow(phi, psi) ;
      phi[phi.size()-1].multilinInterp(J[J.size()-1], I11) ;
    }
    sprintf(file, "%s/deformedTarget", path) ;
    I11.write_imagesc(file) ;
    if (param.verb)
      cout << "Target: " << J[J.size()-1].sum() << " " << I11.sum() << endl ;

    if(param.verb)      
      cout << "Printing deformed template" << endl ;
    if (param.matchDensities) 
      phi[phi.size()-1].multilinInterpDual(J[0], I11) ;
    else
      psi[psi.size()-1].multilinInterp(J[0], I11) ;

    sprintf(file, "%s/deformedTemplate", path) ;
    I11.write_imagesc(file) ;
    if (param.verb)
      cout << "Template: " << J[0].sum() << " " << I11.sum() << endl ;

    if (param.verb)
      cout << "Printing matching" << endl ;
    for (unsigned int t=0; t<phi.size(); t++) {
      sprintf(file, "%s/forwardMatching%03d", path, t) ;
      phi[t].write(file) ;
      if (param.ndim==1)
	phi[t].write_grid(file) ;
      else
	psi[t].write_grid(file) ;
      sprintf(file, "%s/backwardMatching%03d", path, t) ;
      psi[t].write(file) ;
      if (param.ndim == 1)
	psi[t].write_grid(file) ;
      else
	phi[t].write_grid(file) ;
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

    if (param.verb)
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
    w.resize(param.time_disc, D) ;
    wIni.zeros(w.size(), w.d) ;
    step.resize(w.size()) ;
    for (unsigned int t=0; t<w.size(); t++)
      step[t] = 0.01 ;
      
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
    if (param.verb) 
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
      addProduct(Id, dt, ft, y1) ;
      addProduct(Id, -dt, ft, y2) ;
      // y1.copy(ft) ;
      // y1 *= dt ;
      // y1 += Id ;
      // y2.copy(ft) ;
      // y2 *= -dt ;
      // y2 += Id ;
      if (param.matchDensities) {
	y2.multilinInterpDual(J[t+1], It1) ;
	y1.multilinInterpDual(J[t], It2) ;
      }
      else {
	y1.multilinInterp(J[t+1], It1) ;
	y2.multilinInterp(J[t], It2) ;
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

  // void get_increment_forward(const VectorMap &y, const Vector &Jcur, const Vector & Jnext, Vector &It) {
  //   double dt = 1.0/w.size() ;
  //   y.multilinInterp(Jnext, It) ;
  //   It -= Jcur ;
  //   It /= dt ;
  // }

  // void get_increment_backward(const VectorMap &y, const Vector &Jcur, const Vector & Jnext, Vector &It) {
  //   double dt = 1.0/w.size() ;
  //   y.multilinInterp(Jcur, It) ;
  //   It -= Jnext ;
  //   It /= -dt ;
  // }

  double norm_increment_forward(const VectorMap &v, const Vector &Jcur, const Vector & Jnext) {
    Vector It ;
    VectorMap y, Id ;
    double dt = 1.0/param.time_disc ;

    Id.idMesh(v.d) ;
    if (param.matchDensities) {
      addProduct(Id, -dt, v, y) ;
      y.multilinInterpDual(Jnext, It) ;
    }
    else {
      addProduct(Id, dt, v, y) ;
      y.multilinInterp(Jnext, It) ;
    }

    //cout << "forward" << endl;    
    It -= Jcur ;
    It /= dt ;
    It *= It ;
    return It.sum() ;
  }

  double norm_increment_backward(const VectorMap &v, const Vector &Jcur, const Vector & Jnext) {
    Vector It ;
    VectorMap y, Id ;
    double dt = 1.0/param.time_disc ;

    Id.idMesh(v.d) ;
    if (param.matchDensities){
      addProduct(Id, dt, v, y) ;
      y.multilinInterpDual(Jcur, It) ;
    }
    else {
      addProduct(Id, -dt, v, y) ;
      y.multilinInterp(Jcur, It) ;
    }
      
      
    It -= Jnext ;
    It /= -dt ;
    It *= It ;
    return It.sum() ;
  }

  class scalProdDef {
  public:
    scalProdDef(){};
    scalProdDef(Morphing *sh){_sh = sh ;}
    double operator()(VectorMap &w, VectorMap &ww){
      VectorMap Kw ;
      _sh->kernel(w, Kw) ;
      return Kw.scalProd(ww);
    }
  private:
    Morphing *_sh ;
  } ;

  class scalProdImg {
  public:
    scalProdImg(){};
    double operator()(const TimeVector &w, const TimeVector &ww){
      return w.sumScalProd(ww);
    }
  } ;

    
  class optimDefPart: public optimFunBase<VectorMap, VectorMap > {
  public:
    virtual ~optimDefPart(){};
    optimDefPart(Morphing *sh) {
      _sh = sh ;
    }
    void setImages(Vector *Jcur, Vector *Jnext) {
      _Jcur = Jcur ;
      _Jnext = Jnext ;
      //      cout << Jcur->d<< endl ;
    }

    double epsBound() {
      return _sh->_epsBound ;
    }

    double computeGrad(VectorMap &w, VectorMap& grad){
      Vector It1, It2, divf, foo1, foo2, div1, div2 ;
      VectorMap ft, y1, y2, GI1, GI2, GI, tmp, Id;
      double dt ;
      // if (_sh->param.matchDensities)
      // 	dt = -1.0/_sh->w.size() ;
      // else
      dt = 1.0/_sh->w.size() ;
      //cout << "in compute grad" << endl ;
      _sh->kernel(w, ft) ;
      //      cout << "compute increments" << endl;
      Id.idMesh(ft.d) ;
      addProduct(Id, dt, ft, y1) ;
      addProduct(Id, -dt, ft, y2) ;

      if (_sh->param.matchDensities){
	y2.multilinInterpDual(*_Jnext, It1) ;
	y1.multilinInterpDual(*_Jcur, It2) ;
      }
      else {
	y1.multilinInterp(*_Jnext, It1) ;
	y2.multilinInterp(*_Jcur, It2) ;
      }
      It1 -= *_Jcur ;
      It1 /= dt ;

      It2 -= *_Jnext ;
      It2 /= -dt ;

      //_sh->get_increment_forward(ft, *_Jcur, *_Jnext, It1) ;
      //_sh->get_increment_backward(y2, *_Jcur, *_Jnext, It2) ;
      // cout << "done" << endl ;

      if (_sh->param.matchDensities) {
	y2.multilinInterpGradient(It1, GI1) ;
	y1.multilinInterpGradient(It2, GI2) ;
	GI1 *= *_Jnext ;
	GI2 *= *_Jcur ;
      }
      else { 
	y1.multilinInterpGradient(*_Jnext, GI1) ;
	y2.multilinInterpGradient(*_Jcur, GI2) ;
	GI1 *= It1 ;
	GI2 *= It2 ;
      }
      grad.copy(GI1) ;
      grad += GI2 ;
    
      
      // if (_sh->param.matchDensities) {
      // 	y1.multilinInterp(*_Jnext, foo1) ;
      // 	foo1 *= It1 ;
      // 	gradient(foo1, GI1, _sh->param.spaceRes) ;
      // 	tmp -= GI1 ;
      // 	y1.multilinInterp(*_Jnext, foo2) ;
      // 	foo2 *= It2 ;
      // 	gradient(foo2, GI2, _sh->param.spaceRes) ;
      // 	tmp -= GI2 ;
      // }
    
      //grad.copy(tmp) ;
      if (_sh->param.matchDensities) 
	grad *= -_sh->param.lambda/2 ;
      else
	grad *= _sh->param.lambda/2 ;
      grad += w ;
      return _sh->kernelNorm(grad) ;
    }


    double objectiveFun(VectorMap &w) {
      double res ;
      Vector It1, It2 ;
      VectorMap ft;
      _sh->kernel(w, ft) ;
      //cout << "computing norm" << endl ; 
      res = _sh->norm_increment_forward(ft, *_Jcur, *_Jnext) ;
      res += _sh->norm_increment_backward(ft, *_Jcur, *_Jnext) ;
      res = (w.scalProd(ft) + (_sh->param.lambda/2) * res)/w.length() ;    
      return res ;
    }

    double endOfIteration(VectorMap &w) {
      return -1 ;
    }


    void endOfProcedure(VectorMap &w) {
      if (_sh->param.verb)
	cout << "obj fun: " << objectiveFun(w) << " grad norm " << gradNorm << endl ;
    }
    void startOfProcedure(VectorMap &w) {
      if (_sh->param.verb)
	cout << "obj fun (initial): " << objectiveFun(w) << "    " ;
    }
      //_sh->Z0.copy(Z) ;
      //_sh->Print() ;
    //}

  private:
    Morphing *_sh ;
    Vector *_Jcur, *_Jnext ;
  };

  /**
     Conjugate gradient step (deformation)
  */
  void gradientStepC() 
  {
    VectorMap ww ;
    conjGrad<VectorMap, VectorMap, scalProdDef, optimDefPart> cg ;
    optimDefPart opt(this) ;
    scalProdDef scp(this) ;
    _epsBound = param.epsMax ;
    
    
    for (unsigned int t=0; t<w.size(); t++) {
      opt.setImages(&(J[t]), &(J[t+1])) ;
      //cout << "Starting cg" << endl ;
      cg(opt, scp, w[t], ww, 5, step[t], param.epsMax, param.minVarEn, param.gs, 0) ;
      w[t].copy(ww) ;
      step[t] = cg.getStep() ;
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
      cout << "iteration " << it << flush ;
	
      for (int cnt=0; cnt < 1; cnt++) {
	gradientStepC() ;
	if (param.verb)
	  cout << endl ;
      } 
	
      if (param.verb)
	cout << "energy after gradient: " << enerw() << endl ; ; 
      smoothIConj(20, -1) ;
      if (param.verb)
	cout << "energy after CG: " << enerw() << endl ; ; 
	
	
      // if (it >0 && it % 1 == 0) 
      // 	timeChange() ; 
	
      en = enerw() ;
	
      cout << "   energy: " << en << endl ; ; 
      if (enOld-en < 0.00001*en)
	test = test + 0.25 ;
      else
	test = 0 ;
	
      enOld = en ;
	
      if (param.tolGrad > 0.00000001)
	param.tolGrad = tol0 / (1 + 10*it) ;
      Print() ;
      it = it+1 ;
    }
      
    return en ;
  }
    

  /**
     Image interpolation along v by conjugate gradient
  */


  class ImageVar {
  public:
    ImageVar() {} ;
    ImageVar(Morphing *sh) {_sh = sh;}

    void computeLocalMaps() {
      unsigned int T = _sh->w.size() ;
      double dt = 1.0/T ;
      VectorMap Id, f ;
      y1.resize(T, _sh->w.d) ;
      y2.resize(T, _sh->w.d) ;
      Id.idMesh(_sh->w.d) ;

      for (unsigned int t=0; t<T; t++) {
	_sh->kernel(_sh->w[t], f) ;
	addProduct(Id, dt, f, y1[t]) ; 
	addProduct(Id, -dt, f, y2[t]) ; 
      }
    }

    void operator() (const TimeVector &It, TimeVector &gI) {
      unsigned int T = _sh->w.size() ;
      //cout << T << " " << It.size() << endl ;
      //double dt = 1.0/T ;
      Vector foo1, foo2 ;
      // TimeVector dI1, dI2 ;
      // dI1.resize(T-2, It.d) ;
      // dI2.resize(T-2, It.d) ;
      // //cout << " operator" << endl ;
      // for (unsigned int t=1; t<T-1; t++) {
      // 	y1[t].multilinInterp(It[t], dI1[t-1]) ;
      // 	if (_sh->param.matchDensities)
      // 	  dI1[t-1] *= div1[t] ;
      // 	dI1[t-1] -= It[t-1] ;
      // 	y2[t].multilinInterp(It[t-1], dI2[t-1]) ;
      // 	if (_sh->param.matchDensities)
      // 	  dI2[t-1] *= div2[t] ;
      // 	dI2[t-1] -= It[t] ;
      // 	dI2[t-1] *= -1 ;
      // }

      //cout << "GI" << endl ;
  
      gI.resize(T-1, It.d) ;
      // gI[0].zero() ;
      // for (unsigned int t=1; t<T-1; t++) {
      // 	y1[t].multilinInterpDual(dI1[t-1], gI[t]) ;
      // 	if (_sh->param.matchDensities)
      // 	  gI[t] *= div1[t] ;
      // 	gI[t] += dI2[t-1] ;
      // }
      for (unsigned int t=0; t<T-1; t++) {
	// y2[t].multilinInterpDual(dI2[t], foo1) ;
	// if (_sh->param.matchDensities)
	//   foo1 *= div2[t] ;
	// gI[t] -= foo1 ;
	// gI[t] -= dI1[t] ;
	gI[t].copy(It[t]) ;
	gI[t] *= 2 ;
      }


      /* //Id.idMesh(_sh->w.d) ; */
      for (unsigned int t=0; t<T; t++) {
      	if (t<T-1) {
      	  if (_sh->param.matchDensities) {
	    y2[t].multilinInterpDual(It[t], foo1) ;
	    y2[t].multilinInterp(foo1, foo2) ;
	  }
	  else {
	    y1[t].multilinInterp(It[t], foo1) ;
	    y1[t].multilinInterpDual(foo1, foo2) ;
	  }
	  gI[t] += foo2 ;
	  if (t>0) {
	    gI[t-1] -= foo1 ;
      	    if (_sh->param.matchDensities)
	      y2[t].multilinInterp(It[t-1], foo1) ;
	    else 
	      y1[t].multilinInterpDual(It[t-1], foo1) ;
      	    gI[t] -= foo1 ;
      	  }
      	}

      	if (t >0) {
      	  //y2.multilinInterpDual(It[t-1], foo1) ;
      	  if (_sh->param.matchDensities) {
	    y1[t].multilinInterpDual(It[t-1], foo1) ;
	    y1[t].multilinInterp(foo1, foo2) ;
	  }
	  else {
	    y2[t].multilinInterp(It[t-1], foo1) ;
	    y2[t].multilinInterpDual(foo1, foo2) ;
	  }
      	  gI[t-1] += foo2 ;
      	  if (t < T-1) {
      	    gI[t] -= foo1 ;
	    if (_sh->param.matchDensities)
	      y1[t].multilinInterp(It[t], foo1) ;
	    else
	      y2[t].multilinInterpDual(It[t], foo1) ;
	    gI[t-1] -= foo1 ;
      	  }
      	}
      }
      //cout << "en operator" << endl ;
    }

  private:
    Morphing *_sh ;
    TimeVectorMap y1, y2 ;
    TimeVector div1, div2 ;
  } ;

  void smoothIConj(int nbStep, _real error)
  {
    linearCG<TimeVector, scalProdImg, ImageVar> lcg ;
    scalProdImg scp ;
    ImageVar imv(this) ;
    TimeVector JJ0, JJ, b ;
    unsigned int T = w.size() ;
    double dt = 1.0/T, offset ;
    VectorMap f, y1, y2, Id ;
    Vector foo, divf1, divf2 ;
    // cout << "Initial Image Energy " << 2* energyM()*w.length() / (T*T) << endl ;


    b.zeros(T-1, J.d) ;
    //JJ.resize(T-1, J.d) ;
    JJ0.resize(T-1, J.d) ;
    for (unsigned int t=1; t<T; t++)
      JJ0[t-1].copy(J[t]) ; 

    Id.idMesh(w.d) ;
    kernel(w[0], f) ;
    addProduct(Id, dt, f, y1) ; 
    addProduct(Id, -dt, f, y2) ; 

    if (param.matchDensities)
      y2.multilinInterp(J[0], b[0]) ;
    else 
      y1.multilinInterpDual(J[0], b[0]) ;
    offset = J[0].norm2() ;

    //b[0] *= -1 ;
    if (param.matchDensities)
      y1.multilinInterpDual(J[0], foo) ;
    else
      y2.multilinInterp(J[0], foo) ;
    b[0] += foo ;
    offset += foo.norm2() ;
      
    kernel(w[T-1], f) ;
    addProduct(Id, dt, f, y1) ; 
    addProduct(Id, -dt, f, y2) ; 

    if (param.matchDensities)
      y2.multilinInterpDual(J[T], b[T-2]) ;
    else
      y1.multilinInterp(J[T], b[T-2]) ;
    offset += b[T-2].norm2() ;
    //b[T-2] *= -1 ;
    if (param.matchDensities)
      y1.multilinInterp(J[T], foo) ;
    else
      y2.multilinInterpDual(J[T], foo) ;
    offset += J[T].norm2() ;
    b[T-2] += foo ;
    //cout << "Starting lcg" << endl;
    imv.computeLocalMaps() ;
    double initEn ;
    TimeVector gg ;
    imv(JJ0, gg) ;
    initEn = gg.sumScalProd(JJ0) - 2*b.sumScalProd(JJ0) + offset ;
    if (param.verb)
      cout << "Initial Energy: " << T*T*initEn/(2*w.length()) << " " << energyM() << endl ;
    lcg(JJ0, JJ, b, 20, 1e-2, param.verb, imv, scp) ;
    for (unsigned int t=1; t<T; t++)
      J[t].copy(JJ[t-1]) ; 
    imv(JJ, gg) ;
    initEn = gg.sumScalProd(JJ) - 2*b.sumScalProd(JJ) + offset ;
    if (param.verb)
      cout << "Final Energy: " << T*T*initEn/(2*w.length()) << " " << energyM() << endl ;
    // cout << "Final Image Energy " << 2* energyM()*w.length() / (T*T) << endl ;
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
	
      if (param.matchDensities) {
	y2.multilinInterpDual(J[t+1], It1) ;
	y1.multilinInterpDual(J[t], It2) ;
      }
      else {
	y1.multilinInterp(J[t+1], It1) ;
	y2.multilinInterp(J[t], It2) ;
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
	  
	if (param.matchDensities) {
	  y2.multilinInterpDual(J2[t+1], It1) ;
	  y1.multilinInterpDual(J2[t], It2) ;
	}
	else {
	  y1.multilinInterp(J2[t+1], It1) ;
	  y2.multilinInterp(J2[t], It2) ;
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
