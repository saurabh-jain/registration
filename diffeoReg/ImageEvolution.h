/**
   ImageEvolution.h
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

#ifndef _IMEVOL_
#define _IMEVOL_
#include "ImageMatching.h"
#include "matchingBase.h"
#include "param_matching.h"


/**
   Class providing diffeomorphic evolution functions in image space
*/
class ImageEvolution: public ImageMatching
{
 public:
  using ImageMatching::_kern ;
  using ImageMatching::kernel ;

  /*  using ImageMatching<AE>::Template ;
      using MatchingBase<deformableImage, AE>::kernel ;
      using MatchingBase<deformableImage, AE>::savedTemplate ;
      using MatchingBase<deformableImage, AE>::Target ;
      using MatchingBase<deformableImage, AE>::savedTarget ;
      using MatchingBase<deformableImage, AE>::param ;
      using MatchingBase<deformableImage, AE>::savedParam;
      using MatchingBase<deformableImage, AE>::mask ;
      using MatchingBase<deformableImage, AE>::init_fft ;
      using MatchingBase<deformableImage, AE>::makeMask ;
      using MatchingBase<deformableImage, AE>::inverseKernel ;
      using MatchingBase<deformableImage, AE>::kernelNorm ;
      using MatchingBase<deformableImage, AE>::_psi ;
      using MatchingBase<deformableImage, AE>::_phi ;
      using MatchingBase<deformableImage, AE>::big_adjointStar;
      using MatchingBase<deformableImage, AE>::GeodesicDiffeoEvolution ;
      using MatchingBase<deformableImage, AE>::VarGeodesicDiffeoEvolution ;
      using MatchingBase<deformableImage, AE>::alpha;*/

  ImageEvolution(param_matching &par) {init(par);}
  ImageEvolution(char *file, int argc, char** argv) {init(file, argc, argv);}
  ImageEvolution(char *file, int k) {init(file, k);}
  typedef deformableImage::Tangent Tangent ;
  // Template, initial image momentum, 

  ImageEvolution(){} ;
  virtual void affineReg() {cout << "Warning affineReg: component is not implemented (2)" << endl ;}


  /**
     for multigrid (does not work)
  */
  void restore(Tangent& Z){
    //  using ImageMatching<AE>::Template ;
    Domain D1, D2 ;
    Ivector m, M ;
    D1.copy(Template.domain()) ;
    m.resize(D1.n) ;
    M.resize(D1.n) ;
    for(unsigned int i=0; i<D1.n; i++) {
      m[i] = D1.getm(i) ;
      M[i] = m[i] + (D1.getM(i) - m[i])/2 ;
    }
    D2.create(m, M) ;

    VectorMap v ;
    Tangent ZZ ;
    Template.getMomentum(Z, v) ;
    kernel(v, v) ;
    //param.sigma /= 2 ;

    deformableImage foo ;
    Template.copy(savedTemplate) ;
    Target.copy(savedTarget) ;
    param.sizeKernel = savedParam.sizeKernel ;
    
    VectorMap fooM ;
    v.rescale(Template.domain(), fooM) ;
    v.copy(fooM) ;
    
    mask.al(Template.domain()) ;
    makeMask(1, Template.domain()) ;
    
    //    init_fft() ;
    
    //  inverseKernel(v, fooM) ;

    Template.infinitesimalAction(v, ZZ) ;
    ZZ *= -1 ;
    //  ZZ.write_imagesc("toto2") ;
    
    //  cout << "proj2" << endl ;
    imageTangentProjection(Template, ZZ, Z) ;
  }


  /**
     for multigrid (does not work)
  */
  void downscale(Tangent& Z){
    //  cout << "downscaling" << endl ;

    Domain D1, D2 ;
    Ivector m, M ;
    D1.copy(Template.domain()) ;
    m.resize(D1.n) ;
    M.resize(D1.n) ;
    for(unsigned int i=0; i<D1.n; i++) {
      m[i] = D1.getm(i) ;
      M[i] = m[i] + (D1.getM(i) - m[i])/2 ;
    }
    D2.create(m, M) ;

    deformableImage foo ;
    VectorMap v, foov ;
    Tangent ZZ ;
    Template.getMomentum(Z, foov) ;
    kernel(foov, v) ;
    
    Template.rescale(D2, foo) ;
    Template.copy(foo) ;
    Template.computeGradient(param.spaceRes, param.gradientThreshold) ;
    Target.rescale(D2, foo) ;
    Target.copy(foo) ;
    v.rescale(D2, foov)  ;
    //   Z.rescale(D2, foo.img()) ;
    //    Z.copy(foo.img()) ;
    //    Z /= 2 ;
    //  Template.write_image("toto") ;
    //  Z.write_imagesc("toto2") ;
    //param.sigma *= 2; 

    //  cout << "initializing kernel" << endl ;
    mask.al(D2) ;
    makeMask(1, D2) ;

    //  param.sizeKernel /= 2 ;
    // init_fft() ;
    
    //  cout << "proj1" << endl ;
    //  VectorMap v, dI ;
    //   Template.getMomentum(Z, v) ;
    //   kernel(v, v) ;
    Template.infinitesimalAction(foov, Z) ;
    Z /= 2 ;
    //  ZZ *= -1 ;
    //  ZZ.write_imagesc("toto2") ;
    
    //  cout << "proj2" << endl ;
    //  imageTangentProjection(Template, ZZ, Z) ;
  }

  /**
     for multigrid (does not work)
  */
  void upscale(Tangent& Z){
    Domain D1, D2 ;
    Ivector m, M ;
    VectorMap v, foov ;
    Tangent ZZ ;
    
    D1.copy(Template.domain()) ;
    m.resize(D1.n) ;
    M.resize(D1.n) ;
    for(unsigned int i=0; i<D1.n; i++) {
      m[i] = D1.getm(i) ;
      M[i] = m[i] + (D1.getM(i) - m[i])*2 ;
    }
    D2.create(m, M) ;

    deformableImage foo ;
    Template.getMomentum(Z, foov) ;
    kernel(foov, v) ;
    Template.rescale(D2, foo) ;
    Template.copy(foo) ;
    Template.computeGradient(param.spaceRes, param.gradientThreshold) ;
    Target.rescale(D2, foo) ;
    Target.copy(foo) ;
    v.rescale(D2, foov) ;
    foov *= 2 ;
    mask.al(D2) ;
    makeMask(1, D2) ;
    //  param.sizeKernel *= 2 ;
    //    init_fft() ;
    //  Template.write_image("toto2") ;
    Template.infinitesimalAction(foov, ZZ) ;
    imageTangentProjection(Template, ZZ, Z) ;
  }
  


  /*
    void ImageTangentProjectionInZ(const deformableImage &I, const VectorMap &Km0, VectorMap &Lw, 
    Tangent & finalZ) {
    Vector foov, I1 ;
    Tangent Z, grd, grd1, Z2 ;
    VectorMap foom, id, idw, w, Lw2 ;
    int iterFP = 50 ;

    _real eps = 0.001 ;
    _real sigma = 0.01 ;
    _real rho = 0.0001 ;
    _real gn, gn0 = 1 ;
    //  bool updateLw = true ;
    _real ener, oldEner, ener0 ;

    w.copy(Km0) ;
    inverseKernel(w, Lw) ;
    if (param.verb)
    cout << "norm before projection:" << kernelNorm(Lw) << endl ; 
    kernel(Lw, w) ;
    //  inverseKernel(w, Lw2) ;
    //  cout << "norm before FP:" << kernelNorm(Lw2)  << " " <<  Lw.norm2() << " " << Lw2.norm2() << endl ; 

    w *= eps ;
    Lw *= eps ;
    id.idMesh(Km0.d) ;
    idw.copy(id) ;
    idw -= w;
    idw.multilinInterp(I.img(), I1) ;

    Z.zeros(w.d) ;
    Lw.scalProd(I.gradient(), Z) ;
    double nI = I.gradient().norm2() ;
    Z /=  -(1 + nI) ;
    I.getMomentum(Z, Lw) ;
    kernel(Lw, w) ;
    idw.copy(id) ;
    idw -= w;
    idw.multilinInterp(I.img(), foov) ;
    foov -= I1 ;
    oldEner = Lw.scalProd(w) + foov.norm2()/(sigma*sigma);
    ener0 = oldEner ;
    
    //  w.zero() ;
    //  cout << "start loop" << endl ;
    for (int i=1; i<= iterFP; i++) {
    idw.copy(id) ;
    idw -= w;
    idw.multilinInterp(I.img(), foov) ;
    gradient(foov, foom, param.spaceRes); 
    foov -= I1 ;
    foom *= foov ;
    foom /= sigma*sigma ;
    foom -= Lw ;
    kernel(foom, foom) ;
    I.infinitesimalAction(foom, grd) ;
      
    grd1.copy(grd) ;
    grd1 *= rho  ;
    Z2.copy(Z) ;
    Z2 += grd1 ;
    I.getMomentum(Z2, Lw2) ;
    kernel(Lw2, w) ;
    idw.copy(id) ;
    idw -= w;
    idw.multilinInterp(I.img(), foov) ;
    foov -= I1 ;
    gn = grd.norm2() ;
    if (i==1) {
    gn0 = gn ;
    }
    ener = foov.norm2()/(sigma*sigma) + Lw2.scalProd(w) ;
    while(ener > oldEner && rho > 1e-10) {
    rho /= 2 ;
    grd1.copy(grd) ;
    grd1 *= rho  ;
    Z2.copy(Z) ;
    Z2 += grd1 ;
    I.getMomentum(Z2, Lw2) ;
    kernel(Lw2, w) ;
    idw.copy(id) ;
    idw -= w;
    idw.multilinInterp(I.img(), foov) ;
    foov -= I1 ;
    ener = foov.norm2()/(sigma*sigma) + Lw2.scalProd(w) ;
    //   cout << "i = " << i << " ener = " << ener << " " << oldEner << " " << gn << " " << rho << endl ;
    }
    if (rho < 1e-10)
    break ;
    //    cout << "i = " << i << " ener = " << ener << " " << oldEner << " " << gn << " " << rho << endl ;
    Z.copy(Z2) ;
    Lw.copy(Lw2) ;
    rho *=  2 ;
    if (oldEner - ener < 0.001 *abs(oldEner) || ener < 0.0001)
    break ;
    oldEner = ener ;
    }

    Z /= eps ;
    I.getMomentum(Z, Lw) ;
    kernel(Lw, w) ;
    finalZ.copy(Z) ;
    
    if (param.verb) {
    cout << "gradient norm reduction " << gn/gn0 << endl ;
    cout << "max Z = " << finalZ.maxAbs() << endl ;
    cout << "norm after FP:" << kernelNorm(Lw) << endl ; 
    cout << "energy: " << ener0 << " " << ener << endl ;
    }
    }
  */
    


  /**
     Finds the best normal vector field (relative to I) equivalent to Km0; 
     the result is stored in w.
  */
  /*
    void imageTangentProjection(const deformableImage &I, const VectorMap &m0, VectorMap &Lw) {
    VectorMap gradI, b, foo2, p, q, r, foo, w, Lw0  ;
    Vector Izero ;
    Vector nI  ;
    unsigned int K = 100 ;
    _real mu=0, muold, alfa, be, ener2, ener  ;
    
    //  cout << "projecting momentum on image tangent space" << endl ;
    
    kernel(m0, w) ;
    _real N = sqrt(w.scalProd(m0)) ;
    
    Lw0.copy(m0) ;
    w /= N ;
    //  Template.al(I.d) ;
    Izero.al(I.domain()) ;
    gradI.copy(I.gradient()) ;
    gradI.norm(nI) ;

    for (unsigned int i=0; i< Izero.size(); i++)
    if (nI[i] < param.gradientThreshold) {
    Izero[i] = 0 ;
    }
    else {
    Izero[i] = 1 ;
    }
    

    b.copy(w) ;
    b *= Izero ;

    Lw.copy(m0) ;
    Lw/=N ;
    
    Lw *= Izero ;
    kernel(Lw, w) ;
    w *= Izero ;
    
    _real old_ener ;

    r.copy(b) ;
    r -= w ;
    foo2.copy(w) ;
    foo2 -= b ;
    old_ener = foo2.norm2() ;
    
    if (r.maxNorm() < 0.0000001){
    w*= N ;
    Lw.copy(Lw0) ;
    if (param.verb)
    cout << "CG: nothing to do" << endl ;
    return ;
    }


    for (unsigned int i2 =0; i2<K; i2++) {
    if (i2 == 0) {
    mu = r.norm2() ;
    p.copy(r) ;
    }
    else {
    muold = mu ;
    mu = r.norm2() ;
    be = mu/muold ;
    foo.copy(p) ;
    p *= be ;
    p += r ;
    }

    foo2.copy(p) ;
    foo2 *= Izero ;
    kernel(foo2,q) ;
    q *= Izero ;
    
    alfa = mu / p.scalProd(q) ;
    foo2.copy(p) ;
    foo2 *= alfa ;
    Lw += foo2 ;
    foo2.copy(q) ;
    foo2 *= alfa ;
    r -= foo2 ;
    w += foo2 ;
      
    foo2.copy(w) ;
    foo2 /= 2 ;
    foo2 -= b ;
    ener = Lw.scalProd(foo2) ;
      
    foo2.copy(w) ;
    foo2 -= b ;
    ener2 = foo2.norm2() ;
    //    cout << "iter tangent " << i2 << " ener = " << ener << " " << ener2/old_ener << endl  ;
    if (ener2 < 0.0000001 || ener2 < 0.01 * old_ener) 
    {
    if (param.verb)
    cout << "iter tangent " << i2 << " ener = " << ener2 << " " << old_ener  << endl ;
    w *= N ;
    Lw *= N ;
    Lw *= Izero ;
    return ;
    }
    }
    
    if (param.verb)
    cout << "iter tangent " << K << " ener = " << ener2 << " " << (old_ener-ener2)/fabs(old_ener) << endl ;

    Lw *= Izero ;
    Lw *= N ;
    }


    void normalComponent(const deformableImage &I, const VectorMap &Lw, Tangent &Z){
    Lw.scalProd(I.normal(), Z) ;
    }
  */


  // Solves grad(I).K(gradI.Z) = -ZI
  void imageTangentProjection(const deformableImage &I, const Tangent &ZI, Tangent &Z) {
    VectorMap gradI, normal, foo2, m, Km, foo, Lw , w ;
    Vector b, AZ, IZ,  p, q, r ;
    Vector nI, foo1 ;
    unsigned int K = 100 ;
    _real old_ener ;
    double eps = param.epsilonTangentProjection ;
    
    //  cout << "projecting momentum on image tangent space" << endl ;
    //  I.getMomentum(ZI, Lw) ;
    //  kernel(Lw, w) ;
    double normZ = 1e-4 + sqrt(ZI.norm2()) ;
    
    b.copy(ZI) ;
    //  I.normal().norm(nI) ;
    //  b *= nI ;
    //  normal.scalProd(b, foo1) ;
    //  foo2.copy(normal) ;
    //  foo2 *= foo1 ;
    b /= normZ ;
    
    Z.al(ZI.d) ;
    Z.zero() ;
    //cout << "Z" << endl ;
    
    foo2.copy(I.normal()) ;
    foo2 *= Z ;
    //cout << "*Z" << endl ;
    kernel(foo2, foo2) ;
    //cout << "kernel" << endl ;
    I.normal().scalProd(foo2, AZ) ;
    foo1.copy(Z) ;
    foo1 *= eps ;
    AZ += foo1 ;
    //cout << "AZ" << endl ;
    
    foo1.copy(AZ) ;
    foo1 -= b ;
    old_ener = foo1.norm2() ;
    //cout << "ener" << endl; 

    r.copy(b) ;
    r -= AZ ;
    
    if (r.maxAbs() < 0.001){
      if (param.verb)
	cout << "CG: nothing to do" << endl ;
      return ;
    }
    
    //cout << "Starting iterations " << endl ;
    _real mu=0, muold, alfa, be, ener2  ;
    for (unsigned int i2 =0; i2<K; i2++) {
      if (i2 == 0) {
	mu = r.norm2() ;
	p.copy(r) ;
      }
      else {
	muold = mu ;
	mu = r.norm2() ;
	be = mu/muold ;
	//      foo.copy(p) ;
	p *= be ;
	p += r ;
      }
      
      foo2.copy(I.normal()) ;
      foo2 *= p ;
      kernel(foo2, foo2) ;
      I.normal().scalProd(foo2, q) ;
      foo1.copy(p) ;
      foo1 *= eps ;
      q += foo1 ;
      
      alfa = mu / p.sumProd(q) ;
      foo1.copy(p) ;
      foo1 *= alfa ;
      Z += foo1 ;
      foo1.copy(q) ;
      foo1 *= alfa ;
      r -= foo1 ;
      AZ += foo1 ;
      
      foo1.copy(AZ) ;
      foo1 /= 2 ;
      foo1 -= b ;
      //ener = Z.sumProd(foo1) ;
      
      foo1.copy(AZ) ;
      foo1 -= b ;
      ener2 = foo1.norm2() ;
      //  cout << "iter tangent CG " << i2 << " ener = " << ener << " " << ener2/old_ener << endl  ;
      if (ener2 < 1e-4 || ener2 < 0.0001 * old_ener) 
	{
	  if (param.verb)
	    cout << "iter tangent CG " << i2 << " ener = " << ener2 << " " << old_ener << endl  ;
	  Z *= normZ ;
	  return ;
	}
    if (param.verb)
      cout << "iter tangent CG " << i2 << " ener = " << ener2 << " " << old_ener << endl  ;
    }

    if (param.verb)
      cout << "iter tangent CG (max iter) " << K << " ener = " << ener2 << " " << old_ener << endl  ;
    Z *= normZ ;
  }



  /**
     Adjoint star action on momenta relative to the geodesic starting at Lv
  */
  void adjointStarTransport(const deformableImage &I0, const VectorMap &Lv, const VectorMap &Lw, deformableImage &I2, VectorMap &Lwc, Tangent &Z) {
    deformableImage It ;
    Tangent Z1 ;
    VectorMap wt, wc;
    It.copy(I0) ;
 
    // from template 1  to template 2
    geodesicImageEvolutionFromVelocity(I0, Lv, It, 1.0) ;
    // applying the adjoint to the second momentum
    big_adjointStar(Lw, _psi, Lwc) ;
    // getting new target
    // cout << "new target" << endl ;
    geodesicImageEvolutionFromVelocity(It, Lwc, I2, 1.0) ;
    kernel(Lwc, wc) ; 
    //  cout << "new momentum" <<endl ;
    It.computeGradient(param.spaceRes,param.gradientThreshold) ;
    // cout << "max momentum = " << Lwc.max() << " " << Lwc.scalProd(wc) << endl ; 
    It.infinitesimalAction(wc, Z1) ;
    // imageTangentProjection(It, wc, wt, Z) ;
    imageTangentProjection(It, Z1, Z) ;
  }

  
  /**
     Parallel transport in image space
  */

  void parallelImageTransportOLD(const deformableImage &I0, const Tangent &Z0, const Tangent &M0, 
				 deformableImage &I2, VectorMap &Lwc, Tangent &Zwc) { 
    deformableImage It ;
    VectorMap vt, wt, Lvt, Lwt, wc , Lvt2, vt2, Lv ;
    Tangent Zt, Zvt, Zwt, Zv2, Mt, dI, Mc ; 
    
    Zt.copy(Z0) ;
    It.copy(I0) ;
    It.computeGradient(param.spaceRes,param.gradientThreshold) ;
    It.getMomentum(Zt, Lvt) ;
    Lv.copy(Lvt) ;
    Mt.copy(M0) ;
    It.getMomentum(Mt, Lwt) ;
    kernel(Lvt, vt) ;
    kernel(Lwt, wt) ;
    if (param.verb) {
      cout << "norm momentum = " << Lwt.scalProd(wt) << endl ; 
      cout << "norm Lv = " << Lvt.scalProd(vt) << endl ; 
    }


    unsigned int N_k = param.parallelTimeDisc ;
    _real delta = 1.0/N_k ;


    for (unsigned int  k=0; k<N_k; k++) {
      //    cout << "T = " << k*delta << endl ;
      //    param.accuracy *= 10 ; 
      var_geodesicImageEvolution(It, Zt, Mt, dI, Zwc, delta);
      //    param.accuracy /= 10 ; 
      geodesicImageEvolutionFromVelocity(It, Lvt, I2, Lvt2, Zvt, delta) ;
      // geodesicImageEvolution(It, Zt, I2, Zvt, delta) ;
      I2.getMomentum(Zvt, Lvt2) ;
      kernel(Lvt2, vt2) ;
      //    cout << "norm test: " << Lvt2.scalProd(vt2) << endl ;
      //    cout << "test for norms " << kernelNorm(Lvt) << " " << kernelNorm(Lvt2) << endl ;
      It.copy(I2) ;
      It.computeGradient(param.spaceRes,param.gradientThreshold) ;
      Zt.copy(Zvt) ;
      //    cout << "Projection" << endl ;
      imageTangentProjection(It, dI, Mc) ;
      //    It.getMomentum(Zvt, Lvt2) ;
      //    cout << "test for norms " << kernelNorm(Lvt2) << endl ;
      //    It.getMomentum(Zwc, Lwc) ;
      //    I2.write_image("currentTemplate") ;
      Mc /= -delta ;
      It.getMomentum(Mc, Lwc) ;
      It.getMomentum(Zvt, Lvt2) ; 

      kernel(Lwc, wc) ;
      kernel (Lvt2, vt2) ;
      
      _real Mag_v = Lvt.scalProd(vt), Mag_w = Lwt.scalProd(wt) ;
      _real Coup_vw = Lwt.scalProd(vt) ;
      
      Lvt.copy(Lvt2) ;
      vt.copy(vt2) ;
      _real Coup_vwc =Lwc.scalProd(vt),   Mag_wc  = Lwc.scalProd(wc) ;
      _real alfa, be ;
      if (param.verb) {
	cout << "Mag_v = " << Mag_v << endl ; 
	cout << "Mag_wc = " << Mag_wc << " " << Mag_wc/Mag_w << endl ;
	cout << "Coup_vwc = " << Coup_vwc << " " << Coup_vwc/Coup_vw << endl ;
	cout << "Norm Lv: " << Lvt.scalProd(vt) << endl ; 
      }
      be = sqrt((Mag_v*Mag_w-Coup_vw*Coup_vw)/(Mag_v*Mag_wc-Coup_vwc* Coup_vwc));
      alfa = Coup_vw/Mag_v-(Coup_vwc/Mag_v) * be ;
      if (param.verb)
	cout << "Alpha[" << k << "] = " << alfa << " ; " << "Beta[" << k << "] = " << be << endl  ;



      Mt.copy(Mc) ;
      Mt *= be ;
      Zv2.copy(Zvt) ;
      Zv2 *= alfa ;
      Mt += Zv2 ;
      It.getMomentum(Mt, Lwt) ;
      kernel(Lwt, wt) ;
      if (param.verb)
	cout << "norm momentum = " << Lwt.scalProd(wt) << " " << Mt.maxAbs() <<  " " << M0.maxAbs() << endl << endl ; 
      //    geodesicImageEvolutionFromVelocity(It, Lwt, I2, Lvt2, 1.0) ;
      //    I2.write_image("currentTarget") ;
    }

    Zwc.copy(Mt) ;
    Lwc.copy(Lwt) ;
    //  imageTangentProjectionFP(It, wt, Lwc, Z) ;
    //    kernel(Lwc, wc) ;
    //    cout << "Compared norms; before projection: " << wt.scalProd(Lwt) << " after projection: " << wc.scalProd(Lwc) << endl ; 
    geodesicImageEvolutionFromVelocity(I0, Lv, It, 1.0) ;
    geodesicImageEvolutionFromVelocity(It, Lwc, I2, 1.0) ;
  }


  void parallelImageTransport(const deformableImage &I0, const Tangent &Z0, const Tangent &Z1, 
			      deformableImage &I2, VectorMap &Lwt, Tangent &Zt) {
    VectorMap vt, zz, wt, Lvt, foo, id, semi, psi ;
    Tangent Zvt, foo0, Ztmp ; 
    deformableImage It ;

    Zt.copy(Z1) ;
    I0.getMomentum(Z0, Lvt) ;
    I0.getMomentum(Z1, Lwt) ;
    kernel(Lvt, vt) ;
    kernel(Lwt, wt) ;
    It.copy(I0) ;
    It.computeGradient(param.spaceRes,param.gradientThreshold) ;
  
    _real nw = sqrt(Lwt.scalProd(wt)) + 1;
    nw = 1 ;
    Lwt /= nw ;
    wt /= nw ;

    id.idMesh(Lvt.d) ;
    
    int N_k ; 
    //= param.parallelTimeDisc ;
    //_real dt = 1.0/N_k ;
    _real dt, M = vt.maxNorm() ;
     N_k = 2*ceil(param.accuracy*M+1) ;  
     if (N_k > param.parallelTimeDisc)
       N_k = param.parallelTimeDisc ;
     dt = 1.0/N_k  ; 

    //    if (param.verb)
    //      cout << "ParallelTranslation " << T << endl ;

    _real Mag_v = Lvt.scalProd(vt), Mag_w = Lwt.scalProd(wt) ;
    _real Coup_vw = Lwt.scalProd(vt) ;

    if (param.verb)
      cout << "T = " << 0 << " " << Mag_v << " " << Mag_w << " " << Coup_vw << endl ;
    
    if (Mag_v < 1e-10)
      return ;
  
    for (int t=0; t<N_k; t++) {
      semi.copy(vt) ;
      semi *= dt ;
  
      if (t>0){
	foo.copy(id) ;
	foo -=  semi ;
	foo.multilinInterp(psi, psi) ;
      }
      else {
	psi.copy(id) ;
	psi -=  semi ;
      }

      psi.multilinInterp(I0.img(), It.img()) ;
      It.computeGradient(param.spaceRes,param.gradientThreshold) ;

      // PT increment
      adjointStar(wt, Lvt, zz) ;
      adjointStar(vt, Lwt, foo) ;
      zz += foo ;
      adjoint(vt, wt, foo) ;
      inverseKernel(foo, foo) ;
      zz -= foo ;
      zz *= dt/2 ;
      Lwt -= zz ;
      kernel(Lwt, wt) ;
      // Geodesic increment
      adjointStar(vt, Lvt, zz) ;
      zz *= dt ;
      Lvt -= zz ;
      kernel(Lvt, vt) ;
      vt.scalProd(It.normal(), Ztmp) ;
      Ztmp *= -1 ;
      // It.infinitesimalAction(vt, Ztmp) ;
      imageTangentProjection(It, Ztmp, Zvt) ;
      It.getMomentum(Zvt, Lvt) ;
      kernel(Lvt, vt) ;
    
      // Norm correction for geodesic 
      _real   Mag_vt  = Lvt.scalProd(vt) ;
      Lvt *= sqrt(Mag_v/Mag_vt) ;
      vt *= sqrt(Mag_v/Mag_vt) ;
      
      // New momentum
      kernel(Lwt, wt) ;
      wt.scalProd(It.normal(), Ztmp) ;
      Ztmp *= -1 ;
      //      It.infinitesimalAction(wt, Ztmp) ;
      imageTangentProjection(It, Ztmp, Zt) ;
      It.getMomentum(Zt, Lwt) ;
      kernel(Lwt, wt) ;

      // Norm and inner product correction
      _real Coup_vwt = Lwt.scalProd(vt) ;
      _real   Mag_wt  = Lwt.scalProd(wt) ;
    
      _real alfa, be ;
      if (param.verb) {
	cout << "T = " << (t+1)*dt << " ; Mag_v = " << Mag_vt << endl ; 
	cout << "Mag_wc = " << Mag_wt << " " << Mag_wt/Mag_w << endl ;
	cout << "Coup_vwc = " << Coup_vwt << " " << Coup_vwt/Coup_vw << endl ;
      }
      be = sqrt((Mag_v*Mag_w-Coup_vw*Coup_vw)/(Mag_v*Mag_wt-Coup_vwt* Coup_vwt));
      alfa = Coup_vw/Mag_v-(Coup_vwt/Mag_v) * be ;
      if (param.verb)
	cout << "Alpha[" << t << "] = " << alfa << " ; " << "Beta[" << t << "] = " << be << endl ;
      
    
      Lwt *= be ;
      foo.copy(Lvt) ;
      foo *= alfa ;
      Lwt += foo ;
      kernel(Lwt, wt) ;

      Zt *= be ;
      foo0.copy(Zvt) ;
      foo0 *= alfa ;
      Zt += foo0 ;
    }
    Lwt *= nw ;
    Zt *= nw ;
    geodesicImageEvolutionFromVelocity(It, Lwt, I2, 1) ;
  }

  /* void parallelImageTransportFromVelocity(const deformableImage &I0, const Tangent &Z0, const Tangent &Z1, deformableImage &I2, VectorMap &Lwc, Tangent &Zwc) {  */
  /*     deformableImage It ; */
  /*     VectorMap vt, wt, Lvt, Lwt, wc , Lvt2, vt2, v, w, Lv, Lw ; */
  /*     Vector jac, jact ; */
  /*     Tangent Zvt, Zwt, Zv2, dI ;  */

  /*     Zvt.copy(Z0) ; */
  /*     Zwt.copy(Z1) ; */
  /*     I0.getMomentum(Z0, Lv) ; */
  /*     I0.getMomentum(Z1, Lw) ; */
  /*     kernel(Lv, v) ; */
  /*     kernel(Lw, w) ; */
    
  /*     vt.copy(v) ; */
  /*     wt.copy(w) ; */
  /*     Lvt.copy(Lv) ; */
  /*     Lwt.copy(Lw) ;  */
  /*     if (param.verb) */
  /* 	cout << "norm momentum = " << Lwt.scalProd(wt) << endl ;  */
    

  /*     unsigned int N_k = param.parallelTimeDisc ; */
  /*     _real delta = 1.0/N_k ; */
    
  /*     It.copy(I0) ; */
  /*     It.computeGradient(param.spaceRes,param.gradientThreshold) ; */
  /*     VectorMap psit ; */

  /*     psit.idMesh(I0.img().d) ; */
  /*     jact.al(vt.d) ; */
  /*     jact = 1 ; */

  /*     for (unsigned int  k=0; k<N_k; k++) { */
  /* 	if (param.verb) */
  /* 	  cout << "T = " << k*delta << endl ; */

  /* 	var_geodesicImageEvolutionFromVelocity(It, Lvt, Lwt, Lwc, Zwc, delta); */
  /* 	//var_geodesicImageEvolution(It, Zvt, Zwt, dI, Zwc, delta); */
      
  /* 	// GeodesicDiffeoEvolution(Lvt, Lvt2,  delta) ; */
  /* 	geodesicImageEvolutionFromVelocity(It, Lvt, I2, Lvt2, Zvt, delta) ; */

  /* 	_psi.multilinInterp(psit, psit) ; */
  /* 	psit.multilinInterp(I0.img(), It.img()) ; */
  /* 	It.computeGradient(param.spaceRes,param.gradientThreshold) ; */

  /* 	Zwc /= delta ; */
  /* 	Lwc /= delta ; */
  /* 	kernel(Lwc, wc) ; */
  /* 	kernel (Lvt2, vt2) ; */
      
  /* 	_real Mag_v = Lvt.scalProd(vt), Mag_w = Lwt.scalProd(wt) ; */
  /* 	_real Coup_vw = Lwt.scalProd(vt) ; */

  /* 	Lvt.copy(Lvt2) ; */
  /* 	vt.copy(vt2) ; */
  /* 	_real Coup_vwc =Lwc.scalProd(vt) ; */
  /* 	_real   Mag_wc  = Lwc.scalProd(wc) ; */
  /* 	_real alfa, be ; */
  /* 	if (param.verb) { */
  /* 	  cout << "Mag_v = " << Mag_v << endl ;  */
  /* 	  cout << "Mag_wc = " << Mag_wc << " " << Mag_wc/Mag_w << endl ; */
  /* 	  cout << "Coup_vwc = " << Coup_vwc << " " << Coup_vwc/Coup_vw << endl ; */
  /* 	  cout << "Norm Lv: " << Lvt.scalProd(vt) << endl ;  */
  /* 	} */
  /* 	be = sqrt((Mag_v*Mag_w-Coup_vw*Coup_vw)/(Mag_v*Mag_wc-Coup_vwc* Coup_vwc)); */
  /* 	alfa = Coup_vw/Mag_v-(Coup_vwc/Mag_v) * be ; */
  /* 	if (param.verb) */
  /* 	  cout << "Alpha[" << k << "] = " << alfa << " ; " << "Beta[" << k << "] = " << be << endl ; */
      
    
  /* 	Lwt.copy(Lwc) ; */
  /* 	Lwt *= be ; */
  /* 	Lvt2.copy(Lvt) ; */
  /* 	Lvt2 *= alfa ; */
  /* 	Lwt += Lvt2 ; */
  /* 	kernel(Lwt, wt) ; */

    
  /* 	Zwt.copy(Zwc) ; */
  /* 	Zwt *= be ; */
  /* 	Zv2.copy(Zvt) ; */
  /* 	Zv2 *= alfa ; */
  /* 	Zwt += Zv2 ; */
  /* 	if (param.verb) */
  /* 	  cout << "norm momentum = " << Lwt.scalProd(wt) << endl ;  */
  /* 	//    I2.write_image("currentTarget") ; */
  /*     } */

  /*     char path[256] ; */
  /*     sprintf(path, "%s/PTTarget", param.outDir) ; */
  /*     It.img().write_image(path) ; */
  /*     Zwc.copy(Zwt) ; */
  /*     Lwc.copy(Lwt) ; */
  /*     //  imageTangentProjectionFP(It, wt, Lwc, Z) ; */
  /*     //    kernel(Lwc, wc) ; */
  /*     //    cout << "Compared norms; before projection: " << wt.scalProd(Lwt) << " after projection: " << wc.scalProd(Lwc) << endl ;  */
  /*     //  geodesicImageEvolutionFromVelocity(I0, Lv, It, 1.0) ; */
  /*     geodesicImageEvolutionFromVelocity(It, Lwc, I2, 1.0) ; */
  /*   } */


    /**
       Geodesics in image space
    */
    void geodesicImageEvolutionFromVelocity(const deformableImage &I0, const VectorMap &Lv, deformableImage &I1, 
					    VectorMap &Lv1, Tangent &Z, _real delta) {
      VectorMap Lv2, v2 ;
      Tangent Z2 ;
      _real norm1, norm2 ;
    
      GeodesicDiffeoEvolution(Lv, Lv2,  delta) ;
      _psi.multilinInterp(I0.img(), I1.img()) ;
      I1.computeGradient(param.spaceRes,param.gradientThreshold) ;
      kernel(Lv2, v2) ;
      //  imageTangentProjection(I1, Lv2, Lv1) ;
      I1.normal().scalProd(v2, Z2) ;
      Z2 *=-1 ;
      //  inverseKernel(alpha, Lalpha) ;
      imageTangentProjection(I1, Z2, Z) ;
      I1.getMomentum(Z, Lv1) ;
    
      norm1 = sqrt(kernelNorm(Lv)) ;
      norm2 = sqrt(kernelNorm(Lv1)) ;
      Lv1 *= norm1/norm2 ;
      Z *= norm1/norm2 ;
      //  normalComponent(I1, Lv1, Z) ;
    }

    void geodesicImageEvolutionFromVelocity(const deformableImage &I0, const VectorMap &Lv, deformableImage &I1, _real delta) {
      VectorMap Lv1 ;
      GeodesicDiffeoEvolution(Lv, Lv1,  delta) ;
      _psi.multilinInterp(I0.img(), I1.img()) ;
    }
  

    /**
       Jacobi fields in image space
    */
    /* void var_geodesicImageEvolutionFromVelocity(const deformableImage& I0, const VectorMap &Lv, const VectorMap &Lw,  */
    /* 						VectorMap &Lwc, Tangent &Z,  _real delta) { */
    /*   deformableImage I ; */
    /*   Vector Lalpha ; */
    
    /*   VarGeodesicDiffeoEvolution(Lv, Lw, delta) ; */
    /*   _psi.multilinInterp(I0.img(), I.img()) ; */
    /*   I.computeGradient(param.spaceRes,param.gradientThreshold) ; */
    
    /*   I.normal().scalProd(alpha, Lalpha) ; */
    /*   Lalpha *=-1 ; */
    /*   //  inverseKernel(alpha, Lalpha) ; */
    /*   imageTangentProjection(I, Lalpha, Z) ; */
    /*   I.getMomentum(Z, Lwc) ; */
    /*   //  normalComponent(I, Lwc, Z) ; */
    /* } */

    void var_geodesicImageEvolution(const deformableImage& I0, const Tangent &Z0, const Tangent &dZ0, 
				    Tangent &dI, Tangent &dZ, _real delta) {
      deformableImage I ;
      Tangent It, dIt, dZt, divv, fooI, fooI2, Zt ;
      VectorMap dv,gradI, graddI, id, foo1, foo2 ;
    
      VectorMap Lv, Lvt, vt ;
      I0.getMomentum(Z0, Lv) ;
      Lvt.copy(Lv) ;
      kernel(Lvt, vt) ;
      double correct, initNorm = Lvt.scalProd(vt) ;

      _real T, dt, M = delta*vt.maxNorm() ;
      T = ceil(param.accuracy*M+1) ;  
      if (T > param.Tmax)
	T = param.Tmax ;
      dt = delta/T  ; 
    
      dIt.zeros(vt.d) ;
      id.idMesh(vt.d) ;
      dZt.copy(dZ0) ;
      Zt.copy(Z0) ;
      It.copy(I0.img()) ;

      for (unsigned int t=0; t<T; t++) {
	gradient(It, gradI, param.spaceRes) ;
	gradient(dIt, graddI, param.spaceRes) ;
	foo1.copy(gradI) ;
	foo1 *= dZt ;
	foo2.copy(graddI) ;
	foo2 *= Zt ;
	foo2 += foo1 ;
	kernel(foo2, dv) ;
	dv *= -1 ;
      
	foo1.copy(vt) ;
	foo1 *= -dt ;
	foo1 += id ;


	dv.scalProd(gradI, fooI) ;
	fooI *= -dt ;
	fooI += dIt ;
	foo1.multilinInterp(fooI, dIt) ;
      

	divergence(vt, divv, param.spaceRes) ;
	fooI.copy(divv) ;
	fooI *= dZt ;
	foo2.copy(dv) ;
	foo2 *= Zt ;
	divergence(foo2, fooI2, param.spaceRes) ;
	fooI += fooI2 ;
	fooI *= -dt ;
	fooI += dZt ;
	foo1.multilinInterp(fooI, dZt) ;

	fooI.copy(divv) ;
	fooI *= Zt ;
	fooI *= -dt ;
	fooI += Zt ;
	foo1.multilinInterp(fooI, Zt) ;
	//    Zt += fooI ;

	foo1.multilinInterp(It, It) ;
      
	I.img().copy(It) ;
	I.computeGradient(param.spaceRes,param.gradientThreshold) ;
	I.getMomentum(Zt, Lvt) ;
	kernel(Lvt, vt) ;
	correct = sqrt(initNorm/Lvt.scalProd(vt)) ;
	Lvt *= correct ;
	vt *= correct;
	Zt *= correct ; 
      }

      dI.copy(dIt) ;
      dZ.copy(dZt) ;
    }  


    void geodesicImageEvolution(const deformableImage& I0, const Tangent &Z0,  deformableImage &I, Tangent &Z, _real delta) {
      double correct ;
      Tangent It, divv, fooI, Zt, jac ;
      VectorMap dv, gradI, id, foo1, foo2, psi ;
    
      VectorMap Lv, Lvt, vt ;
      I0.getMomentum(Z0, Lv) ;
      Lvt.copy(Lv) ;
      kernel(Lvt, vt) ;
      double initNorm = Lvt.scalProd(vt) ;
    
      _real T, dt, M = delta*vt.maxNorm() ;
      T = ceil(param.accuracy*M+1) ;  
      if (T > param.Tmax)
	T = param.Tmax ;
      dt = delta/T  ; 

      id.idMesh(vt.d) ;
      Zt.copy(Z0) ;
      It.copy(I0.img()) ;
      psi.copy(id) ;
      jac.al(vt.d) ;
      jac = 1 ;

      for (unsigned int t=0; t<T; t++) {
	gradient(It, gradI, param.spaceRes) ;

	foo1.copy(vt) ;
	foo1 *= -dt ;
	foo1 += id ;
      
	foo1.multilinInterp(psi, psi) ;
      
	divergence(vt, divv, param.spaceRes) ;
	fooI.copy(divv) ;
	fooI *= jac ;
	fooI *= -dt ;
	fooI += jac ;
	foo1.multilinInterp(fooI, jac) ;
	//    Zt += fooI ;


	psi.multilinInterp(I0.img(), It) ;
	psi.multilinInterp(Z0, Zt) ;
	Zt *= jac ;
      
	I.img().copy(It) ;
	I.computeGradient(param.spaceRes,param.gradientThreshold) ;
	I.getMomentum(Zt, Lvt) ;
	kernel(Lvt, vt) ;
	correct = sqrt(initNorm/Lvt.scalProd(vt)) ;
	Lvt *= correct ;
	vt *= correct;
	Zt *= correct ; 
	//    cout << "norms: " << Lvt.scalProd(vt) << " " << initNorm << endl ;
      }

      Z.copy(Zt) ;
      I.getMomentum(Z, Lvt) ;
      kernel(Lvt, vt) ;
      //    cout << "norms: " << Lvt.scalProd(vt) << " " << initNorm << endl ;
      //  Z*= -1 ;
    }  

    //  void imageTangentProjectionFP(const deformableImage &I, const VectorMap &m0, VectorMap &Lw) {
    //  Tangent foo; imageTangentProjectionFP(I, m0, Lw, foo);}


    ~ImageEvolution(){}
  };


#endif
