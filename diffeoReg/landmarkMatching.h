/**
   landmarkMatching.h
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

#ifndef _LMATCHING_
#define _LMATCHING_
#include "pointSetEvolution.h"
#include "optimF.h"
#include "PointSetMatchingAffine.h"

/** 
    Class implementing LDDMM landmark with respect to the initial momentum
*/

template <class KERNEL>
class scalProdLandmarks: public PointSetEvolution<KERNEL> {
 public:
  scalProdLandmarks(){};
  scalProdLandmarks(PointSet &x){_p.copy(x);}
  void init(PointSet &x){_p.copy(x);}
  double operator()(const Tangent &a, const Tangent &aa) const{
    Tangent Kaa ;
    PointSetEvolution<KERNEL>::kernel(_p, aa, Kaa) ;
    double res = 0 ; 
    for(unsigned int k=0; k< a.size(); k++) 
      for(int kk=0; kk<a.dim();kk++) 
	res+= a[k][kk]*Kaa[k][kk];  
    return res;
  }
 private:
  PointSet _p ;
};


class LandmarkObjective
{
 public:
  PointSet target ;
  double energy(const PointSet& p0) 
  {
    double res =0, u ;
    //  cout << "energy: " << p0.size() << " " << p0.dim() << endl;
    for(unsigned int k=0; k<p0.size(); k++)
      for(int kk=0; kk<p0.dim(); kk++) {
	//      cout << p0[k][kk] << " " << p1[k][kk] << endl ;
	u = (p0[k][kk]-target[k][kk]) ;
	res +=  u*u ;
      }
    return res ;
  }
  void gradient(const PointSet& p0, Tangent &b) 
  {
    b.al(p0.size(), p0.dim()) ;
    for (unsigned int k=0; k<p0.size(); k++)
      for(int kk=0; kk<p0.dim(); kk++)
	b[k][kk] = (p0[k][kk] - target[k][kk])/p0.size() ;
  }
} ;

//double LDDMMLandmarkEnergy(const PointSet& p0, const PointSet& p1) ;
//void LDDMMLandmarkGradient(const PointSet& p0, const PointSet & p1, Tangent &b) ;
template <class KERNEL, class OBJECTIVE>
class LandmarkMatchingBase: public PointSetEvolution<KERNEL>
{
public:
  using PointSetEvolution<KERNEL>::meta ;
  using PointSetEvolution<KERNEL>::Template ;
  using PointSetEvolution<KERNEL>::Target ;
  using PointSetEvolution<KERNEL>::param ;
  //using Diffeomorphisms::param;
  using PointSetEvolution<KERNEL>::geodesicPointSetEvolution ;
  using PointSetEvolution<KERNEL>::gamma ;
  using PointSetMatching::affTrans ;
  using PointSetMatching::init ;
  LandmarkMatchingBase(param_matching &par) {PointSetMatching::init(par); initObjective();}
  LandmarkMatchingBase(char *file, int argc, char** argv) {PointSetMatching::init(file, argc, argv); initObjective();}
  LandmarkMatchingBase(char *file, int k) {PointSetMatching::init(file, k);initObjective();}

  OBJECTIVE objective ;
  Tangent a0;
  LandmarkMatchingBase(){} ;

  virtual void initObjective() { cout << "initObjective is not available in this function" << endl ; exit(1) ;}

  virtual void affineReg(){
    PointSetAffineEnergy enerAff ;
    //    Template.domain().putm(enerAff.MIN) ;
    PointSetMatching::initializeAffineReg(enerAff) ;
    PointSetMatching::affineReg(enerAff) ;
    PointSetMatching::finalizeAffineTransform(gamma) ;    
  }

  /*  void affineReg(){
    PointSetAffineEnergy enerAff ;
    Matching::AffineReg(enerAff) ;
    } */

  class optimLandmark: public optimFunBase<Tangent, Tangent>{
  public:
    bool meta ;
    virtual ~optimLandmark(){};
    optimLandmark(LandmarkMatchingBase *lm) {
      _lm = lm ;
      meta = lm->meta ;
      if (meta==false)
	cout << "Not metamorphosis" << endl;
      sigma = lm->param.sigma ;
      scp.init(lm->Template) ;
    }
    double computeGrad(Tangent &a, Tangent& grada){
      Tangent b, p1,dp, ga, a1 ;
      _lm->geodesicPointSetEvolution(_lm->Template, a, p1, a1, 1) ;
      _lm->objective.gradient(p1, b) ;
      //      LDDMMLandmarkGradient(p1, _lm->Target, b) ;
      if (meta==false)
	b /= sigma*sigma ;
      _lm->dualVarGeodesicPointSetEvolution(_lm->Template, a, b, dp, ga, 1) ;
      
      _lm->inverseKernel(_lm->Template, ga, grada) ;
      if (meta==false)
	for(unsigned int k = 0; k<a.size(); k++)
	  for(int kk=0; kk<a.dim(); kk++) {
	    grada[k][kk] += a[k][kk] ;
	  }
      return scp(grada, grada) ;
    }

    double objectiveFun(Tangent &a) {
      PointSet p1 ;
      Tangent a1 ;
      _lm->geodesicPointSetEvolution(_lm->Template, a, p1, a1, 1) ;
      double ener = _lm->objective.energy(p1) ;
      //      double ener = LDDMMLandmarkEnergy(p1, _lm->Target) ;
      if (meta==false) {
	cout << "not meta" << endl; 
	ener = scp(a, a) + ener / (sigma*sigma) ;
      }

      //      cout << "ener = " << ener << endl ;
      return ener ;
    }
    
    double endOfIteration(Tangent &a) {
      _lm->a0.copy(a) ;
      _lm->Print() ;
      return objectiveFun(a) ;
    }
    
    void endOfProcedure(Tangent &a) {
      //      cout << "end of procedure" << endl; 
      _lm->a0.copy(a) ;
      PointSet p1;
      Tangent a1 ;
      _lm->geodesicPointSetEvolution(_lm->Template, _lm->a0, p1, a1, 1) ;
      //      cout << p1 << endl ;
      _lm->Print() ;
    }
    scalProdLandmarks<KERNEL> scp ;
  private:
    LandmarkMatchingBase *_lm ;
    double sigma ;
  };

  void gradientLandmarkMatching() {a0.zeros(Template.size(), Template.dim()); gradientLandmarkMatching(a0);}
  void gradientLandmarkMatching(Tangent & aa) {
    Tangent a ;

    conjGrad<Tangent, Tangent, scalProdLandmarks<KERNEL>, optimLandmark> cg ;
    
    optimLandmark opt(this) ; //opt( Template,  Target,  param.sigma) ;
    scalProdLandmarks<KERNEL> scp(Template) ;
    cg(opt, scp, aa, a, param.nb_iter, 0.001, param.epsMax, param.minVarEn, param.gs, param.verb) ;
  }

  virtual ~LandmarkMatchingBase(){}

  virtual void initialPrint() {} ;
  virtual void Print() {} ; 
};

template <class KERNEL>
class LandmarkMatching: public LandmarkMatchingBase<KERNEL, LandmarkObjective>
{
public:
  using PointSetEvolution<KERNEL>::Template ;
  using PointSetEvolution<KERNEL>::Target ;
  using PointSetEvolution<KERNEL>::geodesicPointSetEvolution ;
  //using Diffeomorphisms::param;
  using LandmarkMatchingBase<KERNEL, LandmarkObjective>::param ;
  //  using PointSetEvolution<KERNEL>::param ;
  using LandmarkMatchingBase<KERNEL, LandmarkObjective>:: objective ;
  using LandmarkMatchingBase<KERNEL, LandmarkObjective>:: a0 ;
  LandmarkMatching(param_matching &par){PointSetMatching::init(par); initObjective();}
  LandmarkMatching(char *file, int argc, char** argv){PointSetMatching::init(file, argc, argv); initObjective();}
  LandmarkMatching(char *file, int k){PointSetMatching::init(file, k);initObjective();}
  void initObjective(){
    objective.target.copy(Target) ;
  }

  void saveTrajectories(char * filename, int size, int dim, DiffEqOutput &out) {
    ofstream ofs(filename) ;
    if (ofs.fail()) {
      cerr << "unable to open " << filename << endl; 
      exit(1) ;
    }

    int j ;
    ofs << out.counts() << " " << size << " " << dim << endl ;
    for (int i=0; i<=out.counts(); i++) {
      ofs << out.getTime(i) << endl; 
      j = 0 ;
      for(int k=0; k<size; k++) {
	for(int kk=0; kk<dim; kk++)
	  ofs << out.getState(i, j++) << " " ;
	ofs << endl ;
      }
      ofs << endl ;
      for(int k=0; k<size; k++) {
	for(int kk=0; kk<dim; kk++)
	  ofs << out.getState(i, j++) << " " ;
	ofs << endl ;
      }
      ofs << endl ;
    }
  }

  void initialPrint() {  initialPrint(param.outDir) ;}
  void Print() {  Print(param.outDir) ;}

  void Print(char* path)
    {
      if (!param.printFiles)
	return ;
      char file[256] ;
      PointSet p1 ;
      Tangent a1  ;
      DiffEqOutput out ;
      out.setCounts(30) ;
      //      cout << "Print" << endl ;
      
      geodesicPointSetEvolution(Template, a0, p1, a1, 1, out) ;
      //      cout << p1 << endl ;
      
      sprintf(file, "%s/deformedTemplate.lmk", path) ;
      if (param.verb)
	cout << "writing " << file << endl ;
      p1.write_points(file) ;
      
      sprintf(file, "%s/initialMomentum.lmk", path) ;
      if (param.verb)
	cout << "writing " << file << endl ;
      a0.write_points(file) ;

      sprintf(file, "%s/trajectories.lmk", path) ;
      if (param.verb)
	cout << "writing " << file << endl ;
      saveTrajectories(file, Template.size(), Template.dim(), out) ;
    }

  void initialPrint(char* path)
    {
      if (!param.printFiles)
	return ;
      char file[256] ;
      
      sprintf(file, "%s/template.lmk", path) ;
      if (param.verb)
	cout << "writing " << file << endl ;
      Template.write_points(file) ;
      sprintf(file, "%s/binaryTemplate", path) ;
      if (param.verb)
	cout << "writing " << file << endl ;
      Template.write(file) ;
      sprintf(file, "%s/target.lmk", path) ;
      if (param.verb)
	cout << "writing " << file << endl ;
      Target.write_points(file) ;
      sprintf(file, "%s/binaryTarget", path) ;
      if (param.verb)
	cout << "writing " << file << endl ;
      Target.write(file) ;
    }
} ;



#endif
