/**
   matchingBase.h
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

#ifndef _MATCHINGBASE_
#define _MATCHINGBASE_
#include "diffeomorphism.h"
#include "param_matching.h"
#ifdef MEM
extern mem ;
#endif


/**
   Base class for matching.
   Defines input and output, kernels and basic functions
*/

template < class OBJECT> class MatchingBase : public Diffeomorphisms
{
public:
  OBJECT Template, Target, input1, input2, savedTemplate, savedTarget;
  //  AE enerAff ;
  Matrix gamma, affTrans ;
  param_matching savedParam ;

  MatchingBase(){} ;
  MatchingBase(param_matching &par){ param.copy(par); Load();} ;
  MatchingBase(char *file, int argc, char** argv){
    param.read(file) ;
    param.read(argc, argv) ;
    Load() ;
  }
  MatchingBase(char *file){
    param.read(file) ;
    Load() ;
  }
  MatchingBase(char *file, int k){
    param.read(file) ;
    if (k==0)
      Load() ;
    else
      LoadTemplate() ;
  }
  virtual ~MatchingBase(){};

  _real my_exp(_real a) { if (a>100) return exp(100); else if (a<-100) return exp(-100); else return exp(a);}

  virtual void Load() {
    OBJECT tmp, tmpbis, tmp1, tmp2 ;
    if (param.keepTarget) {
      //swap the target and template files
      swapFiles() ;
    }
    
    if (param.verb > 1)
      cout << "Reading Template from " << param.fileTemp << endl ;
    if (!param.doNotModifyTemplate && !param.doNotModifyImages){
      if (param.readBinaryTemplate)
	get_binaryTemplate(tmp1) ;
      else
	get_template(tmp1) ;

      if (param.revertIntensities){
	revertIntensities(tmp1) ;
      }

      if (param.crop1) 
	crop1(tmp1, tmp) ;
      else
	copy(tmp1, tmp) ;

      if (param.expand_margin.size() > 0)
	expandBoundary(tmp) ;

      rescaleDim(tmp, tmp1) ;    
      if (param.binarize) 
	binarize(tmp1) ;
    }
    else {
      if (param.readBinaryTemplate)
	get_binaryTemplate(tmp1) ;
      else
	get_template(tmp1) ;
    }

    if (param.foundTarget) {
      if (!param.doNotModifyImages) {
	if (param.verb > 1)
	  cout << "Reading Target from " << param.fileTarg << endl ;
	get_target(tmp2) ;
	if (param.revertIntensities){
	  revertIntensities(tmp2) ;
	}

	if (param.crop2) 
	  crop2(tmp2, tmp) ;
	else
	  copy(tmp2, tmp) ;
	if (param.expand_margin.size() > 0) {
	  if (param.verb > 1)
	    cout << "Expanding Target boundaries" << endl ;
	  expandBoundary(tmp) ;
	}

	if (param.binarize) 
	  binarize(tmp) ;

	if (!param.expandToMaxSize) {
	  rescaleTarget(tmp, tmp2) ;
	}
	else {
	  rescaleDim(tmp, tmp2) ;    
	  expandToMaxSize(tmp1, tmp2) ;
	}
	  
	if (param.affine) {
	  copy(tmp2, tmp) ;
	  affineInterpolation(tmp, tmp2) ;
	}

	if (param.flipTarget)
	  flip(tmp2, param.flipDim) ;
      }
      else {
	if (param.verb > 1)
	  cout << "Reading Target from " << param.fileTarg << endl ;
	get_target(tmp2) ;
      }
    }

    gamma.zeros(param.dim.size()+1, param.dim.size() + 1) ;
    
    copy(tmp1, input1) ;
    if (param.foundTarget)
      copy(tmp2, input2) ;

    if (param.doDefor) {
      // setKernel() ;
      Domain D(imageDim) ;
      makeMask(1, D) ;
      //if (!param.keepFFTPlans)
      //init_fft() ;
      if (!param.doNotModifyTemplate && !param.doNotModifyImages)
	convImage(tmp1, Template) ;
      else
	copy(tmp1, Template) ;
      if (param.foundTarget) {
	if (!param.doNotModifyImages)
	  convImage(tmp2, Target) ;
	else
	  copy(tmp2, Target) ;
	getAuxiliaryParameters() ;
	loadAuxiliaryFiles() ;
	affineReg() ;  
      }
    }
    else {
      copy(tmp1,Template) ;
      if (param.foundTarget)
	copy(tmp2, Target) ;
      getAuxiliaryParameters() ;
      loadAuxiliaryFiles() ;
    }
    

    if (param.scaleScalars) {
      scaleScalars() ;
    }

    if (param.keepTarget) {
      OBJECT foo; 
      copy(Template, foo) ;
      copy(Target, Template) ;
      copy(foo, Target) ;
    }
    copy(Template, savedTemplate) ;
    if (param.foundTarget)
      copy(Target, savedTarget) ;
    savedParam.copy(param) ;

    if (param.verb > 1)
      cout << "end init" << endl ;
  }


  virtual void LoadTemplate() 
  {
    OBJECT tmp, tmpbis, tmp1, tmp2 ;
    if (!param.doNotModifyTemplate && !param.doNotModifyImages){
      if (param.readBinaryTemplate)
	get_binaryTemplate(tmp1) ;
      else
	get_template(tmp1) ;

      if (param.revertIntensities){
	revertIntensities(tmp1) ;
      }

      if (param.crop1) 
	crop1(tmp1, tmp) ;
      else
	copy(tmp1, tmp) ;

      if (param.expand_margin.size() > 0)
	expandBoundary(tmp) ;

      rescaleDim(tmp, tmp1) ;    
      if (param.binarize) 
	binarize(tmp1) ;
    }
    else {
      if (param.readBinaryTemplate)
	get_binaryTemplate(tmp1) ;
      else
	get_template(tmp1) ;
    }

    Domain D(imageDim) ;
    //setKernel() ;
    makeMask(1, D) ;

    copy(tmp1,  input1) ;
    //if (!param.keepFFTPlans)
    // init_fft() ;
    
    convImage(tmp1, Template) ;
    
    if (param.verb > 1)
      cout << "end init" << endl ;
  }
  
  
  // Virtual components
  //  virtual void init_fft() {cout << "Warning init_fft: component is not implemented" << endl ;}
  virtual void affineReg() {cout << "Warning affineReg: component is not implemented" << endl ;}
  virtual void convImage(const OBJECT& in, OBJECT& out)  {copy(in, out);}
  virtual void convImageNofft(const OBJECT &in, OBJECT &out) {copy(in, out) ;}
  virtual void swapFiles() {
    char foo[256] ;
    strcpy(param.fileTemp, foo) ;
    strcpy(param.fileTarg, param.fileTemp) ;
    strcpy(foo, param.fileTarg);
  }
  virtual void get_template(OBJECT &obj) {cout << "get_template: component is not implemented" << endl ; exit(1) ;}
  virtual void get_binaryTemplate(OBJECT &obj) {cout << "get_binaryTemplate: component is not implemented" << endl ; exit(1) ;}
  virtual void get_target(OBJECT &obj) {cout << "get_target: component is not implemented" << endl ; exit(1) ;}
  virtual void revertIntensities(OBJECT& obj) { ;} 
  virtual void getAuxiliaryParameters(){} ;
  virtual void loadAuxiliaryFiles(){} ;
  virtual void crop1(const OBJECT &src, OBJECT& dest){ copy(src, dest);}
  virtual void crop2(const OBJECT &src, OBJECT& dest){ copy(src, dest);}
  virtual void copy(const OBJECT &src, OBJECT& dest){ dest.copy(src);}
  virtual void expandBoundary(OBJECT &img){;}
  virtual void rescaleDim(const OBJECT &img, OBJECT &res){copy(img, res) ;}
  virtual void rescaleTarget(const OBJECT &img, OBJECT& dest){copy(img,dest) ;}
  virtual void expandToMaxSize(OBJECT &tmp1, OBJECT &tmp2) {copy(tmp1,tmp2);}
  virtual void scaleScalars(){ ;}
  virtual void binarize(OBJECT &tmp1){;}
  virtual void flip(OBJECT &tmp1, int dim){;}
  virtual void affineInterpolation(OBJECT &img1, OBJECT &img2) {copy(img1, img2);}
};


#endif
