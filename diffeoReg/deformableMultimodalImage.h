/**
   deformableMultimodaImage.h
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
#ifndef _DEFMIMAGE_
#define _DEFMIMAGE_

#include "deformableObject.h"

/**
Class for the action of diffeomorphisms on vectors of images
*/
class deformableMultimodalImage: public std::vector<deformableImage> {
public:
  void computeGradient(vector<_real> &resol, double gradientThreshold){ 
    for (unsigned int k=1; k<size(); k++)
      (*this)[k].computeGradient(resol, gradientThreshold) ;
  }
  VectorMap & gradient(unsigned int k) { return (*this)[k].gradient() ;}
  VectorMap gradient(unsigned int k) const { return (*this)[k].gradient() ;}

  VectorMap & normal(unsigned int k) { return (*this)[k].normal() ;}
  VectorMap normal(unsigned int k) const { return (*this)[k].normal() ;}

  Vector & img(unsigned int k) {return (*this)[k].img();}
  Vector img(unsigned int k) const {return (*this)[k].img();}

  // needed members
  void get_template(const param_matching &param) 
  {
    for (unsigned int k=0; k<size(); k++)
      (*this)[k].get_image(param.fileTempList[k], param.dim.size());
  }
  void get_target(const param_matching &param)
  {
    for (unsigned int k=0; k<size(); k++)
      (*this)[k].get_image(param.fileTargList[k], param.dim.size());
  }
  void get_binaryTemplate(const param_matching &param) 
  {
    for (unsigned int k=0; k<size(); k++)
      (*this)[k].img().read(param.fileTempList[k].c_str());
  }
  void get_binaryTarget(const param_matching &param)
  {
    for (unsigned int k=0; k<size(); k++)
      (*this)[k].img().read(param.fileTargList[k].c_str());
  }
  void crop (const Domain &D, deformableMultimodalImage &source) const {
    for (unsigned int k=0; k<size(); k++) (*this)[k].img().crop(D, source[k].img());
}
  deformableMultimodalImage& copy (const deformableMultimodalImage &source) {
    resize(source.size()) ;
    for(unsigned int k=0; k<size(); k++)
      (*this)[k].img().copy(source[k].img()); 
    return *this;
  }
  void rescale(const Domain &D, deformableMultimodalImage &res) const {
    res.resize(size()) ;
    for(unsigned int k=0; k<size(); k++)
      (*this)[k].img().rescale(D, res[k].img()) ; 
  }
  Domain & domain(){ return (*this)[0].img().d;}
  Domain  domain() const{ return (*this)[0].img().d;}
  void expandBoundary(std::vector<int> &margin, _real value){
    for(unsigned int k=0; k<size(); k++)
      (*this)[k].img().expandBoundary(margin, value);
  }
  void expandBoundary(std::vector<int> &margin){
    for(unsigned int k=0; k<size(); k++)
      (*this)[k].img().expandBoundary(margin);
  }
  void expandBoundary(Domain &margin, _real value){
    for(unsigned int k=0; k<size(); k++)
      (*this)[k].img().expandBoundary(margin, value);
  }
  void expandBoundary(Domain &margin){
    for(unsigned int k=0; k<size(); k++)
      (*this)[k].img().expandBoundary(margin);
  }
};



void affineInterp(deformableMultimodalImage &src, deformableMultimodalImage &res, const _real mat[DIM_MAX][DIM_MAX+1]) ;

#endif
