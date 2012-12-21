/**
   deformableImage.h
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

#ifndef _DEFIMAGE_
#define _DEFIMAGE_

#include "deformableObject.h"

/**
   Class associated to the action of diffeomorphisms on Vectors.
  Contains a vector and its gradient
*/
class deformableImage: public deformableObject<Vector> {
public:
  typedef Vector Tangent ;

  /**
Infinitesimal action of a vector field on the object
  */
  void infinitesimalAction(const VectorMap &v, Tangent &Z) const{
    _real eps =0.01 ;
    VectorMap id, foom ;
    id.idMesh(v.d) ;
    foom.copy(v) ;
    foom *= - eps ;
    foom += id ;
    foom.multilinInterp(_img, Z) ;
    Z -= _img ;
    Z /= eps ;
  }
  void infinitesimalActionALT(const VectorMap &v, Tangent &Z) const{
    v.scalProd(_gradient, Z) ;
    Z *= -1 ;
  }

  /**
Horizontal momentum associated to a vector variation
  */
  void getMomentum(const Tangent &Z, VectorMap &Lv)  const{
    // Lv.copy(_gradient) ;
    Lv.copy(_normal) ;
    Lv *= Z ;
    Lv *= -1 ;
  }

  /**
computes the gradient of the image
  */
  void computeGradient(vector<_real> &resol, double gradientThreshold0){ 
    ::gradient(_img, _gradient, resol) ;
    double gradientThreshold ;
    if (gradientThreshold0 > 0)
      gradientThreshold = gradientThreshold0 ;
    else
      gradientThreshold = (_img.max() - _img.min())/100 ;
    Vector weight ;
    _gradient.norm(weight) ;
    weight += gradientThreshold ;
    _normal.copy(_gradient) ;
    for(unsigned int j=0; j<weight.size(); j++)
      if (weight[j] > gradientThreshold)
	for (unsigned int i=0; i<_gradient.size(); i++)
	  _normal[i][j] /= weight[j] ;
      else
	for (unsigned int i=0; i<_gradient.size(); i++)
	  _normal[i][j] = 0 ;
  }
  VectorMap & gradient() { return _gradient ;}
  VectorMap gradient() const { return _gradient ;}

  VectorMap & normal() { return _normal ;}
  VectorMap normal() const { return _normal ;}

  Vector & img() {return _img;}
  Vector img() const {return _img;}

  void zeros(Domain &D) {_img.zeros(D);}
  void zero() {_img.zero();}
  void subCopy(const deformableImage &x, const Ivector &MIN0, const Ivector &MAX0){ _img.subCopy(x._img,MIN0,MAX0);}

  // needed members
  void get_image(const string &file, int dim) {_img.get_image(file.c_str(), dim); }
  void get_image(const char *file, int dim) {_img.get_image(file, dim); }

  void crop (const Domain &D, deformableImage &source) const {_img.crop(D, source._img);}
  deformableImage& copy (const deformableImage &source) {_img.copy(source._img); return *this;};
  void rescale(const Domain &D, deformableImage &res) const {_img.rescale(D, res._img) ; }
  Domain & domain(){ return _img.d;}
  Domain  domain() const{ return _img.d;}

  void expandBoundary(std::vector<int> &margin, _real value){ _img.expandBoundary(margin, value);}
  void expandBoundary(std::vector<int> &margin){_img.expandBoundary(margin);}
  void expandBoundary(Domain &margin, _real value){_img.expandBoundary(margin, value);}
  void expandBoundary(Domain &margin){_img.expandBoundary(margin);}
  void flip(int dm){ Vector tmp; _img.flip(tmp, dm) ; _img.copy(tmp) ;}
  void revertIntensities(){
    double mm = _img.max() ;
    _img *=-1 ;
    _img += mm ;
  }
  void binarize(double t, int M){ _img.binarize(t,M) ;}
  void  scaleScalars(double z) {
    _real M0 = _img.max();
      _img *= z/(M0+0.001) ;
  }

protected:
  Vector _img ;
  VectorMap _gradient ;
  VectorMap _normal ;
};

void affineInterp(deformableImage &src, deformableImage &res, const _real mat[DIM_MAX][DIM_MAX+1]) ;

#endif
