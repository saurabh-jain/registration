/**
   deformableObject.h
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

#ifndef _DEFOBJECT_
#define _DEFOBJECT_
#include "Vector.h"
#include "param_matching.h"

/**
Generic class for diffeomorphisms acting on objects
*/
template <class T> class deformableObject{
public:
  void infinitesimalAction(const VectorMap &v, T &Z) const {cerr << "this function is not implemented" << endl ; exit(1);}
  void getMomentum(const T &Z, VectorMap &v) const {cerr << "this function is not implemented" << endl ; exit(1);}
  virtual ~deformableObject(){};

  // required members
  virtual void get_template(const param_matching &param) {cout << "get_template component is not implemented" << endl ; exit(1);}
  virtual void get_target(const param_matching &param) {cout << "get_target component is not implemented" << endl ; exit(1);}
  virtual void get_binaryTemplate(const param_matching &param) {cout << "get_binaryTemplate component is not implemented" << endl ; exit(1);}
  virtual void get_binaryTarget(const param_matching &param) {cout << "get_binaryTarget component is not implemented" << endl ; exit(1);}
  //  void get_image(char *file, int dim) {cerr << "this function is not implemented" << endl ; exit(1);}
  void crop (const Domain &D, deformableObject &source) const {cerr << "this function is not implemented" << endl ; exit(1);}
  deformableObject& copy (const deformableObject &source) {return *this;};
  void rescale(const Domain &D, deformableObject &res) const {cerr << "this function is not implemented" << endl ; exit(1);}
  Domain & domain(){ Domain * d = new Domain[1] ; return d[0];}
  void affineInterp(deformableObject &res, const _real mat[DIM_MAX][DIM_MAX+1]) const {cerr << "this function is not implemented" << endl ; exit(1);}

} ;
#endif
