/**
   PointSetMatchingAffine.h
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

#ifndef _PSAE_
#define _PSAE_
#include "pointSetMatching.h"

#ifndef DIM_MAX
#define DIM_MAX 5 ;
#endif 
static int _T = 10 ;
//static int _affine_time=10 ;
class BasicPointSetAffineEnergy{
public:
  virtual _real operator() (const Matrix& gamma) const {
    cout << "not implement in BasicPointSetAffineEnergy" << endl;
    return -1 ;
  }
  PointSet Tp, Tg;
  std::vector<_real> x0 ;
  std::vector<_real> sz ;
  virtual ~BasicPointSetAffineEnergy(){};
};

class PointSetAffineEnergy: public BasicPointSetAffineEnergy {
public:
  void init(PointSet &TP, PointSet &TG, std::vector<_real> &SZ, std::vector<_real> &X0) {
    Tp.copy(TP) ;
    Tg.copy(TG);
    sz.resize(SZ.size()) ;
    for(unsigned int k=0; k<sz.size(); k++)
      sz[k] = SZ[k]  ;
    x0.resize(X0.size()) ;
    for(unsigned int k=0; k<x0.size(); k++)
      x0[k] = X0[k]  ;
  }
  _real operator()(const Matrix &gamma) const {
    PointSet DI ;
    Matrix AT ;
    unsigned int N = gamma.nRows()-1 ;
    gamma.integrate(AT, _T) ;

    _real mat[DIM_MAX][DIM_MAX+1] ;
    for (unsigned int i=0; i<N; i++)
      for (unsigned int j=0; j<=N; j++)
	mat[i][j] = AT(i, j) ;

    affineInterp(Tg, DI, mat) ;

    DI -= Tp ;
    return DI.norm2() ;
  }
  virtual ~PointSetAffineEnergy(){};
};


#endif
