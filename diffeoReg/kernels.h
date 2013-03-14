/**
   kernels.h
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

#ifndef _KERNELS_
#define _KERNELS_
#include "pointSet.h"

static double myExp(const double x){if (x>100) return exp(100); else if (x<-100) return exp(-100); else return exp(x);}

class GaussKernel{
public:
  GaussKernel(){sigma = 1 ; _sig2 = 2 ;}
  GaussKernel(double sig){ sigma=sig; _sig2 = 2*sig*sig;}
  void setWidth(const double sig){ sigma=sig; _sig2 = 2*sig*sig;}

  double operator()(const Point &x, const Point &y) const {
    double res = 0, u ;
    for (unsigned int k=0; k<x.size(); k++) {
      u = x[k]-y[k] ;
      res += u*u ;
    }
    return myExp(- res/_sig2) ;
  }

  double diff(const Point &x, const Point &y) const {
    double res = 0, u ;
    for (unsigned int k=0; k<x.size(); k++) {
      u = x[k]-y[k] ;
      res += u*u ;
    }
    return -myExp(- res/_sig2)/_sig2 ;
  }
  double diff2(const Point &x, const Point &y) const {
    double res = 0, u ;
    for (unsigned int k=0; k<x.size(); k++) {
      u = x[k]-y[k] ;
      res += u*u ;
    }
    return myExp(- res/_sig2)/(_sig2*_sig2) ;
  }
private:
  double sigma, _sig2 ;
} ;
#endif 
