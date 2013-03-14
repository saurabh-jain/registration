/**
   metamorphosis.cpp
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

//#include "morphing2.h"
#include "morphingNew.h"
#ifdef _PARALLEL_
#include "mkl.h"
#endif

int main(int argc, char** argv)
{
  if (argc < 2)
    {
      cout << "syntax: match_im param_file" << endl ;
      exit(1) ;
    }

  Morphing mo(argv[1], argc, argv) ;
  //  mo.param.verb= 1 ;

   mo.param.tolGrad = 0 ;
   mo.param.lambda = 1 / (mo.param.sigma*mo.param.sigma) ;

#ifdef _MKL_
   mkl_set_num_threads(mo.param.nb_threads) ;
   mkl_set_dynamic(false) ;
#endif

  if (mo.param.matchDensities) {
    double s1, s2 ;
    s1 = mo.Template.img().sum() ;
    s2 = mo.Target.img().sum() ;
    mo.Target.img() *= s1/s2 ;
    cout << mo.Template.img().sum() << " " <<  mo.Target.img().sum() << endl ;
  }
  mo.initialPrint() ;  

  // minitest
  // VectorMap phi, v, dv, w, GI1 ;
  // Vector I, I1, I2 ;
  // double e1, e2 ;
  // I.copy(mo.Template.img());
  // phi.idMesh(I.d) ;
  // v.zeros(phi.d) ;
  // v[0].copy(0.55) ;
  // v[1].copy(-0.5) ;
  // v *= mo.mask ;
  // v += phi ;
  // dv.zeros(phi.d) ;
  // dv[0].copy(0.0002) ;
  // dv[1].copy(0.001) ;
  // dv *= mo.mask ;
  // w.copy(v) ;
  // w += dv ;
  // //v.multilinInterp(I, I1) ;
  // //w.multilinInterp(I, I2) ;
  //   v.multilinInterpDual(I, I1) ;
  //  w.multilinInterpDual(I, I2) ;
  // e1 = I2.norm2() - I1.norm2() ;
  // //v.multilinInterpGradient(I1, GI1) ;
  //  v.multilinInterpGradient(I1, GI1) ;
  // GI1 *= I;
  // e2 = 2 * dv.scalProd(GI1) ;
  // cout << e1 << " " << e2 << endl ;
  // return(0);


  mo.morphing() ;
  mo.Print() ;
  
}


