/**
   initialMomentumMatchingSYM.cpp
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

#include "shooting.h"

int main(int argc, char** argv)
{

  if (argc != 2)
    {
      cout << "syntax: initialMomentumMatchingSYM param_file1" << endl ;
      exit(1) ;
    }

  ShootingSYM mo(argv[1], argc, argv) ;

  mo.param.verb= 1 ;

  mo.initialPrint() ;  
  VectorMap Lv, Lv0 ;
  if (mo.param.foundInitialMomentum) {
    VectorMap Lv0 ;
    Lv0.read(mo.param.fileInitialMom) ;
  }
  else 
    Lv0.zeros(mo.Template.domain()) ;

    mo.gradientImageMatching(Lv0) ;
    //  mo.gradDesc(Lv0, Lv, mo.param.nb_iter, 0.001, mo.param.epsMax, mo.param.minVarEn, mo.param.gs, mo.param.verb, (void *) &(mo.param)) ;
  mo.Print() ;
}


