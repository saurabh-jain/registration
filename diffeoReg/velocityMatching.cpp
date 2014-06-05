/**
   velocityMatching.cpp
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
#include "velocity.h"


int main(int argc, char** argv)
{
  if (argc < 2)
    {
      cout << "syntax: velocityMatching param_file" << endl ;
      exit(1) ;
    }

  Velocity mo(argv[1], argc, argv) ;
  //  mo.param.read(argc, argv) ;

  // mo.param.verb= 1 ;

  mo.initialPrint() ; 

  // cout << "starting matching" << endl ;
  mo.matching() ;
  //  mo.gradientImageMatching(mo.Z0) ;


  //  mo.PrintTime(mo.Lv0) ;
  
}


