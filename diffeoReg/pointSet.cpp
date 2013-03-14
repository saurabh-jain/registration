/**
   pointSet.cpp
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
#include "pointSet.h"

void affineInterp(const PointSet &src, PointSet &res, const double mat[DIM_MAX][DIM_MAX+1])
{
  int N = src.dim() ;
  res.al(src.size(), src.dim()) ;
  for (unsigned int i=0; i< src.size(); i++) {
    for (int j=0; j< N; j++) {
      res[i][j] = mat[j][N] ;
      for(int jj=0; jj< N; jj++)
	res[i][j] += mat[j][jj] * src[i][jj] ;
    }
  }
}
