/**
   deformableObjects.cpp
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

#include "deformableImage.h"
//#include "deformableMultimodalImage.h"



void affineInterp(deformableImage &src, deformableImage &res, const _real mat[DIM_MAX][DIM_MAX+1])
{
  affineInterp(src.img(), res.img(), mat); 
}

/*
void affineInterp(deformableMultimodalImage &src, deformableMultimodalImage &res, const _real mat[DIM_MAX][DIM_MAX+1])
{
  res.resize(src.size()) ;
  for(unsigned int k=0; k<src.size(); k++)
    affineInterp(src[k].img(), res[k].img(), mat); 
}
*/


