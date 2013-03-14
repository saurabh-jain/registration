/**
   byteSwap.h
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

#ifndef _BYTESWAP_
#define _BYTESWAP_

static void byteSwap(char * ptr, unsigned int size) {
  char q ;
  for (unsigned int k=0; k<size/2; k++) {
    q = ptr[k] ;
    ptr[k] = ptr[size-1-k] ;
    ptr[size-1-k] = q ;
  }
}



#endif
