/**
   Ivector.h
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

#ifndef _IVECTOR_
#define _IVECTOR_

// index class for multidimensional arrays

#include <new>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <cmath>
#include <vector>

typedef double _real ;
using namespace std ;

#ifndef DIM_MAX
#define DIM_MAX 5
#endif

/* Defines multi-dimensional indices
**/
class Ivector:public std::vector<int>
{
  friend class Domain ;
public:
  /**
     1 if the index is not empty
  */
  int ok(){return (size()>0) ;}

  /** 
      Checks that all components are nonnegative
  */ 
  int pos(){unsigned int i; if (size()<=0) return 0; for(i=0; i<size(); i++) if ((*this)[i] < 0) return 0 ;
                   return 1;}
  /**
     like size()
  */
  //  int length() const ;

  Ivector& copy(const Ivector &src) {
    unsigned int i ;
    resize(src.size()) ;
    for (i=0; i<size(); i++)
      (*this)[i] = src[i] ;
    return *this ;
  }

  int operator == (const Ivector &dest) const {
    unsigned int i ;
    if (dest.size() != size())
      return 0 ;
    for(i=0; i<dest.size(); i++)
      if (dest[i] != (*this)[i])
	return 0 ;
    return 1 ;
  }

  int operator != (const Ivector &dest)  const {
    unsigned int i ;
    if (dest.size() != size())
      return 1 ;
    for(i=0; i<dest.size(); i++)
      if (dest[i] != (*this)[i])
	return 1 ;
    return 0 ;
  }

  int operator <= (const Ivector &dest) const {
    unsigned int i ;

    if (dest.size() != size())
      return 0 ;
    for(i=0; i<dest.size(); i++)
      if ((*this)[i] < dest[i])
	return 0 ;
    return 1 ;
  }

  int operator < (const Ivector &dest) const {
    unsigned int i, eq = 1 ;

    if (dest.size() != size())
      return 0 ;
    for(i=0; i<dest.size(); i++)     {
      if ((*this)[i] < dest[i])
	return 0 ;
      if ((*this)[i] > dest[i])
	eq = 0 ;
    }
    if (eq)
      return 0 ;
    return 1 ;
  }

  int inc(const Ivector &max) {
    Ivector min ;
    min.resize(size()) ;
    min.zero() ;
    return inc(min, max) ;
  }

  int inc(const Ivector &min, const Ivector &max) {
    unsigned int i, j ;
    if ((*this) == max)
      return -1 ;

#ifdef SECURE
    if ((this->size()==0) || !(min <= (*this)) || !((*this) <= max)) {
      cerr << "Incrementation Ivector impossible" << endl ;
      exit(1) ;
    }
#endif

    i = max.size()-1 ;
    while(i>=0) {
      if((*this)[i] < max[i]) {
	(*this)[i]++ ;
	for (j=i+1; j<max.size(); j++)
	  (*this)[j] = min[j] ;
	break ;
      }
      i-- ;
    }

    return i ;
  }

  void zero() {unsigned int i; for (i=0; i<size(); i++) (*this)[i] = 0 ;}
  Ivector& operator =(const Ivector &source) {copy(source); return *this; }

  void operator +=(const std::vector<int> &v) {
#ifdef SECURE
    if (size() != v.size()) {
      cerr << "+= distinct dimensions" << endl ;
      exit(1) ;
    }
#endif

    unsigned int i ;
    for(i=0; i<v.size(); i++)
      (*this)[i] += v[i] ;
  }
      
  void operator -=(const std::vector<int> &v) {
#ifdef SECURE
    if (size() != v.size()) {
      cerr << "+= distinct dimensions" << endl ;
      exit(1) ;
    }
#endif

    unsigned int i ;
    for(i=0; i<v.size(); i++)
      (*this)[i] -= v[i] ;
  }
      
  void operator +=(const int v) {
    unsigned int i ;
    for(i=0; i<size(); i++)
      (*this)[i] += v ;
  }
      
  void operator -=(const int v) {
    unsigned int i ;
    for(i=0; i<size(); i++)
      (*this)[i] -= v ;
  }
      
  void min(const Ivector &v) {
#ifdef SECURE
    if (size() != v.size()) {
      cerr << "+= distinct dimensions" << endl ;
      exit(1) ;
    }
#endif

    unsigned int i ;
    for(i=0; i<v.size(); i++)
      if ((*this)[i] > v[i])
	(*this)[i] = v[i] ;
  }
      
  void max(const Ivector &v) {
#ifdef SECURE
    if (size() != v.size()) {
      cerr << "+= distinct dimensions" << endl ;
      exit(1) ;
    }
#endif

    unsigned int i ;
    for(i=0; i<v.size(); i++)
      if ((*this)[i] < v[i])
	(*this)[i] = v[i] ;
  }
} ;


/**
   Range for multidimensional arrays. Defined by a minimum and a maximum index.
*/
class Domain
{
public:
  unsigned int n ;
  unsigned int length ;

  Domain() {n = 0; length = 0 ; }
  Domain(const Ivector &v){n = v.size() ;M.copy(v); m.resize(v.size()); m.zero(); cum.resize(v.size()) ; calc_cum();}
  Domain(const Ivector &u, const Ivector &v){create(u, v);}
  void create(const Ivector &u, const Ivector &v){
#ifdef SECURE
    if (u.size() != v.size())
      { cerr << "Domain: inconsistent dimensions" << endl ; exit(1) ;}
#endif
    M.copy(v); m.copy(u); n = v.size(); cum.resize(v.size()) ; calc_cum();
}

  Domain& operator=(const Domain &src){return copy(src);} 
  int operator != (const Domain &dest) const {
    return (dest.m != m || dest.M != M) ;
  }

  int dimension() const {return n;}
  int minWidth() {
    int res = M[0] - m[0] + 1 ;
    for (unsigned int k=1; k<n; k++)
      if (res > M[k]-m[k] + 1 )
	res = M[k] - m[k] + 1 ;
    return res ;
  }
  int maxWidth() {
    int res = M[0] - m[0] + 1 ;
    for (unsigned int k=1; k<n; k++)
      if (res < M[k]-m[k] + 1 )
	res = M[k] - m[k] + 1 ;
    return res ;
  }
  int getCum(int i) const {return cum[i];}
  int getCumMin() const {return cumMin;}

  /** 
      gets vector position from multi index
  */

  /**
     gets index from vector position
  */

  /**
     translation parallel to one of the dimensions
  */
  inline int move(int step, int direction) { return step * cum[direction];}



  /**
     stores lower bound in index I
  */
  void putm(Ivector &I) const {I.resize(n); for(unsigned int i=0; i<n;i++) I[i] = m[i];}
  void shiftPlus(Ivector &S){ m+=S ;M+=S;}
  void shiftMinus(Ivector &S){ m-=S ;M-=S;}
  
  /**
     stores upper bound in index I
  */
  void putM(Ivector &I) const {I.resize(n); for(unsigned int i=0; i<n;i++) I[i] = M[i];}
  
  /** 
      sets value of ith index of  lower bound
  */
  void setm(int i, int k) {m[i] = k ;}
  /**
     sets value of ith index of  upper bound
  */
  void setM(int i, int k) {M[i] = k;}
  /**
     returns value of ith index of  lower bound
  */
  int getm(const int i) const {return m[i];}
  /**
     returns value of ith index of  upper bound
  */
  int getM(const int i) const {return M[i];}

  /**
     increments index I within the domain
  */
  int inc(Ivector &I) const {return I.inc(m, M);}

  /** relative position */
  int rPos(const int i0, const int dim, const int step) const { return i0 + step * cum[dim] ; }
  _real rPos(std::vector<_real>::iterator i0, const int dim, const int step) const { return *(i0 + step * cum[dim]) ; }

  /**
     resize the domain to u dimensions
  */
  void resize(int u) {
    m.resize(u) ;
    M.resize(u) ;
    cum.resize(u) ;
    n = u ;
  }

  Domain& copy(const Domain &s) {
    n = s.n ;
    length = s.length ;
    m.copy(s.m) ;
    M.copy(s.M) ;
    cum.copy(s.cum) ;
    cumMin = s.cumMin ;
    return (*this) ;
  }


  /**
     computation needed for vector-array transcription
  */
  int calc_cum2() {
    int i ;

    if (n>0) {
      cum[0] = 1 ;
      for(i=1; i<(int)n; i++)
	cum[i+1] = (M[i] - m[i] +1) * cum[i] ;
      length = cum[n-1] * (M[n-1] - m[n-1] + 1) ;
      cumMin = 0 ;
      for(i=0; i<(int) n; i++)
	cumMin += cum[i]*m[i] ;
      return length ;
    }
    return -1 ;
  }

  int calc_cum() {
    int i ;

    if (n>0) {
      cum[n-1] = 1 ;
      for(i=n-2; i>=0; i--)
	cum[i] = (M[i+1] - m[i+1] +1) * cum[i+1] ;
      length = cum[0] * (M[0] - m[0] + 1) ;
      cumMin = 0 ;
      for(i=0; i<(int) n; i++)
	cumMin += cum[i]*m[i] ;
      return length ;
    }
    return -1 ;
  }

  /**
     Tests if not empty
  */
  int positive() const { return length > 0 ; }

  int operator == (const Domain &dest) const { return (dest.m==m && dest.M==m) ;}


  /**
     vector index from array indices
  */
  int position(const Ivector & u) const {
    unsigned int i ;
#ifdef SECURE
    if (u.size() != n) {
      cerr << "Position: distinct dimensions" << endl ;
      exit(1) ;
    }
    for(i=0; i<n; i++)
      if (u[i] < m[i] || u[i] > M[i]) {
	cerr << "position: index out of limits" << endl 
	     << "index : " << i << " bornes : " << m[i] << " and " << M[i]
	     << "  value : " << u[i] << endl ;
	exit(1) ;
      }
#endif
    
    int j=-cumMin ; ;
    for(i=0; i<n; i++)
      j += u[i] * cum[i] ;
    return j ;
  }


  /** 
      array indices (u) from vector index (i)
  */
  void fromPosition(Ivector & u, const int i) const {
#ifdef SECURE
    if (u.size() != n) {
      cerr << "fromPosition: distinct dimensions" << endl ;
      exit(1) ;
    }
#endif

    u.resize(n) ;
    unsigned int j ;
    int ii = i;
    for(j=0; j<n; j++) {
      u[j] = ii/cum[j] + m[j];
      ii = ii%cum[j] ;
    }
  }

  int position_unsecure(const Ivector & u)  {
    unsigned int i ;
#ifdef SECURE
    if (u.size() != n) {
      cerr << "Position: distinct dimensions" << endl ;
      exit(1) ;
    }
#endif

    for(i=0; i<n; i++)
      if (u[i] < m[i] || u[i] > M[i])
	return -1 ;

    int j=-cumMin ; ;
    for(i=0; i<n; i++)
      j += u[i] * cum[i] ;

    return j ;
  }


private:
  Ivector m, M, cum ;
  int cumMin ;
} ;

/*
// boolean operators
int operator == (const Ivector &dest, const Ivector &source) ;
int operator != (Ivector &dest, Ivector &source) ;
int operator <= (Ivector &dest, Ivector &source) ;
int operator < (Ivector &dest, Ivector &source) ; 
//int operator != (Domain &dest, Domain &source) ;
*/

//Ivector &min(Ivector &u, Ivector &v) ;
//Ivector &max(Ivector &u, Ivector &v) ;

inline ostream & operator << (ostream &os, const Domain& D) {
  for (unsigned int i=0; i< D.n; i++)
    os << D.getm(i) << " " ;
  os << endl ;
  for (unsigned int i=0; i< D.n; i++)
    os << D.getM(i) << " " ;
  os << endl ;
  return os ;
}

inline ostream & operator << (ostream &os, const Ivector& D) {
  for (unsigned int i=0; i< D.size(); i++)
    os << D[i] << " " ;
  os << endl ;
  return os ;
}

#endif
