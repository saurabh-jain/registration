/**
   VectorBase.h
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

#ifndef _VECTORBASE_
#define _VECTORBASE_
// Multi dimensional array

#ifdef _PARALLEL_
#include <omp.h>
#include <mkl.h>
#endif
#include "Ivector.h"
#include <algorithm>


//MST_USING_NAMESPACE_MST;

inline int absMax(int a, int b){ return (int) ( (a+b + abs(a-b))/2) ;}
inline int absMin(int a, int b){ return (int) ( (a+b - abs(a-b))/2) ;}
inline _real absMax(_real a, _real b){ return (a+b + fabs(a-b))/2 ;}
inline _real absMin(_real a, _real b){ return (a+b - fabs(a-b))/2 ;}
#ifndef minV
#define minV(a,b) a>b? b: a
#endif
#ifndef maxV
#define maxV(a,b) a<b? b: a
#endif


template <class NUM> class _VectorBase: public std::vector<NUM>
{
public:
  typedef typename  std::vector<NUM>::iterator _iterator ;
  Domain d ;
  Domain & domain(){return d;}


  Domain domain() const {return d;}
  _VectorBase(){} ;
  _VectorBase(Ivector& T) {  al(T);}

  _VectorBase<NUM>& copy (const _VectorBase<NUM> &src){
    al(src.d) ;
    std::copy(src.begin(), src.end(), (*this).begin()) ;
    return (*this) ;
  }

  _VectorBase<NUM>& copy (const NUM x){
    //  virtual _Vector<NUM>& copy (const NUM x){
    int i, le = size();
    for (i=0; i<le; i++)
      (*this)[i] = x ;
    return (*this) ;
  }

  //virtual ~_Vector() {};

  _VectorBase<NUM> (const _VectorBase<NUM> &src){copy(src) ;}
  _VectorBase<NUM>& operator = (const _VectorBase<NUM> &src){copy(src) ; return *this;}
  _VectorBase<NUM>& operator = (const NUM x){copy(x) ; return *this;}

  /**
     copy from Array3D structure (analyze interface)
  */
  /*
  _VectorBase<NUM> &copy(const Array3D<NUM>& A) {
    if(!A.isEmpty()){
      Ivector I, m, M ;
      m.resize(3) ;
      M.resize(3) ;
      I.resize(3) ;
      m[0] = A.getZmin() ;
      m[1] = A.getYmin() ;
      m[2] = A.getXmin() ;
      M[0] = A.getZmax() ;
      M[1] = A.getYmax() ;
      M[2] = A.getXmax() ;
      Domain D(m, M) ;
      al(D) ;
      for(int z = A.getZbegin(); z <= A.getZend(); z++)
	for(int y = A.getYbegin(); y <= A.getYend(); y++)
	  for(int x = A.getXbegin(); x <= A.getXend(); x++){
	    I[0] = z; I[1] = y; I[2] = x ;
	    _real uu = *(A.data(z, y, x)) ;
	    (*this)[I] = uu; 
	  }
    }
    return *this;
  }
  */

  unsigned int size() const { return vector<NUM>::size();}

/**
  allocation between 0 and u
*/
  int al(const Ivector &u) {
    Ivector Zero ;
    Zero.resize(u.size()) ;
    Zero.zero() ;
    Domain D(Zero, u) ;
    return al(D) ;
  }

  /**
     allocation between u and v
  */
  int al(const Ivector &u, const Ivector &v)
  {
    Domain D(u, v) ;
    D.calc_cum() ;
    return al(D) ;
  }

  /**
     allocation over a Domain
  */
  //  virtual int al(const Domain &u)
  int al(const Domain &u)
  {
    if (u.positive())
      {
	d.copy(u) ;
	vector<NUM>::resize(u.length) ;
      }
    return u.length ;
  }


  unsigned int length() const {return d.length;}
  int pos(){return d.positive();}

  /** value from index
   */
  NUM & operator [](Ivector &I) 
  {
    int i = d.position(I);
    return (*this)[i] ;
  }

  /** value from index
   */
  NUM  operator [](const Ivector &I) const 
  {
    int i = d.position(I);
    return (*this)[i] ;
  }

  /** value from integer
   */
  NUM & operator [](const unsigned int i) 
  {
    return *(this -> begin() + i) ;
  }
  /** value from integer
   */
  NUM  operator [](const unsigned int i) const 
  {
    return *(this -> begin() + i) ;
  }

  /** Shifts in given direction
   */
  void shift(int dir, int step, _VectorBase<NUM> &dest) const
  {
    dest.copy((*this)) ;
    for(unsigned int c =0; c< length(); c++) {
      unsigned int ii = d.rPos(c, dir, step) ;
      if (ii >=0 && ii < length())
	dest[c] =  (*this)[ii] ;
    }
  }

  /** crops source in a subdomain
   */
  void crop (const Domain &D, _VectorBase<NUM> &dest) const 
  {
    Ivector im, iM ;
    Domain DD ;
    im.resize(D.n) ;
    iM.resize(D.n) ;
    for (unsigned int i=0; i< d.n; i++) {
      im[i] = 0 ;
      iM[i] = D.getM(i) - D.getm(i) ;
    }
    DD.create(im, iM) ;
    dest.al(DD) ;
    int i, le = dest.size();
    Ivector I ;
    D.putm(I) ;
    for (i=0; i<le; i++){
      dest[i] = (*this)[I] ;
      D.inc(I) ;
    }
  }

  void crop (const Domain &D)
    {
      _VectorBase<NUM> tmp ; 
      tmp.copy(*this) ;
      tmp.crop(D, *this) ;
    }

  /**
     crops from x in *this 
  */
  void subCopy(const _VectorBase<NUM> &x)
  {
    Ivector MIN, MAX, I, tmp, M, m ;
    long c=0;
    long iy ;
    
    x.d.putm(MIN) ;
    x.d.putM(MAX) ;
    d.putm(m) ;
    d.putM(M) ;
 
    // cout << MIN << MAX << m << M << endl ;
    if (d.n == 2) {
      for (int k0=MIN[0]; k0 <= MAX[0] ; k0++) 
	for (int k1=MIN[1]; k1 <= MAX[1] ; k1++) {
	    iy = ((k0-m[0]) * (M[1]-m[1]+1) + (k1-m[1])) ;
	    (*this)[iy] = x[c++] ;
	  }
    }
    else if (d.n == 3) {
      for (int k0=MIN[0]; k0 <= MAX[0] ; k0++) 
	for (int k1=MIN[1]; k1 <= MAX[1] ; k1++) 
	  for (int k2=MIN[2]; k2 <= MAX[2] ; k2++) {
	    iy = ((k0-m[0]) * (M[1]-m[1]+1) + (k1-m[1])) * (M[2]-m[2]+1) + k2 - m[2] ;
	    (*this)[iy] = x[c] ;
	    //	    if (c < 1000)
	    //cout << iy << " ; " << c <<" -- " << flush  ;
	    c++ ;
	  }
      //      cout << endl ;
      /*
      double test = 0 ;
      Ivector II ;
      II.resize(3) ;
      for (int k0=MIN[0]; k0 <= MAX[0] ; k0++) 
	for (int k1=MIN[1]; k1 <= MAX[1] ; k1++) 
	  for (int k2=MIN[2]; k2 <= MAX[2] ; k2++) {
	    II[0] = k0 ;
	    II[1] = k1;
	    II[2] = k2 ;
	    test = test + fabs((*this)[II] - x[II]) ;
	  }

      cout << "Test " << test << "::" << endl ;
      */
    }
    else {
    I.copy(MIN) ;
    iy = d.position(I) ;
    while(c < (int) x.length()) {
	(*this)[iy] = x[c] ;
	c++ ;
	I.inc(MIN, MAX) ;
	iy = d.position(I) ;
      }
    }
  }

  void subCopy(const _VectorBase<NUM> &x, const Ivector &MIN0, const Ivector &MAX0)
  {
    Ivector I ;
    unsigned int c=0;
    int iy ;

    I.copy(MIN0) ;
    iy = d.position(I) ;

    while(c < x.d.length) {
      (*this)[iy] = x[c] ;
      c++ ;
      I.inc(MIN0, MAX0) ;
      iy = d.position(I) ;
    }
  }

  void symCopy(const _VectorBase<NUM> &x)
  {
    Ivector I, I0, MIN, MAX, MIN0, MAX0 ;
    unsigned int c=0;
    int iy ;

    d.putm(MIN) ;
    d.putM(MAX) ;
    x.d.putm(MIN0) ;
    x.d.putM(MAX0) ;
    I0.copy(MIN0) ;
    while(c < x.d.length) {
      I.copy(I0) ;
      for (unsigned int i=0; i<d.n; i++) {
	if (I[i] < 0)
	  I[i] = MAX[i] + I[i] + 1 ;
	else
	  I[i] += MIN[i] ;
      }
      iy = d.position(I) ;	
      (*this)[iy] = x[c] ;
      c++ ;
      I0.inc(MIN0, MAX0) ;
    }
  }

  void flip(_VectorBase<NUM> &res, int dm) {
    //    cout << "start flip" << endl ;
    res.al(d) ;
    Ivector I, I0, MIN, MAX ;
    unsigned int c=0;
    int iy ;

    d.putm(MIN) ;
    d.putM(MAX) ;
    I0.copy(MIN) ;
    while(c < d.length) {
      I.copy(I0) ;
      I[dm] = MAX[dm] - I[dm] + MIN[dm] ;
      //      cout << I[dm] << " " << I0[dm] << " ; " << flush ;
      iy = d.position(I) ;	
      res[iy] = (*this)[c] ;
      c++ ;
      I0.inc(MIN, MAX) ;
    }
    //    cout << "end flip" << endl ;
  }

  /**
     crops *this in x using allocated domain for x
  */
  void extract(_VectorBase<NUM> &x)
  {
    Ivector I, MIN, MAX ;
    unsigned int c=0 ;
    int iy ;
    x.d.putm(MIN) ;
    x.d.putM(MAX) ;
    I.copy(MIN) ;
    iy = d.position(I) ;
    
    while(c < x.length())
      {
	x[c] = (*this)[iy] ;
	c++ ;
	I.inc(MIN, MAX) ;
	iy = d.position(I) ;
      }
  }

  void extract(_VectorBase<NUM> &x, const Ivector &MIN0, const Ivector &MAX0)
  {
    Ivector I ;
    unsigned int c=0;
    int iy ;
    
    I.copy(MIN0) ;
    iy = d.position(I) ;
    
    while(c < x.d.length) {
      x[c] = (*this)[iy] ;
      c++ ;
      I.inc(MIN0, MAX0) ;
      iy = d.position(I) ;
    }
  }


  void expandBoundaryCentral(int margin, NUM value)
    {
      //      cout << "value = " << value << endl ;
      _VectorBase<NUM> tmp ;
      tmp.copy(*this) ;
      Ivector I, J ;
      d.putm(I) ;
      d.putM(J) ;
      //      cout << d ;

      I-= margin ;
      J+= margin ;
      // J+=margin ;

      al(I,J) ;
      copy(value) ;
      I+= margin ;
      J-= margin ;
      //      cout << d ;
      subCopy(tmp, I, J) ;
      //      write_imagesc("test") ;
    }


  void expandBoundary(std::vector<int> &margin, NUM value)
    {
      //      cout << "value = " << value << endl ;
      _VectorBase<NUM> tmp ;
      tmp.copy(*this) ;
      Ivector I, J ;
      d.putm(I) ;
      d.putM(J) ;
      //      cout << d ;

      //      I-= margin ;
      J+= margin ;
      J+=margin ;

      al(I,J) ;
      copy(value) ;
      I+= margin ;
      J-= margin ;
      //      cout << d ;
      subCopy(tmp, I, J) ;
    }


  void expandBoundary(Domain &D, NUM value)
    {
      //      cout << "value = " << value << endl ;
      _VectorBase<NUM> tmp ;
      tmp.copy(*this) ;
      Ivector I, J ;

      Ivector diff1, diff2 ;
      diff1.resize(d.n);
      diff2.resize(d.n) ;

      for (unsigned int k=0; k<d.n; k++) {
	diff1[k] = (int) floor(((D.getM(k) - D.getm(k)) - (d.getM(k) - d.getm(k)))/2) ;
	diff2[k] = (D.getM(k) - D.getm(k)) - (d.getM(k) - d.getm(k)) - diff1[k] ;
      }

      al(D) ;
      copy(value) ;
      D.putm(I) ;
      D.putM(J) ;
      I+= diff1 ;
      J-= diff2 ;
      //      cout << d ;
      subCopy(tmp, I, J) ;
    }

  void expandBoundary(int margin, NUM value)
    {
      //      cout << "value = " << value << endl ;
      _VectorBase<NUM> tmp ;
      tmp.copy(*this) ;
      Ivector I, J ;
      d.putm(I) ;
      d.putM(J) ;

      //      I-= margin ;
      J+= 2*margin ;

      al(I,J) ;
      copy(value) ;
      I+= margin ;
      J-= margin ;
      subCopy(tmp, I, J) ;
    }


  /**
     save in Array3D structure (analyze interface)
  */
  /*
  void writeInArray3D(Array3D<NUM> &A){
    if(d.n == 3){
      A.setDim(d.getM(0) - d.getm(0) + 1, d.getM(1) - d.getm(1) + 1, d.getM(2) - d.getm(2) + 1) ; 
      Ivector I ;
      d.putm(I) ;
      for (unsigned int i=0; i<length(); i++) {
	_real uc ;
	if ((*this)[i] < 0)
	  uc = 0 ;
	else if ((*this)[i] > 255)
	  uc = 255 ;
	else 
	  uc = (unsigned char) fabs((*this)[i]) ;
	
	A[I[0]-d.getm(0)][I[1]-d.getm(1)][I[2]-d.getm(2)] = uc ;
	d.inc(I) ;
      }
    }
    else {
      cerr << "writeInArray3D only in 3D" << endl ;
      exit(1) ;
    }
  }
  */

  /**
     save in binary format 
  */
  void write(const char * path)
  {
    ofstream ofs ;
    ofs.open(path) ;
    if (ofs.fail()) {
      cerr << "Unable to open " << path << " in write" << endl ;
      exit(1) ;
    }

    int foo = 12345 ;

    ofs.write((char *) &foo, sizeof(int)) ;
    foo = d.n ;
    ofs.write((char *) &foo, sizeof(int)) ;
    for (unsigned int i=0; i<d.n; i++) {
      foo = d.getm(i) ; 
      ofs.write((char *) &foo, sizeof(int)) ;
    }

    for (unsigned int i=0; i<d.n; i++) {
      foo = d.getM(i) ; 
      ofs.write((char *) &foo, sizeof(int)) ;
    }

    ofs.write((char *) &((*this)[0]), size()*sizeof(NUM)) ;
    ofs.close() ;
  }

  /**
     read from binary format
  */
  void read(const char * path)
  {
    ifstream ifs ;
    ifs.open(path) ;
    if (ifs.fail()) {
      cerr << "Unable to open " << path << " in read" << endl ;
      exit(1) ;
    }

    bool oldVersion ;    
    int foo, N ;
    Ivector m, M ;
    ifs.read((char *) &N, sizeof(int)) ;
    if (N != 12345)
      oldVersion = true ;
    else {
      oldVersion = false ;
      ifs.read((char *) &N, sizeof(int)) ;
    }
    m.resize(N) ;
    M.resize(N) ;
    if (N != 3)
      oldVersion = false ;
    if (oldVersion) {
      for (int i=0; i<N; i++) {
	ifs.read((char *) &foo, sizeof(int)) ;
	m[N-i-1] = foo ;
      }

      for (int i=0; i<N; i++) {
	ifs.read((char *) &foo, sizeof(int)) ;
	M[N-i-1] = foo ;
      }
    }
    else {
      for (int i=0; i<N; i++) {
	ifs.read((char *) &foo, sizeof(int)) ;
	m[i] = foo ;
      }
      
      for (int i=0; i<N; i++) {
	ifs.read((char *) &foo, sizeof(int)) ;
	M[i] = foo ;
      }
    }

    d.create(m, M) ;
    al(d) ;

    if (oldVersion) {    
      Ivector II, MIN, MAX ;
      d.putm(MIN) ;
      d.putM(MAX) ;
      II.resize(N) ;
      
      int k=0 ;
      double *data = new double[size()] ;
      cout << MIN << MAX << " " << size() << endl ;
      ifs.read((char *) data, size()*sizeof(double)) ;
    
      // cout << "loop" << endl ;
      for (int k0=MIN[2]; k0 <= MAX[2] ; k0++) 
	for (int k1=MIN[1]; k1 <= MAX[1] ; k1++) 
	  for (int k2=MIN[0]; k2 <= MAX[0] ; k2++) {
	    II[2] = k0 ;
	    II[1] = k1;
	    II[0] = k2 ;
	    //	    if (k%1 == 0)
	    //cout << k0 << " " << k1 << " " << k2 << " ;; " << flush ;
	    (*this)[II] = data[k++] ;
	  }
    }
    else
      ifs.read((char *) &((*this)[0]), size()*sizeof(NUM)) ;
    ifs.close() ;
  }

};


/**
   Adds arithmetic operation to _VectorBase
*/
template <class NUM> class _Vector: public _VectorBase<NUM>
{
public:
  using _VectorBase<NUM>::d ;
  using _VectorBase<NUM>::copy ;
  using _VectorBase<NUM>::expandBoundary ;
  using _VectorBase<NUM>::expandBoundaryCentral ;

  typedef typename  std::vector<NUM>::iterator _iterator ;
  typedef  std::vector<_Vector<NUM> > __VectorMap ;

  void zeros(const Domain &d_) {this->al(d_); for(unsigned int i=0; i<this->length(); i++) (*this)[i]= 0 ;}
  void ones(const Domain &d_) {this->al(d_); for(unsigned int i=0; i<this->length(); i++) (*this)[i]= 1 ;}



  /** 
      set all values with 0
  */
  void zero()
  {
    int i, le = this->length() ;
    for(i=0; i<le; i++)
      (*this)[i] = 0 ;
  }

  /** 
     binarize relative to a threshold
  */
  void binarize(double binT, double coeff)
    {
      int i, le = this->length() ;
      for(i=0; i<le; i++)
	if ((*this)[i] > binT)
	  (*this)[i] = coeff;
	else
	  (*this)[i] = 0 ;
    }

  void expandBoundaryCentral(int margin) { expandBoundaryCentral(margin, avgBoundary()) ;}
  void expandBoundary(std::vector<int> &margin) { expandBoundary(margin, avgBoundary()) ;}
  void expandBoundary(Domain &margin) { expandBoundary(margin, avgBoundary()) ;}
  void expandBoundary(int margin) { expandBoundary(margin, avgBoundary()) ;}

  /**
     rescales *this to domain D, stores result in res
  */
  void rescale(const Domain &D, _Vector<NUM> &res) const
  { 
    //    cout << "rescale" << endl ;
    Ivector kmin, kmax, I ;
    unsigned int i, k ;
    std::vector<_real> offset ;
    std::vector<_real> ratio, rkmin, rkmax ;
    ratio.resize(D.n) ;
    kmin.resize(D.n) ;
    kmax.resize(D.n) ;
    rkmin.resize(D.n) ;
    rkmax.resize(D.n) ;
    offset.resize(D.n) ;
    
    for (i=0; i<D.n; i++) {
      ratio[i] = (_real) (this->d.getM(i) - this->d.getm(i) +1)/(D.getM(i) - D.getm(i) + 1) ;
      offset[i] = this->d.getm(i) - ratio[i] * D.getm(i) ;
    }

    //  Vector resTmp ;
    res.al(D) ;
    D.putm(I) ;
    unsigned int c=0 ;

    while(c < D.length) {
      for(k=0;k<D.n; k++) {
	kmin[k] = (int) floor(offset[k] + ratio[k] * I[k]) ;
	kmax[k] = (int) ceil(offset[k] + ratio[k] * (I[k]+1)) - 1 ;
	if(kmax[k] >= this->d.getM(k))
	  kmax[k] = this->d.getM(k) ;
	if (kmax[k] == kmin[k])
	  rkmin[k] = ratio[k] ;
	else {
	  rkmin[k] = kmin[k] + 1 - ratio[k] * I[k] ;
	  rkmax[k] = ratio[k] * (I[k] + 1) - kmax[k] ;
	} 
      }

      Domain Dloc(kmin, kmax) ;
      Ivector J ;
      J.copy(kmin) ;
      unsigned int c2=0 ;
      _real u = 0, ww = 0 ; 
      while (c2 < Dloc.length) {
	_real zz = 1  ;
	for (unsigned int kloc = 0; kloc < D.n; kloc ++) 
	  if (J[kloc] == kmin[kloc]) 
	    zz *= rkmin[kloc] ;
	  else if (J[kloc] == kmax[kloc])
	    zz *= rkmax[kloc] ;
	
	ww += zz ;
	u += zz * (*this)[J] ;
	Dloc.inc(J) ;
	c2++ ;
      }
      u /= ww ;
      res[c] = u ;
      D.inc(I) ;
      c++ ;
    }
  }

  // arithmetic operations
  _Vector<NUM> operator + (const _Vector<NUM> & src) const
  { 
    _Vector<NUM> *res = new _Vector<NUM> ; 
    (*res).al(d); 
    for (unsigned int i=0; i<this->length(); i++)    (*res)[i] = (*this)[i] + src[i] ; 
    return (*res); 
  }
  _Vector<NUM> operator - (const _Vector<NUM> & src) const
  { 
    _Vector<NUM> res ; 
    res.al(this->d); 
    for (unsigned int i=0; i<this->length(); i++)    (res)[i] = (*this)[i] - src[i] ; 
    return (res); 
  }
  void sqr() {  for(unsigned int i=0; i<this->length(); i++)  (*this)[i] *= (*this)[i] ;}
  void operator += (double x) { 
    for (_iterator I= (*this).begin(); I!=(*this).end(); ++I)    
      (*I) += x ; 
  }
  void operator -= (double x) { 
    for (_iterator I= (*this).begin(); I!=(*this).end(); ++I)    
      (*I) -= x ; 
  }
  void operator *= (double x) { 
    for (_iterator I= (*this).begin(); I!=(*this).end(); ++I)    
      (*I) *= x ; 
  }
  void operator /= (double x) { 
    for (_iterator I= (*this).begin(); I!=(*this).end(); ++I)    
      (*I) /= x ; 
  }

/*   void operator += (NUM x) {  */
/*     for (_iterator I= (*this).begin(); I!=(*this).end(); ++I)     */
/*       (*I) += x ;  */
/*   } */
/*   void operator -= (NUM x) {  */
/*     for (_iterator I= (*this).begin(); I!=(*this).end(); ++I)     */
/*       (*I) -= x ;  */
/*   } */
/*   void operator *= (NUM x) {  */
/*     for (_iterator I= (*this).begin(); I!=(*this).end(); ++I)     */
/*       (*I) *= x ;  */
/*   } */
/*   void operator /= (NUM x) {  */
/*     for (_iterator I= (*this).begin(); I!=(*this).end(); ++I)     */
/*       (*I) /= x ;  */
/*   } */

  void operator += (const _Vector<NUM> &x) { 
    int i=0 ;
    for (_iterator I= (*this).begin();  I!=(*this).end(); ++I)    
      (*I) += x[i++] ; 
  }

  void operator -= (const _Vector<NUM> &x) { 
    int i=0 ;
    for (_iterator I= (*this).begin();  I!=(*this).end(); ++I)    
      (*I) -= x[i++] ; 
  }

  void operator /= (const _Vector<NUM> &x) { 
    int i=0 ;
    for (_iterator I= (*this).begin();  I!=(*this).end(); ++I)    
      (*I) /= x[i++] ; 
  }

  void operator *= (const _Vector<NUM> &x) { 
    int i=0 ;
    for (_iterator I= (*this).begin();  I!=(*this).end(); ++I)    
      (*I) *= x[i++] ; 
  }

  /**
     sums termwise products between *this and y
  */
  _real sumProd(const _Vector<NUM> &y) const {_real res = 0;  for(unsigned int i=0; i<this->d.length; i++)  res += (*this)[i] * y[i] ;
    return res ;}

  /**
     sum of squares
  */
  _real norm2() const {_real res = 0 ; for (unsigned int i=0; i<this->size(); i++) res += (*this)[i]*(*this)[i] ;  return res ; }

  /**
     squared L2 norm betweem *this and w
  */
  _real dist2(const _Vector<NUM> &w) const { 
    _real res = 0 ;  
    for (unsigned int i=0; i<this->size(); i++) {
      _real u = (*this)[i] - w[i] ;
      res += u*u ;
    }
    return res ;
  }

  /**
     applies fun to each coordinale
  */
  void execFunction(_Vector<NUM> &res,  NUM (*fun)(NUM)) {
    res.al(this->d) ;
    for (unsigned int i=0; i<this->size(); i++)
      res[i] = fun((*this)[i]) ;
  }

  /**
     sum of all components
  */
  _real sum() const  { _real res = 0 ; for (unsigned int i=0; i<this->size(); i++) res += (*this)[i] ; return res ;}

  /**
     Average of the values on the boundary
  */
  _real avgBoundary() const { 
    Ivector I ;
    unsigned int c=0 ;
    _real mm = 0 ;
    int nn = 0 ;

    this->d.putm(I) ;
    while(c < this->d.length) {
      for (unsigned int j=0; j< this->d.n; j++)
	if (I[j] == this->d.getm(j) || I[j] == this->d.getM(j))  {
	  mm +=  (*this)[c] ;
	  nn ++ ;
	  break ;
	}
      c++ ;
      this->d.inc(I) ;
    }
    return mm /nn ;
  }

  /**
     standard dev of the values on the boundary
  */
  _real stdBoundary() const { 
    Ivector I ;
    unsigned int c=0 ;
    _real mm = 0, MM = 0 ;
    int nn = 0 ;

    this->d.putm(I) ;
    while(c < this->d.length) {
      for (unsigned int j=0; j< this->d.n; j++)
	if (I[j] == this->d.getm(j) || I[j] == this->d.getM(j))  {
	  mm +=  (*this)[c] ;
	  MM += (*this)[c] * (*this)[c] ;
	  nn ++ ;
	  break ;
	}
      c++ ;
      this->d.inc(I) ;
    }
    mm /= nn ;
    return MM/nn - mm * mm ;
  }


  /**
     median of the values on the boundary
  */
  _real medianBoundary() const { 
    Ivector I ;
    unsigned int c=0 ;
    std::vector<NUM> tmp ;
    int k = 0 ;

    this->d.putm(I) ;
    while(c < this->d.length) {
      for (unsigned int j=0; j< this->d.n; j++)
	if (I[j] == this->d.getm(j) || I[j] == this->d.getM(j))  {
	  tmp.resize(k+1) ;
	  tmp[k] = (*this)[c] ;
	  k ++ ;
	  break ;
	}
      c++ ;
      this->d.inc(I) ;
    }
    sort(tmp.begin(), tmp.end()) ;
    return(tmp[tmp.size()/2]) ;
  }


  /**
     minimum value
  */
  _real min() const {
    _real res ;
    res = (*this)[0] ;
    for(unsigned int i=1; i<this->length(); i++)
      if ((*this)[i] < res)
	res = (*this)[i] ;
    return res ;
  }

  /**
     minimum value
  */
  _real max() const {
    _real res ;
    res = (*this)[0] ;
    for(unsigned int i=1; i<this->length(); i++)
      if ((*this)[i] > res)
	res = (*this)[i] ;
    return res ;
  }


  /**
     maximum absolute value
  */
  _real maxAbs() const {
    int i ;
    _real le = this->length(), res ;
    res = fabs((*this)[0]) ;
    for(i=1; i<le; i++)
      if (fabs((*this)[i]) > res)
	res = fabs((*this)[i]) ;
    return res ;
  }

  /**
     q quantile (censored min)
  */
  _real min(_real q) const
  {
    int i ;
    std::vector<NUM>  x ;
  
    int le = this->length() ;
    NUM res ;
    x.resize(le) ;
    for(i=0; i<le; i++)
      x[i] = (*this)[i] ;
    
    sort(x.begin(), x.end()) ;
    res = x[(int)(q*le)] ;
    return res ;
  }

  /** 
      1-q quantile (censored max)
  */
  _real max(_real q) const
  {
    int i ;
    std::vector<NUM>  x ;
    int le = this->length();
    NUM res ;
    x.resize(le) ;
    
    for(i=0; i<le; i++)
      x[i] = (*this)[i] ;
    
    sort(x.begin(), x.end()) ;
    res = x[(int)((1-q)*le)] ;
    return res ;
  }


  /** Censors between q and 1-q quantiles
   */
  void censor(_real q)
  {
    int i ;
    std::vector<NUM>  x ;
    int le = this->length();
    NUM res1, res2 ;
    x.resize(le) ;
    
    for(i=0; i<le; i++)
      x[i] = (*this)[i] ;
    
    sort(x.begin(), x.end()) ;
    res1 = x[(int)(q*le)] ;
    res2 = x[(int)((1-q)*le)] ;

    for (i=0; i<le; i++)
      if ((*this)[i] > res2)
	(*this)[i] = res2 ;
      else if ((*this)[i] < res1)
	(*this)[i] = res1 ;
  }

  /** Censors between q and 1-q quantiles
   */
  void censor(const _Vector<char> &Mask, _real q)
  {
    int i, nb ;
    std::vector<NUM>  x ;
    int le = this->length();
    NUM res1, res2 ;
    x.resize(le) ;
    _iterator I ;
    I = x.begin() ;
    
    nb = 0 ;
    for(i=0; i<le; i++)
      if (Mask[i]) {
	x[nb++] = (*this)[i] ;
      }
    x.resize(nb) ;
    
    sort(x.begin(), x.end()) ;
    res1 = x[(int)(q*nb)] ;
    res2 = x[(int)((1-q)*nb)] ;

    for (i=0; i<le; i++)
      if ((*this)[i] > res2)
	(*this)[i] = res2 ;
      else if ((*this)[i] < res1)
	(*this)[i] = res1 ;
  }

  // statistical functions
  _real quantile(_real q)
  {
    int i, ll = this->length() ;
    std::vector<NUM> tmp ;
  
    tmp.resize(ll) ;
    for(i=0; i<ll; i++)
      tmp[i] = (*this)[i] ;
  
    std::sort(tmp.begin(), tmp.end()) ;
    _real res = tmp[(int) floor(q*ll)] ;
    return res ;
  }


  NUM mean()
  {
    int i, ll = this->length() ;
    NUM res ;
    
    res = 0 ;
    for(i=0; i<ll; i++)
      res += (*this)[i] ;
    res /=ll ;

    return res ;
  }

  NUM median()
  {
    int i, ll = this->length() ;
    std::vector<NUM> tmp ;
    tmp.resize(ll) ;
    for(i=0; i<ll; i++)
      tmp[i] = (*this)[i] ;
    std::sort(tmp.begin(), tmp.end()) ;
    NUM res = 0.5*(tmp[(int) floor(0.45*ll)] + tmp[(int) floor(0.55*ll)]) ;
    return res ;
  }


  /** nIter steps of Laplacian smoothing with weight w
   */
  void laplacianSmoothing(int nIter, double w) {
    Ivector I ;
    int nb ;
    NUM S ;
    _Vector<NUM> tmp ;
    tmp.al(this->d) ;

    for (int it=0; it<nIter; it++) {
      this->d.putm(I) ;
      for (int c=0; c<(int) this->d.length; c++) {
	  S = 0 ;
	  nb = 0 ;
	  for (int k=0; k< (int) this->d.n; k++) {
	    if (I[k] < this->d.getM(k)) {
	      S += (*this)[this->d.rPos(c,k,1)] ;
	      nb ++ ;
	    }
	    if (I[k] > this->d.getm(k)) {
	      S += (*this)[this->d.rPos(c,k,-1)] ;
	      nb++ ;
	    }
	  }
	  S *= w/nb  ;
	  tmp[c] = (*this)[c] ;
	  tmp[c] *= 1 - w ;
	  tmp[c] += S ;
	  this->d.inc(I) ;
      }
      copy(tmp) ;
    }
  }


  /** nIter steps of 1-voxel erosion
   */
  void erosion(int nIter, _VectorBase<char> &Mask) {
    Ivector I ;
    bool res ;
    _VectorBase<char> tmp ;
    tmp.al(this->d) ;

    for (int it=0; it<nIter; it++) {
      this->d.putm(I) ;
      for (int c=0; c<(int) this->d.length; c++) {
	  res = 1 ;
	  for (int k=0; k<(int) this->d.n; k++) {
	    if (I[k] < this->d.getM(k)) 
	      if(Mask[this->d.rPos(c,k,1)]==0) {
		res = 0 ;
		//		break ;
	      }   
	    if (I[k] > this->d.getm(k)) 
	      if(Mask[this->d.rPos(c,k,-1)]==0){
		res = 0 ;
		//		break ;
	    }
	  }
	  tmp[c] = res ;
	  this->d.inc(I) ;
      }
      Mask.copy(tmp) ;
    }
  }

};


#endif





