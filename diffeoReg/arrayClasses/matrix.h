/**
   matrix.h
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

#ifndef _MATRIX_
#define _MATRIX_

#define TINY 1.0e-20;

#include "VectorBase.h"
//#include "jama_eig.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>


/**
a simple class for matrix manipulation
*/
class Matrix: public _Vector<double> {
 public:
  typedef _Vector<double> Vector ;
  Matrix():Vector() {} ;

  unsigned int nRows() const {return d.getM(0) + 1 ;}
  unsigned int nColumns() const {return d.getM(1) + 1;}
  double& operator ()(int i, int j) { return I_[i][j] ;}
  double& operator ()(int i, int j) const { return I_[i][j] ;}

  void copy(const Matrix &src) { 
    resize(src.nRows(), src.nColumns()) ;
    unsigned int i, j ;
    for (i=0; i<nRows(); i++)
      for (j=0; j<nColumns(); j++) 
	I_[i][j] = src(i, j) ;
  }

  void resize(unsigned int i1, unsigned int i2) {
    Ivector I1, I2 ;
    I1.resize(2) ;
    I2.resize(2) ;
    I1.zero() ;
    I2[0] = i1- 1; 
    I2[1] = i2 - 1 ;
    d.create(I1, I2) ; 
    al(d) ;

    I_.resize(i1) ;
    I_[0] = begin() ;
    
    for(unsigned int i=1; i<i1; i++)
      I_[i] = I_[i-1] + i2 ;
  } 

  void zeros(unsigned int i1, unsigned int i2) {
    Ivector I1, I2 ;
    I1.resize(2) ;
    I2.resize(2) ;
    I1.zero() ;
    I2[0] = i1- 1; 
    I2[1] = i2 - 1 ;
    d.create(I1, I2) ; 
    Vector::zeros(d) ;
    
    I_.resize(i1) ;
    I_[0] = begin() ;
    
    for(unsigned int i=1; i<i1; i++)
      I_[i] = I_[i-1] + i2 ;
  } 

  void eye(unsigned int i1) {
    Ivector I1, I2 ;
    I1.resize(2) ;
    I2.resize(2) ;
    I1.zero() ;
    I2[0] = i1- 1; 
    I2[1] = i1 - 1 ;
    d.create(I1, I2) ; 
    Vector::zeros(d) ;
    
    I_.resize(i1) ;
    I_[0] = begin() ;
    
    for(unsigned int i=1; i<i1; i++)
      I_[i] = I_[i-1] + i1 ;
    for(unsigned int i=0; i<i1; i++)
      (*this)(i,i) = 1 ;
  } 

  void Print() {
    for (unsigned int i=0; i<nRows(); i++) {
      for (unsigned int j=0; j<nColumns(); j++)
	cout << (*this)(i, j) << " ";
      cout << endl ;
    }
  }

  /**
     Matrix product of src1 and src2
  */
  void matProduct(const Matrix &src1, const Matrix &src2) {
    if (src1.nColumns() != src2.nRows()) {
      cerr << "Matrix product: dimensions are not compatible" << endl ;
      exit(1) ;
    }

    resize(src1.nRows(), src2.nColumns()) ;

    unsigned int i, j, k ;
    std::vector<double>::iterator I = begin() ;

    for (i=0; i< nRows(); i++)
      for (j=0; j<nColumns(); j++) {
	*I = 0 ;
	for (k=0; k<src1.nColumns(); k++) 
	  *I += src1(i, k) * src2(k, j) ;
	I++ ;
      }
  }


  /**
     product of src1' and src2
  */
  void TmatProduct(const Matrix &src1, const Matrix &src2) {
    if (src1.nRows() != src2.nRows()) {
      cerr << "Matrix product: dimensions are not compatible" << endl ;
      exit(1) ;
    }

    resize(src1.nColumns(), src2.nColumns()) ;

    unsigned int i, j, k ;
    std::vector<double>::iterator I = begin() ;
    
    for (i=0; i< nRows(); i++)
      for (j=0; j<nColumns(); j++) {
	*I = 0 ;
	for (k=0; k<src1.nRows(); k++) 
	  *I += src1(k, i) * src2(k, j) ;
	I++ ;
      }
  }

  /**
     product of src1 and src2'
  */
  void matProductT(const Matrix &src1, const Matrix &src2) {
    if (src1.nColumns() != src2.nColumns()) {
      cerr << "Matrix product: dimensions are not compatible" << endl ;
      exit(1) ;
    }

    //  cout << "product" << endl ;
    resize(src1.nRows(), src2.nRows()) ;

    unsigned int i, j, k ;
    std::vector<double>::iterator I = begin() ;

    for (i=0; i< nRows(); i++)
      for (j=0; j<nColumns(); j++) {
	*I = 0 ;
	for (k=0; k<src1.nColumns(); k++) 
	  *I += src1(i, k) * src2(j, k) ;
	I++ ;
      }
  }


  void transpose(const Matrix &src) {
    resize(src.nColumns(), src.nRows()) ;
    for (unsigned int i=0; i<nRows(); i++)
      for (unsigned int j=0; j<nColumns() ; j++)
	I_[i][j] = src(j, i) ;
  }

  void apply(const std::vector<double> &src1,  std::vector<double> &res) const {
    res.resize(nRows()) ;
    for(unsigned int i=0; i<nRows(); i++) {
      res[i] = 0 ;
      for (unsigned int j=0; j<nColumns(); j++)
	res[i] += (*this)(i,j) + src1[j] ;
    }
  }


  double trace() const {
    int minDim = nRows() ;
    if (nColumns() < nRows())
      minDim = nColumns() ;
    double res = 0 ;
    for (int k=0; k<minDim; k++)
      res += (*this)(k,k) ;
    return res ;
  }


  void transpose()  {
    Matrix tmp ;
    tmp.transpose(*this) ;
    (*this).copy(tmp) ;
  }

  // equivalent of matlab's 'eye'
  void idMatrix(unsigned int n) {
    zeros(n, n) ;
    for(unsigned int i=0; i<n; i++)
      (*this)(i, i) = 1 ;
  }

  /** 
      uses TNT and JAMA template libraries
      Eigenvectors are rows of the V matrix
  */
  void eigSym(Matrix &V, std::vector<double> &dg) {
    if (nColumns() != nRows()) {
      cerr << "eig: non square matrix"<< endl ;
      exit(1) ;
    }
    //    TNT::Array2D<double> eigVectors, H(nRows(), nRows()) ; 
    gsl_matrix *H = gsl_matrix_alloc(nRows(), nRows()) ; 
    //    std::vector<std::pair<double, int> > dgTmp ;
    for (unsigned int k=0; k<nRows(); k++)
      for (unsigned int kk=0; kk<nRows(); kk++)
	gsl_matrix_set(H, k, kk, I_[k][kk]) ;
    gsl_vector *eigValues= gsl_vector_alloc(nRows()) ;
    gsl_matrix *eigVectors = gsl_matrix_alloc(nRows(), nRows()) ;
    gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc (nRows());
    gsl_eigen_symmv(H, eigValues, eigVectors, w) ;
    gsl_eigen_symmv_free (w);
    gsl_eigen_symmv_sort (eigValues, eigVectors, GSL_EIGEN_SORT_ABS_DESC);
       
    //TNT::Array1D<double> eigValues(nRows()) ;
    //JAMA::Eigenvalue<double> eigenV(H) ;
    //eigenV.getV(eigVectors) ;
    //eigenV.getRealEigenvalues(eigValues) ;
    V.resize(nRows(), nRows()) ;
    dg.resize(nRows()) ;
    //dgTmp.resize(nRows()) ;

    for (unsigned int k=0; k<nRows(); k++) {
      dg[k] = gsl_vector_get(eigValues, k) ;
      for (unsigned int kk=0; kk<nRows(); kk++)
	V.I_[k][kk] = gsl_matrix_get(eigVectors, k, kk);
      //      V.I_[k][kk] = eigVectors[k][dgTmp[kk].second] ;
    }
    gsl_vector_free (eigValues);
    gsl_matrix_free(eigVectors) ;
    gsl_matrix_free(H) ;
  }


  double integrate(Matrix &res, int T) const {return integrateMatrix(*this, res, T);}

  double integrateMatrix(const Matrix &gm, Matrix &res, int T) const {
    Matrix AT, tmp ;
    unsigned int N = gm.nRows() ;
    std::vector<Matrix> allAT ;
    
    allAT.resize(T+1) ;
    AT.idMatrix(N+1) ;
    tmp.copy(gm) ;
    tmp /= T ;
    for (unsigned int k=0; k<tmp.nRows(); k++)
      for (unsigned int kk=0; kk<tmp.nColumns(); kk++)
	AT(k,kk) += tmp(k,kk) ;
    allAT[0].idMatrix(N+1) ;
    allAT[1].copy(AT) ;
    _real det = std::pow(AT.determinant(), (int) T) ;
    
    for (int t=2; t<=T; t++) 
      allAT[t].matProduct(allAT[1], allAT[t-1]) ;
    res.copy(allAT[T]) ;
    return det ;
  }


  int ludcmpm(Matrix &a,std::vector<int> &indx,double *d) const {
    unsigned int n ;
    unsigned int i,imax,j,k;
    double big,dum,sum,temp;
    std::vector<double> vv;
  

    n= a.nRows();
    vv.resize(n);
    *d=1.0;
    for (i=0;i<n;i++) {
      big=0.0;
      for (j=0;j<n;j++)
	if ((temp=fabs(a(i, j))) > big) 
	  big=temp;
      if (big == 0.0) {
	// cerr << "Singular matrix in routine LUDCMP" << endl;
	return 0;
      }
      else
	vv[i]=1.0/big;  
    }
  
  
    for (j=0;j<n;j++) {
      for (i=0;i<j;i++)  {
	sum=a(i, j);
	for (k=0;k<i;k++) 
	  sum -= a(i, k)*a(k, j);
	a(i, j)=sum;
      }
      big=0.0;
      imax = j; 
      for (i=j;i<n;i++)  {
	sum=a(i, j);
	for (k=0;k<j;k++)
	  sum -= a(i, k)*a(k, j);
	a(i, j)=sum;
	if ( (dum=vv[i]*fabs(sum)) >= big) {
	  big=dum;
	  imax=i;
	}
      }
      if (j != imax) {
	for (k=0;k<n;k++)  {
	  dum=a(imax, k);
	  a(imax, k)=a(j, k);
	  a(j, k)=dum;
	}
	*d = -(*d);
	vv[imax]=vv[j];
      }
      indx[j]=imax;
      if (a(j, j) == 0.0) {
	a(j, j)=TINY;
	cerr << "Singular matrix in routine LUDCMP" << endl;
      }
      if (j != n) {
	dum=1.0/(a(j, j));
	for (i=j+1;i<n;i++) a(i, j) *= dum;
      }
    }
    return 1;
  }


  /**
     Stores the inverse of a0 in *this
  */
  double inverse(const Matrix &a0) {
    gsl_matrix *a = gsl_matrix_alloc(a0.nRows(), a0.nColumns())  ;
    for (unsigned int i=0; i<a0.nRows(); i++)
      for(unsigned int j=0; j<a0.nColumns(); j++)
	gsl_matrix_set(a, i, j, a0(i,j)) ;
    gsl_permutation * p = gsl_permutation_alloc (a0.nRows());
    int s ;
    gsl_linalg_LU_decomp (a, p, &s);
    gsl_vector *b = gsl_vector_alloc(nRows()) ;
    gsl_vector *x = gsl_vector_alloc(nRows()) ;

    int nn=a0.nRows();
    resize(a0.nRows(), a0.nColumns()) ;
    double d_ = 1 ;
  
    for(int j=0;j<nn;j++) {
      d_ *= gsl_matrix_get(a, j, j);
      for(int i=0;i<nn;i++) gsl_vector_set(b, i, 0) ;
      gsl_vector_set(b, j, 1) ;
      gsl_linalg_LU_solve(a, p, b, x) ;
      for(int i=0;i<nn;i++) (*this)(i, j)=gsl_vector_get(x, i);
    }
    gsl_permutation_free (p);
    gsl_vector_free (b);
    gsl_vector_free (x);
    gsl_matrix_free(a) ;

    return d_ ;
  }


  double determinant() const {
    gsl_matrix *a = gsl_matrix_alloc(nRows(), nColumns())  ;
    for (unsigned int i=0; i<nRows(); i++)
      for(unsigned int j=0; j<nColumns(); j++)
	gsl_matrix_set(a, i, j, (*this)(i,j)) ;
    gsl_permutation * p = gsl_permutation_alloc (nRows());
    int s ;
    gsl_linalg_LU_decomp (a, p, &s);
    double d_ = 1 ;
  
    for(unsigned int j=0;j<nRows();j++) {
      d_ *= gsl_matrix_get(a, j, j);
    }
    gsl_permutation_free (p);
    gsl_matrix_free(a) ;

    return d_ ;   
  }



  /**
     Solves the linear equation (*this)x = y
  */
  double solve(std::vector<double> &x, const std::vector<double> &y) const {
    gsl_matrix *a = gsl_matrix_alloc(nRows(), nColumns())  ;
    for (unsigned int i=0; i<nRows(); i++)
      for(unsigned int j=0; j<nColumns(); j++)
	gsl_matrix_set(a, i, j,(*this)(i,j)) ;
    gsl_permutation * p = gsl_permutation_alloc (nRows());
    int s ;
    gsl_linalg_LU_decomp (a, p, &s);
    gsl_vector *xx = gsl_vector_alloc(nRows()) ;
    gsl_vector *yy = gsl_vector_alloc(nRows()) ;

    int nn=nRows();
    double d_ = 1 ;
  
    for(int j=0;j<nn;j++)
      d_ *= gsl_matrix_get(a, j, j);
    
    for(int i=0;i<nn;i++) 
      gsl_vector_set(yy, i, y[i]) ;
    gsl_linalg_LU_solve(a, p, yy, xx) ;
    for(int i=0;i<nn;i++) 
      x[i] = gsl_vector_get(xx, i);
    
    gsl_permutation_free (p);
    gsl_vector_free (xx);
    gsl_vector_free (yy);
    gsl_matrix_free(a) ;

    return d_ ;    
  }

 private:
  std::vector<std::vector<double>::iterator> I_ ;

} ;


inline ostream & operator << (ostream &os, const Matrix& D) {
  for (unsigned int i=0; i< D.nRows(); i++) {
    for (unsigned int j=0; j< D.nColumns(); j++)
      os << setw(8) << D(i,j) << " " ;
    os << endl ;
  }
  return os ;
}


//double integrateMatrix(const Matrix &gm, Matrix &res, int T) ;

#undef TINY

#endif
