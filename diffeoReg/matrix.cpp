#include "matrix.h"

static bool operator <= (std::pair<double, int> &p1, std::pair<double, int> &p2)
{
  return p1.first <= p2.first ;
}

void Matrix::copy(const Matrix &src)
{ 
  resize(src.nRows(), src.nColumns()) ;
  unsigned int i, j ;
  for (i=0; i<nRows(); i++)
    for (j=0; j<nColumns(); j++) 
      I_[i][j] = src(i, j) ;
}

void Matrix::resize(unsigned int i1, unsigned int i2) 
{
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

void Matrix::zeros(unsigned int i1, unsigned int i2) 
{
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

  for(unsigned int i=1; i<i1; i++){
    I_[i] = I_[i-1] + i2 ;
  }
} 

void Matrix::eye(unsigned int i1) 
{
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

void Matrix::Print() {
  for (unsigned int i=0; i<nRows(); i++) {
    for (unsigned int j=0; j<nColumns(); j++)
      cout << (*this)(i, j) << " ";
    cout << endl ;
  }

}

/**
   Matrix product of src1 and src2
*/
void Matrix::matProduct(const Matrix &src1, const Matrix &src2)
{
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
void Matrix::TmatProduct(const Matrix &src1, const Matrix &src2)
{
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
void Matrix::matProductT(const Matrix &src1, const Matrix &src2)
{
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


void Matrix::transpose(const Matrix &src) {
  resize(src.nColumns(), src.nRows()) ;
  for (unsigned int i=0; i<nRows(); i++)
    for (unsigned int j=0; j<nColumns() ; j++)
      I_[i][j] = src(j, i) ;
}

void Matrix::apply(const std::vector<double> &src1,  std::vector<double> &res) const
{
  res.resize(nRows()) ;
  for(unsigned int i=0; i<nRows(); i++) {
    res[i] = 0 ;
    for (unsigned int j=0; j<nColumns(); j++)
      res[i] += (*this)(i,j) + src1[j] ;
  }
}


double Matrix::trace() const {
  int minDim = nRows() ;
  if (nColumns() < nRows())
    minDim = nColumns() ;
  double res = 0 ;
  for (int k=0; k<minDim; k++)
    res += (*this)(k,k) ;
  return res ;
}


void Matrix::transpose() 
{
  Matrix tmp ;
  tmp.transpose(*this) ;
  (*this).copy(tmp) ;
}

// equivalent of matlab's 'eye'
void Matrix::idMatrix(unsigned int n)
{
  zeros(n, n) ;
  for(unsigned int i=0; i<n; i++)
    (*this)(i, i) = 1 ;
}

/** 
 uses TNT and JAMA template libraries
 Eigenvectors are rows of the V matrix
*/
void Matrix::eig(Matrix &V, std::vector<double> &dg)
{
  if (nColumns() != nRows()) {
    cerr << "eig: non square matrix"<< endl ;
    exit(1) ;
  }
  TNT::Array2D<double> eigVectors, H(nRows(), nRows()) ; 
  std::vector<std::pair<double, int> > dgTmp ;
  for (unsigned int k=0; k<nRows(); k++)
    for (unsigned int kk=0; kk<nRows(); kk++)
      H[k][kk] = I_[k][kk] ;
  TNT::Array1D<double> eigValues(nRows()) ;
  JAMA::Eigenvalue<double> eigenV(H) ;
  eigenV.getV(eigVectors) ;
  eigenV.getRealEigenvalues(eigValues) ;
  V.resize(nRows(), nRows()) ;
  dg.resize(nRows()) ;
  dgTmp.resize(nRows()) ;
  for (unsigned int k=0; k<nRows(); k++) {
    dgTmp[k].first = eigValues[k] ;
    dgTmp[k].second = k ;
  }

  std::sort(dgTmp.begin(), dgTmp.end()) ;
  
  for (unsigned int k=0; k<nRows(); k++) {
    dg[k] = dgTmp[k].first ;
    for (unsigned int kk=0; kk<nRows(); kk++)
      V.I_[k][kk] = eigVectors[dgTmp[k].second][kk];
    //      V.I_[k][kk] = eigVectors[k][dgTmp[kk].second] ;
  }
}


double integrateMatrix(const Matrix &gm, Matrix &res, int T)
{
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

  for (int t=2; t<=T; t++) {
    allAT[t].matProduct(allAT[1], allAT[t-1]) ;
  }
  res.copy(allAT[T]) ;
  return det ;
}

