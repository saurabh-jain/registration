/**
   tensorField.h
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
#ifndef _TENSORFIELD_
#define _TENSORFIELD_
#include "Vector.h"
#include "matrix.h"

inline int _lI(int i, int j) {if (i==0) return j ; else if (j==0) return i ; else return i + j + 1;} 

class SymmetricTensor{
public:
  double & operator[](int k) {return _t[k];}
  double operator[](int k) const {return _t[k];}
  double& operator()(int i, int j) {if (i==0) return _t[j] ; else if (j==0) return _t[i] ; else return _t[i + j + 1];}
  double operator()(int i, int j) const {if (i==0) return _t[j] ; else if (j==0) return _t[i] ; else return _t[i + j + 1];}

  void operator +=(SymmetricTensor &t1){
    for (int k=0; k<6; k++) _t[k] += t1[k] ;
  }
  void operator -=(SymmetricTensor &t1){
    for (int k=0; k<6; k++) _t[k] -= t1[k] ;
  }
  void operator +=(double t1){
    for (int k=0; k<6; k++) _t[k] += t1 ;
  }
  void operator *=(double t1){
    for (int k=0; k<6; k++) _t[k] *= t1 ;
  }
  void operator -=(double t1){
    for (int k=0; k<6; k++) _t[k] -= t1 ;
  }
  void operator /=(double t1){
    for (int k=0; k<6; k++) _t[k] /= t1 ;
  }

  double trace() { return _t[0] + _t[3] + _t[5] ;}

  double invert() {return invert(0);}
  double invert(double z) {
    double det ;
    Matrix Mat, invMat ;
    Mat.resize(3,3)  ;
    invMat.resize(3,3) ;

    for(int i=0; i<3; i++) {
      for(int j=0; j<3; j++)
	Mat(i,j) = (*this)(i,j) ;
      Mat(i,i) += z ;
    }

    //    Mat /= 1e-10 + tr ;

    det = invMat.inverse(Mat) ;
    //    invMat *= 1e-10 + tr ;

    //    if (tr > 1e-5 && det < 1e-10) 
    //      cout << "non invertible " << Mat << endl ;

    for(int i=0; i<3; i++)
      for(int j=i; j<3; j++)
	(*this)(i,j) = invMat(i,j) ;

    return det ;
  }

  void eig(Matrix &V, std::vector<double> &dg) {
    Matrix Mat ;
    Mat.resize(3,3)  ;

    for(int i=0; i<3; i++) {
      for(int j=0; j<3; j++)
	Mat(i,j) = (*this)(i,j) ;
    }

    Mat.eigSym(V, dg) ;
  }

  double fa() {
    Matrix Mat, Mat2 ;
    Mat.resize(3,3)  ;

    for(int i=0; i<3; i++)
      for(int j=0; j<3; j++)
	Mat(i,j) = (*this)(i,j) ;

    Mat2.matProduct(Mat,Mat) ;

    double f ;
    f = sqrt(1.5) * sqrt((Mat2.trace() - Mat.trace()*Mat.trace()/3)/(1e-10 + Mat2.trace())) ;

/*     if (f > 0.1) { */
/*       for (int i=0; i< 6; i++) */
/* 	cout << (*this)[i] << " " ; */
/*       cout << endl ; */
/*       cout << Mat << endl ; */
/*       cout << Mat2 << endl ; */
/*       cout << f << endl ; */
/*     } */

    return f ;
  }

  SymmetricTensor& operator =(const SymmetricTensor &t1)
    {
      for (int k=0; k<6; k++) _t[k] = t1[k] ;
      return (*this) ;
    }

  SymmetricTensor& operator =(const double t1)
  {
    for (int k=0; k<6; k++) _t[k] = t1 ;
    return (*this) ;
  }


  
  
private:
  double _t[6] ;
} ;

inline ostream & operator << (ostream &os, const SymmetricTensor& D) {
  for (int i=0; i< 6; i++)
    os << D[i] << " " ;
  //  os << endl ;
  return os ;
}

class ChristoffelSymbols{
public:
  double * operator[](int k) {return _t[k];}
  const double * operator[](int k) const {return _t[k];}
  ChristoffelSymbols& operator =(const ChristoffelSymbols &t1)
  {
    for (int k=0; k<6; k++) for(int l=0; l<3; l++) _t[k][l] = t1[k][l] ;
    return (*this) ;
  }
  ChristoffelSymbols& operator =(const double t1)
  {
    for (int k=0; k<6; k++) for(int l=0; l<3; l++) _t[k][l] = t1 ;
    return (*this) ;
  }
private:
  double _t[6][3] ;
} ;




class SymmetricTensorField: public _Vector<SymmetricTensor>{
public:

  //  using _VectorBase<SymmetricTensor>::copy ;
  typedef _Vector<ChristoffelSymbols> c_symbol ;
  
  void copy(const SymmetricTensorField & src)
  {
    al(src.d) ;
    for (int k=0; k < (int) d.length; k++)
      for (int j=0; j< (int) 6; j++)
	(*this)[k][j] = src[k][j] ;
  }
  
  double& operator()(int c, int i, int j) { return (*this)[c](i,j) ;}
  double operator()(int c, int i, int j) const{ return (*this)[c](i,j) ;}

  double& cs(int c, int i, int j, int k) { if(i<j) return _cs[c][_lI(i,j)][k]; else return _cs[c][_lI(j,i)][k];}
  double cs(int c, int i, int j, int k) const { if(i<j) return _cs[c][_lI(i,j)][k]; else return _cs[c][_lI(j,i)][k];}
  double& inv(int c, int i, int j) {return _inv[c](i,j); }
  double inv(int c, int i, int j) const {return _inv[c](i,j); }

  void inverseTensor() {inverseTensor(1e-10);}
  void inverseTensor(double z) {
    _inv.al(d) ;
    _det.al(d) ;
    for (int c=0; c<(int) d.length; c++) {
      _inv[c] = (*this)[c] ;
      _det[c] = _inv[c].invert(z) ;
//        if (_det[c] < 1e-31 && (*this)[c].trace() > 0.00001) {
//  	Ivector I ;
//  	d.fromPosition(I,c) ;
//  	cout << "tensor not invertible " << I[0] << " " << I[1] << " " << I[2] << " " << (*this)[c] << " " << (*this)[c].trace() << endl ;
//        }
    } 
  }

  void computeTrace(Vector &res) {
    res.al(d) ;
    for(int c=0; c<(int) d.length; c++)
      res[c] = (*this)[c].trace() + 1e-7 ;
  }

  /**
     compute first eigenvector relative to the modulus of the eigenvalue
  */
  void computeFirstEigenvector(VectorMap &res) {
    res.al(d) ;
    Matrix V ;
    std::vector<double> dg ;
    for(int c=0; c<(int) d.length; c++) {
      (*this)[c].eig(V, dg)  ;
      int pick = 2 ;
      if ((abs(dg[0]) > abs(dg[1])) && (abs(dg[0]) > abs(dg[2])))
	pick = 0 ;
      else if ((abs(dg[1]) > abs(dg[0])) && (abs(dg[1]) > abs(dg[2])))
	pick = 1 ;

      for (int k=0; k< (int) res.size(); k++)
	res[k][c] = V(k, pick) ;
    }
  }

  void normalizeByTrace() {
    for(int c=0; c<(int) d.length; c++)
      (*this)[c] /= (*this)[c].trace() ;
  }

  void computeChristoffelSymbols(vector<double> &resol)
  {
    _cs.zeros(d) ;
    Ivector I ;
    d.putm(I) ;
    
    for (int c=0; c< (int) d.length; c++) {
      if (Mask[c]) {
	for (int i=0; i<3; i++)
	  for (int j=i; j<3; j++){
	    double tmp[3] ;
	    for (int k=0; k<3; k++) {
	      tmp[k] = 0 ;
	      if (I[i] > d.getm(i) && I[i] < d.getM(i)) 
		tmp[k] += ((*this)(d.rPos(c,i,1), j, k) - (*this)(d.rPos(c,i,-1), j, k))/resol[i] ;
	      if (I[j] > d.getm(j) && I[j] < d.getM(j)) 
		tmp[k] += ((*this)(d.rPos(c,j,1), i, k) - (*this)(d.rPos(c,j,-1), i, k))/resol[j]  ;
	      if (I[k] > d.getm(k) && I[k] < d.getM(k)) 
		tmp[k] -= ((*this)(d.rPos(c,k,1), i, j) - (*this)(d.rPos(c,k,-1), i, j))/resol[k] ;
	      tmp[k] /= 4 ;
	    }
	    for (int k=0; k<3; k++) {
	      cs(c,i,j,k) = 0 ;
	      for (int l=0; l<3; l++)
		cs(c,i,j,k) += inv(c,k,l) *tmp[l] ;
	    }
	  }
	}
      d.inc(I);
    }
  }

  void pruneOutliers() {
    Ivector I ;
    double q2 = _det.max(.001), q1 = 0 ;
    SymmetricTensor S ;
    int nb =0, nOut, it = 0  ;

    cout << "q2 = " << q2 << endl ;

    do {
      nOut = 0 ;
      d.putm(I) ;
      for (int c=0; c<(int) d.length; c++) {
	if ((_det[c] < q1 && _det[c] > 1e-10)  || (_det[c] > q2)) {
	  nOut ++ ;
	  S = 0 ;
	  nb = 0 ;
	  for (int k=0; k<3; k++) {
	    if (I[k] < d.getM(k)) {
	      S += (*this)[d.rPos(c,k,1)] ;
	      nb ++ ;
	    }
	    if (I[k] > d.getm(k)) {
	      S += (*this)[d.rPos(c,k,-1)] ;
	      nb++ ;
	    }
	  }
	  S /= nb  ;
	  (*this)[c] = S ;
	  _inv[c] = S ;
	  _det[c] = _inv[c].invert() ;
	}
	d.inc(I) ;
      }
      cout << "pruning iteration: " << it++ << "  nOut = " << nOut << endl ;
    }
    while(nOut > 0 && it < 100) ;
  }

  void smoothTensor(int nIter, double w) {
    laplacianSmoothing(nIter, w) ;
  }

  void censorTensor(int nIter) {
    Mask.zeros(d) ;
    for (int c=0; c< (int) d.length; c++) 
      if (_det[c] > 1e-10) 
	Mask[c] = 1 ;
    cout << (int) Mask.sum() << " selectionnes"<< endl;
    erosion(nIter, Mask) ;
    cout << (int) Mask.sum() << " selectionnes"<< endl;
  }


  void computeScalarCurvature(Vector & res, vector<double> &resol) {
    res .zeros(d) ;
    Ivector I ;
    d.putm(I) ;

    for (int c=0; c< (int) d.length; c++) {
      if (Mask[c]) {
	for (int i=0; i<3; i++)
	  for (int j=0; j<3; j++)
	    for (int k=0; k<3; k++) {
	      if (I[k] > d.getm(k) && I[k] < d.getM(k)) 
		res[c] += inv(c,i,j) * (cs(d.rPos(c,k,1), i, j, k) - cs(d.rPos(c,k,-1), i, j, k))/(2*resol[k]) ;
	      if (I[j] > d.getm(j) && I[j] < d.getM(j)) 
		res[c] -= inv(c,i,j) * (cs(d.rPos(c,j,1), i, k, k) - cs(d.rPos(c,j,-1), i, k, k))/(2*resol[j]) ;
	      for (int l=0; l< 3; l++)
		res[c] += inv(c,i,j) * (cs(c,i,j,k)*cs(c,k,l,l) - cs(c,i,k,l)*cs(c,j,l,k)) ;
	    }
      }
      d.inc(I);
    }
  }

  void computeRicciCurvature(SymmetricTensorField & res, vector<double> &resol) {
    res .zeros(d) ;
    Ivector I ;
    d.putm(I) ;

    for (int c=0; c< (int) d.length; c++) {
      if (Mask[c]) {
	for (int i=0; i<3; i++)
	  for (int j=i; j<3; j++)
	    for (int k=0; k<3; k++) {
	      if (I[k] > d.getm(k) && I[k] < d.getM(k)) 
		res(c,i,j) += (cs(d.rPos(c,k,1), i, j, k) - cs(d.rPos(c,k,-1), i, j, k))/(2*resol[k]) ;
	      if (I[j] > d.getm(j) && I[j] < d.getM(j)) 
		res(c,i,j) -= (cs(d.rPos(c,j,1), i, k, k) - cs(d.rPos(c,j,-1), i, k, k))/(2*resol[j]) ;
	      for (int l=0; l< 3; l++)
		res(c,i,j) += (cs(c,i,j,k)*cs(c,k,l,l) - cs(c,i,k,l)*cs(c,j,l,k)) ;
	    }
      }
      d.inc(I);
    }
  }

  Vector &determinant() {return _det;}

  void computeFractionalAnisotropy(Vector & res) {
    res.zeros(d) ;
    for (int c=0; c< (int) d.length; c++)
      res[c] = (*this)[c].fa() ;
  }

  void readTensorField(char *fileIn) 
  {
    (*this).read(fileIn) ;
    (*this)*=1000 ;
    for (int c=0; c< (int) d.length; c++) 
      for (int k=0; k<2; k++)
	(*this)(c,k,k) += 1e-10 ;
   //  ifstream ifs(fileIn) ;

//     if (ifs.fail()) {
//       cerr << "Cannot open " << fileIn << endl ;
//       exit(1) ;
    // }
  }

  void swapMetric() {
    SymmetricTensor tmp ;
    for (int c=0; c< (int) d.length; c++) {
      tmp = _inv[c] ;
      _inv[c] = (*this)[c] ;
      (*this)[c] = tmp ;
    }
  }

  _Vector<char> Mask ;
private:
  c_symbol _cs ;
  _Vector<SymmetricTensor> _inv ;
  Vector _det ;


};




#endif
