/**
   pointSet.h
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
#ifndef _POINTSET_
#define _POINTSET_

// index class for multidimensional arrays

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <vector>

#ifndef DIM_MAX
#define DIM_MAX 5
#endif

using namespace std ;


class Point:public std::vector<double>
{
 public:
  void copy(const Point &src){
    resize(src.size()) ;
    for (unsigned int k=0; k<size(); k++)
      (*this)[k] = src[k] ;
  }
} ;
  
class PointSet: public std::vector<Point>
{
 public:
  void al(int s, int d) {resize(s); for(int k=0; k<s; k++) (*this)[k].resize(d) ; }
  void zeros(int s, int d) {resize(s); for(int k=0; k<s; k++) (*this)[k].resize(d) ; for(int k=0; k<s;k++) for(int kk=0; kk<d; kk++) (*this)[k][kk] = 0 ;}
  void copy(const PointSet &src){
    al(src.size(), src.dim()) ;
    for (unsigned int k=0; k<src.size(); k++)
      (*this)[k].copy(src[k]) ;
  }
  void get_points(char *filein, int dm) {
    ifstream ifs ;
    ifs.open(filein) ;
    char lstr[500], *pch ;
    if (ifs.fail()) {
      cerr << "impossible to open " << filein << endl ;
      exit(1) ;
    }
    string line ;
    int N = 0 ;
    getline(ifs, line) ;
    getline(ifs, line) ;
    if (line.length() > 0 && line[0] != '#') {
      strcpy(lstr, line.c_str()) ;
      pch = strtok(lstr, " \t\n\r") ;
      N = atoi(pch) ;
    }
    al(N, dm) ;
    for (int k=0; k<N; k++) {
      getline(ifs, line) ; //ignore label
      //      cout << line << endl ;
      getline(ifs, line) ;
      //      cout << line << endl ;
      //      cout << k << ": " ;
      if (line.length() > 0 && line[0] != '#') {
	strcpy(lstr, line.c_str()) ;
	pch = strtok(lstr, " \t\n\r") ;
	for (int l=0; l<dm; l++) {
	  (*this)[k][l] = atof(pch) ;
	  //	  cout << (*this)[k][l] << " " ;
	  pch = strtok(NULL, " \t\n\r") ;
	}
	//	cout << endl ;
      }
    }
    ifs.close() ;
  }
  void read(char *filein) {
    ifstream ifs ;
    ifs.open(filein) ;
    if (ifs.fail()) {
      cerr << "impossible to open " << filein << endl ;
      exit(1) ;
    }
    string line ;
    int N, dm ;
    ifs.read((char *) &N, sizeof(int)) ;
    ifs.read((char *) &dm, sizeof(int)) ;
    double *tmp = new double[N*dm];
    ifs.read((char *)tmp, N*dm*sizeof(double)) ;
    int l=0; 
    for (int k=0; k<N; k++) {
      for (int j=0; j<dm; j++)
	(*this)[k][j] = tmp[l++];
    }
    ifs.close() ;
    delete[] tmp ;
  }
  void write_points(char *fileout) const {
    ofstream ofs ;
    ofs.open(fileout) ;
    if (ofs.fail()) {
      cerr << "impossible to open " << fileout << endl ;
      exit(1) ;
    }
    int N=size() ;
    ofs << "Landmarks" << endl ;
    ofs << N << endl;
    for (int k=0; k<N; k++) {
      ofs << k<< endl ;
      for (int j=0; j<dim(); j++)
	ofs << (*this)[k][j] << " ";
      ofs << endl ;
    }
    ofs.close() ;
  }
  void write(char *fileout) const {
    ofstream ofs ;
    ofs.open(fileout) ;
    if (ofs.fail()) {
      cerr << "impossible to open " << fileout << endl ;
      exit(1) ;
    }
    int N=size(), dm=dim() ;
    ofs.write((char *)&N, sizeof(int)) ;
    ofs.write((char *)&dm, sizeof(int)) ;
    double *tmp = new double[N*dm];
    int l=0; 
    for (int k=0; k<N; k++) {
      for (int j=0; j<dm; j++)
	tmp[l++] = (*this)[k][j] ;
    }
    ofs.write((char *)tmp, N*dm*sizeof(double)) ;
    ofs.close() ;
    delete[] tmp ;
  }

  void affineInterp(PointSet &res, const double mat[DIM_MAX][DIM_MAX+1]) const {
    int N = dim() ;
    res.al(size(), dim()) ;
    for (unsigned int i=0; i< size(); i++) {
      for (int j=0; j< N; j++) {
	res[i][j] = mat[j][N] ;
	for(int jj=0; jj< N; jj++)
	  res[i][j] += mat[j][jj] * (*this)[i][jj] ;
      }
    }
  }

  double norm2() {
    double res = 0 ;
    for(int k=0; k<(int) size(); k++)
      for (int kk=0; kk<dim(); kk++)
	res += (*this)[k][kk] * (*this)[k][kk] ;
    return res ;
  }

  double sum() {
    double res = 0 ;
    for(int k=0; k<(int) size(); k++)
      for (int kk=0; kk<dim(); kk++)
	res += (*this)[k][kk] ;
    return res ;
  }

  void operator +=( const double x) {
    for(int k=0; k<(int) size(); k++)
      for (int kk=0; kk<dim(); kk++)
	(*this)[k][kk] += x ;
  }
  void operator -=( const double x) {
    for(int k=0; k<(int) size(); k++)
      for (int kk=0; kk<dim(); kk++)
	(*this)[k][kk] -= x ;
  }
  void operator *=( const double x) {
    for(int k=0; k<(int) size(); k++)
      for (int kk=0; kk<dim(); kk++)
	(*this)[k][kk] *= x ;
  }
  void operator /=( const double x) {
    for(int k=0; k<(int) size(); k++)
      for (int kk=0; kk<dim(); kk++)
	(*this)[k][kk] /= x ;
  }

  void operator +=( const PointSet &x) {
    for(int k=0; k<(int) size(); k++)
      for (int kk=0; kk<dim(); kk++)
	(*this)[k][kk] += x[k][kk] ;
  }

  void operator -=( const PointSet &x) {
    for(int k=0; k<(int) size(); k++)
      for (int kk=0; kk<dim(); kk++)
	(*this)[k][kk] -= x[k][kk] ;
  }

  void operator *=( const PointSet &x) {
    for(int k=0; k<(int) size(); k++)
      for (int kk=0; kk<dim(); kk++)
	(*this)[k][kk] *= x[k][kk] ;
  }

  void operator /=( const PointSet &x) {
    for(int k=0; k<(int) size(); k++)
      for (int kk=0; kk<dim(); kk++)
	(*this)[k][kk] /= x[k][kk] ;
  }


  int dim() const {return (size()>0)?(*this)[0].size():0;}
} ;

class MatSet {
public:
  MatSet() {_s=0; _d=0;}
  MatSet(int s, int d){ resize(s,d) ;}
  void resize(int s, int d){
    _s=s; 
    _d=d;
    _t.resize(s*s*d*d) ;
    _p.resize(s) ;
    int j = 0 ;
    for(int k=0; k<s; k++) {
      _p[k].resize(d) ;
      for(int kk=0; kk<d; kk++) {
	_p[k][kk].resize(s) ;
	for(int l=0; l<s; l++) {
	  _p[k][kk][l] = _t.begin() + j ;
	  j += d ;
	}
      }
    }
  }

  double operator()(int k, int kk, int l, int ll) const {
    return _p[k][kk][l][ll] ;
  }
  double& operator()(int k, int kk, int l, int ll) {
    return _p[k][kk][l][ll] ;
  }

  int size() const {return _s;}
  int dim() const {return _d;}
  void copy(const MatSet &src) {
    resize(src.size(), src.dim()) ;
    for(unsigned int k=0; k< _t.size(); k++)
      _t[k] = src._t[k] ;
  }

  void zeros(int s, int d) {
    resize(s,d) ;
    for(unsigned int k=0; k< _t.size(); k++)
      _t[k] = 0 ;
  }

  void eye(int s, int d) {
    zeros(s,d) ;
    for (int k=0; k<s; k++)
      for(int kk=0; kk<d; kk++)
	_p[k][kk][k][kk] = 1 ;
  }


private:
  int _s, _d ;
  typedef std::vector<double>::iterator iter ;
  std::vector<double> _t ;
  std::vector< std::vector< std::vector<iter> > > _p ;
} ;

struct PointSetScp{
  double operator()(const PointSet &x, const PointSet &y) const {
    double res = 0 ;
    for(unsigned int k=0; k<x.size() ; k++)
      for(int kk=0; kk<x.dim(); kk++)
	res += x[k][kk] * y[k][kk] ;
    return res ;
  }
};


inline ostream & operator << (ostream &os, const PointSet& D) {
  for (unsigned int i=0; i< D.size(); i++) {
    for (int j=0; j< D.dim(); j++)
      os << D[i][j] << " " ;
    os << endl ;
  }
  return os ;
}
typedef PointSet Tangent ;

#ifndef DIM_MAX
#define DIM_MAX 5
#endif

void affineInterp(const PointSet &src, PointSet &res, const double mat[DIM_MAX][DIM_MAX+1]) ;




  
#endif
