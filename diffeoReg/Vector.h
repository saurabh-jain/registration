/**
   Vector.h
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

#ifndef _VECTOR_
#define _VECTOR_
// Multi dimensional array
#include "VectorBase.h"
#include "matrix.h"
#include "byteSwap.h"
//#include "AnalyzeImage.h"
#include <algorithm>
#include <Magick++.h>
#include "pointSet.h"
using namespace Magick ;

class Vector: public _Vector<_real>  
{
 public:
  using _VectorBase<_real>::operator =;

  /**
     reads from an image
  */
  void get_image(const char *file, int dim)
  {
    _Vector<_real> tmp1 ;

    if (dim == 1) {
      ifstream ifs(file) ;
      if (ifs.fail()) {
	cerr << "cannot open " << file << endl ;
	exit(1) ;
      }
      int nb ;
      ifs >> nb ;
      Ivector MIN, MAX ;
      //      unsigned int i ;
      MIN.resize(1) ;
      MIN.zero() ;
      MAX.resize(1) ;
      MAX[0] = nb-1 ;
      Domain D(MIN, MAX) ;
      zeros(D) ;
      for(int ii = 0 ; ii<nb; ii++)
	ifs >> (*this)[ii] ;
      ifs.close() ;
    }
    else if (dim==2) {
      Magick::Image img0;
      try{
	img0.read(file) ;
      }
      catch(Magick::WarningCoder &warning) {
	cerr << "Coder Warning: " << warning.what() << endl;
      }
      int nbl = img0.rows(), nbc = img0.columns() ;
      Magick::PixelPacket *pixels = img0.getPixels(0, 0, nbc, nbl);
      Ivector MIN, MAX ;
      unsigned int i ;
      MIN.resize(2) ;
      MIN.zero() ;
      MAX.resize(2) ;
      MAX[0] = nbl-1 ;
      MAX[1] = nbc-1 ;
      Domain D(MIN, MAX) ;
      zeros(D) ;
      i = 0 ;
      for(int ii = 0 ; ii<nbl; ii++) 
	for (int jj=0; jj<nbc; jj++) {
	  Magick::Color u = pixels[nbc * ii + jj] ; //img0.pixelColor(jj,ii) ;
	  (*this)[i++] = (_real) 255.0*(u.redQuantum()+u.greenQuantum()+u.blueQuantum())/(3.0*MaxRGB) ;
	}
      // cout << max() << " " << min() << endl ;
    }
    else if (dim == 3) {
      loadAnalyze(file) ;
      //      AnalyzeImage<_real> AI0(file) ;
      //      copy(*(AI0.getAnalyzeImage())) ;
    }
    else {
      cerr << "get_image: unsupported input dimensions" << endl ;
      exit(1) ;
    }
  }

  void loadAnalyze(const char *filename0) {
    char filename[256] ;
    cout << "trying " << filename0 << endl ;
    strncpy(filename, filename0, strlen(filename0) - 4) ;
    filename[strlen(filename0) - 4] = '\0' ;
    strcat(filename, ".hdr") ;
    ifstream ifs ;
    ifs.open(filename) ;
    if (ifs.fail()) {
      cerr << " cannot open " << filename << endl;
      exit(1) ;
    }
    int x ;
    char foo ;
    bool bs = false ;
    // cout << sizeof(short) << " " << sizeof(int) << endl ;
    ifs.read((char *) &x, sizeof(int)) ;
    if (x != 348)
      bs = true ;
    for (int k=0; k<38; k++)
      ifs.read(&foo, 1) ;
    //    ifs.seekg(ios::cur + 38) ;
    //x = fread(u, 1, 'int32') ; 
    //fread(u, 28, 'int8') ;
    //fread(u, 1, 'int32') ;
    //fread(u, 1, 'int16') ;
    //fread(u, 2, 'int8') ;
    short sz[3], datatype ;
    ifs.read((char *) sz, 3*sizeof(short));
    if (bs)
      for (int k=0; k<3; k++)
	byteSwap((char *) (sz + k), sizeof(short)) ;

    // cout << sz[0] << " " << sz[1] << " " << sz[2] << endl ;
    for (int k=0; k<22; k++)
      ifs.read(&foo, 1) ;
    //    ifs.seekg(ios::cur + 22) ;
    ifs.read((char *) &datatype, sizeof(short)) ;
    if (bs)
      byteSwap((char *) &datatype, sizeof(short)) ;

    cout << "Datatype: " << datatype << endl ;
    ifs.close() ;

    strcpy(filename, "\0") ;
    strncpy(filename, filename0, strlen(filename0) - 4) ;
    filename[strlen(filename0) - 4] = '\0' ;
    // cout << filename << endl ;
    strcat(filename, ".img") ;
    ifs.open(filename) ;
    if (ifs.fail()) {
      cerr << " cannot open " << filename << endl;
      exit(1) ;
    }

    int nbVox = sz[0]*sz[1]*sz[2] ;
    Ivector I;
    I.resize(3) ;
    for (int k=0; k<3; k++) I[k]= sz[k] - 1;
    al(I) ;
    Ivector II, MIN, MAX ;
    d.putm(MIN) ;
    d.putM(MAX) ;
 
    II.resize(3) ;

    if (datatype == 2) {
      unsigned char *data = new unsigned char[nbVox] ; 
      ifs.read((char *) data, nbVox) ;
      //      for (int k=0; k<nbVox; k++) 
	//	(*this)[k] = (double) data[k] ;
      int k= 0;
     for (int k0=MIN[2]; k0 <= MAX[2] ; k0++) 
      for (int k1=MIN[1]; k1 <= MAX[1] ; k1++) 
	for (int k2=MIN[0]; k2 <= MAX[0] ; k2++) {
	    II[2] = k0 ;
	    II[1] = k1;
	    II[0] = k2 ;
	    (*this)[II] = data[k++] ;
	}
      delete[] data ;
      //      delete[] data ;
    } else if (datatype == 4) {
      short *data = new short[nbVox] ; 
      ifs.read((char *) data, nbVox*sizeof(short)) ;
      for (int k=0; k<nbVox; k++) {
	if (bs)
	  byteSwap((char *) (data + k), sizeof(short)) ;
	//	(*this)[k] = data[k] ;
      }
      int k= 0;
     for (int k0=MIN[2]; k0 <= MAX[2] ; k0++) 
      for (int k1=MIN[1]; k1 <= MAX[1] ; k1++) 
	for (int k2=MIN[0]; k2 <= MAX[0] ; k2++) {
	    II[2] = k0 ;
	    II[1] = k1;
	    II[0] = k2 ;
	    (*this)[II] = data[k++] ;
	}
      delete[] data ;
      //      delete[] data ;
    }
    else if (datatype == 8) {
      int *data = new int[nbVox] ; 
      ifs.read((char *) data, nbVox*sizeof(int)) ;
      for (int k=0; k<nbVox; k++) {
	if (bs)
	  byteSwap((char *) (data + k), sizeof(int)) ;
	//	(*this)[k] = data[k] ;
      }
      int k=0 ;
     for (int k0=MIN[2]; k0 <= MAX[2] ; k0++) 
      for (int k1=MIN[1]; k1 <= MAX[1] ; k1++) 
	for (int k2=MIN[0]; k2 <= MAX[0] ; k2++) {
	    II[2] = k0 ;
	    II[1] = k1;
	    II[0] = k2 ;
	    (*this)[II] = data[k++] ;
	}
      delete[] data ;
      //      delete[] data ;
    }
    else if (datatype == 16) {
      float *data = new float[nbVox] ; 
      ifs.read((char *) data, nbVox*sizeof(float)) ;
      for (int k=0; k<nbVox; k++) {
	if (bs)
	  byteSwap((char *) (data + k), sizeof(float)) ;
	//	(*this)[k] = data[k] ;
      }
      //      delete[] data ;
      int k=0 ;
     for (int k0=MIN[2]; k0 <= MAX[2] ; k0++) 
      for (int k1=MIN[1]; k1 <= MAX[1] ; k1++) 
	for (int k2=MIN[0]; k2 <= MAX[0] ; k2++) {
	    II[2] = k0 ;
	    II[1] = k1;
	    II[0] = k2 ;
	    (*this)[II] = data[k++] ;
	}
      delete[] data ;
    }
    else if (datatype == 64) {
      double *data = new double[nbVox] ; 
      ifs.read((char *) data, nbVox*sizeof(double)) ;
      for (int k=0; k<nbVox; k++) {
	if (bs)
	  byteSwap((char *) (data + k), sizeof(double)) ;
	//	(*this)[k] = data[k] ;
      }
      //      delete[] data ;
      int k=0 ;
     for (int k0=MIN[2]; k0 <= MAX[2] ; k0++) 
      for (int k1=MIN[1]; k1 <= MAX[1] ; k1++) 
	for (int k2=MIN[0]; k2 <= MAX[0] ; k2++) {
	    II[2] = k0 ;
	    II[1] = k1;
	    II[0] = k2 ;
	    (*this)[II] = data[k++] ;
	}
      delete[] data ;
    }
    else {
      cerr << "unknown data type: " << datatype << endl ;
      exit(1) ;
    }

    

    // cout << min() << " " << max() << endl ;
    ifs.close() ;
  }

  void saveAnalyze(const char *filename0) const {
    // cout << filename0 << endl ;
    // cout << min() << " " << max() << endl ;
    char filename[256] ;
    strcpy(filename, "\0") ;
    strncpy(filename, filename0, strlen(filename0) - 4) ;
    filename[strlen(filename0) - 4] = '\0' ;
    strcat(filename, ".hdr") ;
    ofstream ofs ;
    ofs.open(filename) ;
    if (ofs.fail()) {
      cerr << " cannot save in " << filename << endl;
      exit(1) ;
    }
    int x=348 ;
    char foo8=0 ;
    ofs.write((char *) &x, sizeof(int)) ;
    for (int k=0; k<28; k++) ofs.write(&foo8, 1) ;
    int foo32 = 16384 ;
    short foo16 ;
    ofs.write((char *) &foo32, sizeof(int)) ;
    for (int k=0; k<2; k++) ofs.write(&foo8, 1) ;
    foo8 = 'r' ;
    ofs.write(&foo8, 1) ;
    foo8=0 ;
    ofs.write(&foo8, 1) ;
    foo16 = 4 ;
    ofs.write((char *) &foo16, sizeof(short)) ;

    short sz[3], datatype = 64 ;
    for(int k=0; k<3; k++) sz[k] = d.getM(k) - d.getm(k) + 1 ;
    ofs.write((char *) sz, 3*sizeof(short));
    foo16 = 1 ;
    ofs.write((char *) &foo16, sizeof(short)) ;

    for (int k=0; k<20; k++) ofs.write(&foo8, 1) ;
    ofs.write((char *) &datatype, sizeof(short)) ;
    for (int k=0; k<276; k++) ofs.write(&foo8, 1) ;
    ofs.close() ;

    strcpy(filename, "\0") ;
    strncpy(filename, filename0, strlen(filename0) - 4) ;
    filename[strlen(filename0) - 4] = '\0' ;

    strcat(filename, ".img") ;
    ofs.open(filename) ;
    if (ofs.fail()) {
      cerr << " cannot save in " << filename << endl;
      exit(1) ;
    }


    double *data = new double[size()] ; 
    Ivector II, MIN, MAX ;
    d.putm(MIN) ;
    d.putM(MAX) ;
    int k=0 ;
    II.resize(3) ;

     for (int k0=MIN[2]; k0 <= MAX[2] ; k0++) 
      for (int k1=MIN[1]; k1 <= MAX[1] ; k1++) 
	for (int k2=MIN[0]; k2 <= MAX[0] ; k2++) {
	    II[2] = k0 ;
	    II[1] = k1;
	    II[0] = k2 ;
	    data[k++] = (*this)[II] ;
	}
    

    //    for (unsigned int k=0; k<size(); k++) data[k] = (*this)[k] ;
    ofs.write((char *) data, size()*sizeof(double)) ;
    delete[] data ;
    ofs.close() ;
  }



  /** save in char file with 0 -> 127
   */
  void writeZeroCentered(char *file)
  {
    _real m = min(), M = max() ;
    Vector Z ;
    Z.copy(*this) ;
  
    if (fabs(m) > fabs(M)) {
      Z *= 127 / fabs(m) ;
      Z += 128 ;
    }
    else {
      Z *= 127 / fabs(M) ;
      Z += 128 ;
    }
    Z.write_image(file) ;
  }

  /**
     save in vtk scalar format (3D)
  */
  void write_imageVTK(char *file)
  {
    if (d.n != 3) {
      write_image(file) ;
    }

    if (d.n == 3) {
      ofstream ofs ;
      char path[255] ;
      sprintf(path, "%s.vtk", file) ;
      ofs.open(path) ;
      if (ofs.fail()) {
	cerr << "Unable to open " << file << ".vtk in write_imageVTK" << endl ;
	exit(1) ;
      }
      
      ofs << "# vtk DataFile Version 3.0" << endl ;
      ofs << "lddmm 8 0 0 " << d.getM(2) << " " << d.getM(2) << " 0 0 " << d.getM(1) << " " << d.getM(1) << " 0 0 "
	  << d.getM(0) << " " << d.getM(0) << endl ;
      ofs << "ASCII" << endl << "DATASET STRUCTURED_POINTS" << endl << "DIMENSIONS " << d.getM(0)+1 << " " 
	  << d.getM(1)+1 << " " << d.getM(2) + 1 <<  endl;
      ofs << "SPACING 1 1 1" << endl << "ORIGIN 0 0 0" << endl << "POINT_DATA " << d.length << endl ;
      ofs << "SCALARS Scalars_ _real" << endl << "LOOKUP_TABLE default" << endl ;
      for (unsigned int i=0; i<length(); i++)
	ofs << (*this)[i] << " " ;
    }
  }
  
  /**
     saves in image format (png in 2D and analyze in 3D)
  */
  void write_image(const char *file) const
  {
    if (d.n != 1 && d.n != 2 && d.n != 3) {
      cerr << "writing only in 1d, 2d or 3d. Dimension = " << d.n << endl ;
      return ;
    }

    if (d.n == 1) {
      char path[255] ;
      sprintf(path, "%s.txt", file) ;
      ofstream ofs(path) ;
      if (ofs.fail()) {
	cerr << "cannot open " << file << endl ;
	exit(1) ;
      }
      int nb = d.length ;
      ofs << d.length << endl;
      for(int ii = 0 ; ii<nb; ii++)
	ofs << (*this)[ii] << endl;
      ofs.close() ;
    }

    if (d.n == 2) {
      ofstream ofs ;
      char path[255] ;
      sprintf(path, "%s.png", file) ;
      ofs.open(path) ;
      if (ofs.fail()) {
	cerr << "Unable to open " << file << ".png in write_image" << endl ;
	exit(1) ;
      }
      ofs.close() ;
      
      Magick::Image res( Magick::Geometry(d.getM(1)-d.getm(1)+1, d.getM(0)-d.getm(0)+1), "white") ;
      int i = 0 ;
      for (unsigned int ii=0; ii<res.rows(); ii++)
	for (unsigned int jj=0; jj<res.columns(); jj++) {
	  unsigned int uc ;
	  if ((*this)[i] < 0)
	    uc = 0 ;
	  else if ((*this)[i] > 255)
	    uc = 255 ;
	  else 
	    uc = (unsigned int) (*this)[i] ;

	  uc = (unsigned int) round((double) MaxRGB * uc/255.0) ;
	  res.pixelColor(jj,ii,Magick::Color(uc,uc,uc,uc)) ;
	  i++ ;
      }
      res.write(path) ;
    }
    
    if (d.n == 3) {
      char path[255] ;
      sprintf(path, "%s.hdr", file) ;
      saveAnalyze(path) ;
/*       _Vector<_real> tmp0 ; */
/*       tmp0.copy(*this) ; */
/*       AnalyzeImage<_real> Res ; */
/*       Array3D<_real> tmp ; */
/*       tmp0.writeInArray3D(tmp) ; */
/*       Res = tmp ; */
/*       Res.save(file) ; */
    }
  }


  /** 
      saves in image format after rescaling
  */
  void write_imagesc(const char * file) const
  {
    _real mm=min(), MM = max() ;
    Vector tmp ;

    tmp.copy(*this) ;
    tmp -= mm ;
    if (abs(MM-mm) > 0.000001)
      tmp *= 255/(MM-mm) ;
    
    tmp.write_image(file) ;
  }

};


/**
   Vector fields and diffeomorphisms
*/
template <class NUM> class _VectorMap : public std::vector<_Vector<NUM> >
{
public:
  typedef typename  std::vector<NUM>::iterator _iterator ;
  typedef typename  std::vector<NUM>::const_iterator c_iterator ;
  Domain d ;
  int twoPowerDim ;
  void tpd(){twoPowerDim = 1 ; for (unsigned int i=0; i<size(); i++) twoPowerDim *= 2 ;}

  void al(const Domain & d_) {vector<_Vector<NUM> >::resize(d_.n) ; tpd(); for(unsigned int i=0; i<size(); i++) (*this)[i].al(d_) ; d.copy(d_) ;}
  void zeros(const Domain & d_) {vector<_Vector<NUM> >::resize(d_.n) ; tpd(); for(unsigned int i=0; i<size(); i++) (*this)[i].zeros(d_) ; d.copy(d_) ;}
  void ones(const Domain & d_) {vector<_Vector<NUM> >::resize(d_.n) ; tpd(); for(unsigned int i=0; i<size(); i++) (*this)[i].ones(d_) ; d.copy(d_) ;}
  void copy(const _VectorMap<NUM> &src) { al(src.d); d.copy(src.d); for(unsigned int i=0; i<size(); i++) (*this)[i].copy(src[i]); }
  void zero(){for (unsigned int i=0; i<size() ; i++) (*this)[i].zero() ;}

  unsigned int size() const {return vector<_Vector<NUM> >::size();}
  unsigned int length() {return d.length ;}

  /**
     generates the identity map in domain D
  */
  void idMesh(const Domain &D) {
    al(D) ;
    Ivector I ;
    unsigned int c=0 ;
    D.putm(I) ;
    while(c < D.length) {
      for (unsigned int j=0; j< D.n; j++)
	(*this)[j][c] = I[j] ;
      c++ ;
      D.inc(I) ;
    }
  }	    


  /**
     generates the identity map in domain D,
     normalized to [0,1]
  */
  void idMeshNorm(const Domain &D) {
    al(D) ;
    Ivector I ;
    std::vector<_real> u ;
    u.resize(D.n) ;
    for (unsigned int i=0; i<D.n; i++)
      u[i] = D.getM(i) - D.getm(i) ;
    unsigned int c=0;
    D.putm(I) ;

    while(c < D.length) {
      for (unsigned int j=0; j< D.n; j++)
	(*this)[j][c] = I[j]/u[j] ;
      c++ ;
      D.inc(I) ;
    }
  }	    

  /**
     transform an affine trandformation into a VectorMap (diffeomorphism)
  */
  void affineMap(const Domain &D, const _real mat[DIM_MAX][DIM_MAX+1]) 
  {
    NUM tmp[DIM_MAX] ;
    idMesh(D) ;
    for(unsigned int c=0; c<D.length; c++) {
      for (unsigned int i=0; i<d.n; i++) {
	tmp[i] = 0 ;
	for (unsigned int j=0; j<d.n; j++)
	  tmp[i] += mat[i][j] * (*this)[j][c] ;
	tmp[i] += mat[i][d.n] ;
      }  
      for (unsigned int i=0; i<d.n; i++) 
	(*this)[i][c] = tmp[i] ;
    }
  }
    

  /**
     computes dot products at each coordinate and stores them in a vector
  */
  void scalProd(const _VectorMap<NUM> &y, _Vector<NUM> &res)  const {
    res.zeros(d) ;
    for(unsigned int i=0; i<d.length; i++)
      for(unsigned int j=0; j<d.n; j++) 
	res[i] += (*this)[j][i] * y[j][i] ;
  }

  /**
     L2 dot product between vector fields
  */
  NUM scalProd(const _VectorMap<NUM> &y) const {
    NUM res = 0 ;
    for(unsigned int i=0; i<d.length; i++)
      for(unsigned int j=0; j<d.n; j++) 
	res += (*this)[j][i] * y[j][i] ;
    return res ;
  }


  /**
     sum of squares
  */
  NUM norm2() const { 
    NUM res = 0 ;
    for (unsigned int i=0; i<size(); i++)
      res += (*this)[i].norm2() ;
    return res ;
  }

  /**
     computes norms at each coordinate and stores them in a vector
  */
  void norm(_Vector<NUM> &res)  const
  {
    res.zeros(d) ;
    for(unsigned int i=0; i<d.length; i++) {
      for(unsigned int j=0; j<d.n; j++) 
	res[i] += (*this)[j][i] * (*this)[j][i] ;
      res[i] = sqrt(res[i] + 0.0000000001) ;
    }
  }

  _Vector<NUM> norm()  const
  {
    _Vector<NUM> *res = new _Vector<NUM>[1] ;
    (*res).zeros(d) ;
    for(unsigned int i=0; i<d.length; i++) {
      for(unsigned int j=0; j<d.n; j++) 
	(*res)[i] += (*this)[j][i] * (*this)[j][i] ;
      (*res)[i] = sqrt((*res)[i] + 0.0000000001) ;
    }
    return *res ;
  }

  NUM maxNorm() const 
  {_Vector<NUM> res ; norm(res); return res.max();}

  NUM min() const {
    unsigned int i ;
    NUM res, u ;
    res = (*this)[0].min() ;
    for(i=1; i<size(); i++) {
      u = (*this)[i].min() ;
      if ( u < res)
	res = u ;
    }
    return res ;
  }

  NUM max() const {
    unsigned int i ;
    NUM res, u ;
    res = (*this)[0].min() ;
    for(i=1; i<size(); i++) {
      u = (*this)[i].max() ;
      if ( u > res)
	res = u ;
    }
    return res ;
  }


  /**
     computes the differential of *this by finite differences
  */
  void differential(std::vector<_VectorMap<NUM> > &res, vector<_real> &resol) const
  {
    res.resize(size()) ;
    for(unsigned int t=0; t<size(); t++) {
      gradient((*this)[t], res[t], resol) ;
      //      res[t] *= resol[t] ;
    }
  }
  /**
     computes the differential of *this by finite differences
  */
  void differentialDual(std::vector<_VectorMap<NUM> > &res, vector<_real> &resol) const
  {
    res.resize(size()) ;
    _real deter = 1;
    for (unsigned int i=0; i<resol.size(); i++)
      deter *= resol[1] ;
    for(unsigned int t=0; t<size(); t++) {
      gradient((*this)[t], res[t], resol) ;
      res[t] /= resol[t] * deter ;
    }
  }

  /**
     computes the inverse differential of *this by finite differences
  */
  void inverseDifferential(std::vector<_VectorMap<NUM> > &res, vector<_real> &resol) const
  {
    NUM jac ;
    std::vector<_VectorMap<NUM> > grad ;
  
    grad.resize(size()) ;
    res.resize(size()) ;
    
    for(unsigned int t=0; t<size(); t++) {
      gradient((*this)[t], grad[t], resol) ;
      //      grad[t] *= resol[t] ;
      res[t].al(d) ;
    }

    if (size() == 1) 
      for(unsigned int i=0; i<res[0].length(); i++) 
	res[0][0][i] = 1/(grad[0][0][i] + 0.00000000001) ;
    else  if (size() == 2) 
      for(unsigned int i=0; i<res[0].length(); i++) {
	jac = grad[0][0][i] * grad[1][1][i] - grad[1][0][i] * grad[0][1][i] + 0.00000000001;
	res[0][0][i] = grad[1][1][i] / jac ;
	res[0][1][i] = - grad[0][1][i]/jac ;
	res[1][0][i] = - grad[1][0][i]/jac ;
	res[1][1][i] = grad[0][0][i]/jac ;
      }
    else if (size() == 3)
      for(unsigned int i=0; i<res[0].length(); i++) {
	jac = grad[0][0][i] * grad[1][1][i] * grad[2][2][i] 
	  - grad[0][0][i] * grad[1][2][i] * grad[2][1][i]
	  - grad[0][1][i] * grad[1][0][i] * grad[2][2][i]
	  - grad[0][2][i] * grad[1][1][i] * grad[2][0][i]
	  + grad[0][1][i] * grad[1][2][i] * grad[2][0][i] 
	  + grad[0][2][i] * grad[1][0][i] * grad[2][1][i] + 0.00000000001 ;
	res[0][0][i] = (grad[1][1][i] * grad[2][2][i] - grad[1][2][i] * grad[2][1][i]) / jac ;
	res[1][1][i] = (grad[0][0][i] * grad[2][2][i] - grad[0][2][i] * grad[2][0][i]) / jac ;
	res[2][2][i] = (grad[1][1][i] * grad[0][0][i] - grad[1][0][i] * grad[0][1][i]) / jac ;
	res[0][1][i] = -(grad[0][1][i] * grad[2][2][i] - grad[2][1][i] * grad[0][2][i]) / jac ;
	res[1][0][i] = -(grad[1][0][i] * grad[2][2][i] - grad[1][2][i] * grad[2][0][i]) / jac ;
	res[0][2][i] = (grad[0][1][i] * grad[1][2][i] - grad[0][2][i] * grad[1][1][i]) / jac ;
	res[2][0][i] = (grad[1][0][i] * grad[2][1][i] - grad[2][0][i] * grad[1][1][i]) / jac ;
	res[1][2][i] = -(grad[0][0][i] * grad[1][2][i] - grad[0][2][i] * grad[1][0][i]) / jac ;
	res[2][1][i] = -(grad[0][0][i] * grad[2][1][i] - grad[2][0][i] * grad[0][1][i]) / jac ;
      }
    else {
      cerr << "no inverse in dimension higher than 3" << endl ;
      exit(1) ;
    }
  }


  int inverseMap(_VectorMap<NUM> & psi, vector<_real> &resol) const {
    _VectorMap<NUM> psi0 ;
    psi0.idMesh(d) ;
    return inverseMap(psi0, psi, resol) ;
  }

  int inverseMap(const _VectorMap<NUM> & psi0, _VectorMap<NUM> & psi, vector<_real> &resol) const {
    int flag = 1 ;
    psi.copy(psi0) ;
    _VectorMap<NUM> id, foo, dpsi, psiTry ;
    double error, errorTry ;
    id.idMesh(d) ;

    psi.multilinInterp(*this, dpsi) ;
    foo.copy(id) ;
    foo -= dpsi ;
    error = sqrt(foo.norm2())/d.length ;
    for(unsigned int k=0; k<10; k++) {
      flag = inverseDifferential(foo, dpsi, resol) ;
      if (flag == 0)
	break ;
      psiTry.copy(psi) ;
      psiTry += dpsi ;
      psiTry.multilinInterp(*this, dpsi) ;
      foo.copy(id) ;
      foo -= dpsi ;
      errorTry = sqrt(foo.norm2())/d.length ;
      cout << "inversion error " << error << " " << errorTry << endl ;
      if (errorTry < error) {
	psi.copy(psiTry) ;
	error = errorTry ;
      }
      else break ;
    }
    return flag ;
  }

  /**
     applies the inverse differential of *this to src 
  */
  int inverseDifferential(const _VectorMap<NUM> &src, _VectorMap<NUM>  &res, vector<_real> &resol) const
  {
    NUM jac ;
    Matrix D ;
    std::vector<_VectorMap<NUM> > grad ;
    int flag = 1 ;
  
    grad.resize(size()) ;
    res.resize(size()) ;
    
    for(unsigned int t=0; t<size(); t++) {
      gradient((*this)[t], grad[t], resol) ;
      //      grad[t] *= resol[t] ;
      res[t].al(d) ;
    }

    if (size() == 1) {
      D.resize(1,1) ;
      for(unsigned int i=0; i<res[0].length(); i++) {
	D(0,0) = 1/(grad[0][0][i] + 0.00000000001) ;
	if (D(0,0) < 0)
	  flag = 0 ;
	res[0][i] = D(0,0) * src[0][i] ; 
      }
    }
    else  if (size() == 2) {
      D.resize(2,2) ;
      for(unsigned int i=0; i<res[0].length(); i++) {
	jac = grad[0][0][i] * grad[1][1][i] - grad[1][0][i] * grad[0][1][i] + 0.00000000001;
	if (jac < 0)
	  flag = 0 ;
	D(0,0)  = grad[1][1][i] / jac ;
	D(0,1)  = - grad[0][1][i]/jac ;
	D(1,0)  = - grad[1][0][i]/jac ;
	D(1,1)  = grad[0][0][i]/jac ;
	for (unsigned int k=0; k<2; k++) {
	  res[k][i] = 0 ;
	  for (unsigned int l=0; l<2; l++)
	    res[k][i] += D(k,l) * src[l][i] ;
	}
      }
    }
    else if (size() == 3) {
      D.resize(3,3) ;
      for(unsigned int i=0; i<res[0].length(); i++) {
	jac = grad[0][0][i] * grad[1][1][i] * grad[2][2][i] 
	  - grad[0][0][i] * grad[1][2][i] * grad[2][1][i]
	  - grad[0][1][i] * grad[1][0][i] * grad[2][2][i]
	  - grad[0][2][i] * grad[1][1][i] * grad[2][0][i]
	  + grad[0][1][i] * grad[1][2][i] * grad[2][0][i] 
	  + grad[0][2][i] * grad[1][0][i] * grad[2][1][i] + 0.00000000001 ;
	if (jac < 0)
	  flag = 0 ;
	D(0,0) = (grad[1][1][i] * grad[2][2][i] - grad[1][2][i] * grad[2][1][i]) / jac ;
	D(1,1) = (grad[0][0][i] * grad[2][2][i] - grad[0][2][i] * grad[2][0][i]) / jac ;
	D(2,2) = (grad[1][1][i] * grad[0][0][i] - grad[1][0][i] * grad[0][1][i]) / jac ;
	D(0,1) = -(grad[0][1][i] * grad[2][2][i] - grad[2][1][i] * grad[0][2][i]) / jac ;
	D(1,0) = -(grad[1][0][i] * grad[2][2][i] - grad[1][2][i] * grad[2][0][i]) / jac ;
	D(0,2) = (grad[0][1][i] * grad[1][2][i] - grad[0][2][i] * grad[1][1][i]) / jac ;
	D(2,0) = (grad[1][0][i] * grad[2][1][i] - grad[2][0][i] * grad[1][1][i]) / jac ;
	D(1,2) = -(grad[0][0][i] * grad[1][2][i] - grad[0][2][i] * grad[1][0][i]) / jac ;
	D(2,1) = -(grad[0][0][i] * grad[2][1][i] - grad[2][0][i] * grad[0][1][i]) / jac ;
	for (unsigned int k=0; k<3; k++) {
	  res[k][i] = 0 ;
	  for (unsigned int l=0; l<3; l++)
	    res[k][i] += D(k,l) * src[l][i] ;
	}
      }
    }
    else {
      cerr << "no inverse in dimension higher than 3" << endl ;
      exit(1) ;
    }
    return flag ;
  }


  /**
     computes the differential and inverse of *this by finite differences
  */
  void differentialAndInverse(std::vector<_VectorMap<NUM> > &diff, std::vector<_VectorMap<NUM> > &idiff, _Vector<NUM> &jac, vector<_real>&resol) const
  {
    diff.resize(size()) ;
    idiff.resize(size()) ;
  
    for(unsigned int t=0; t<size(); t++) {
      gradient((*this)[t], diff[t], resol) ;
      //      diff[t] *= resol[t] ;
      idiff[t].al(d) ;
    }
    jac.al(d) ;

    if (size() == 1) 
      for(unsigned int i=0; i<idiff[0].length(); i++) {
	jac[i] = diff[0][0][i] ;
	idiff[0][0][i] = 1/(jac[i] + 0.00000000001) ;
      }
    else  if (size() == 2) 
      for(unsigned int i=0; i<idiff[0].length(); i++) {
	jac[i] = diff[0][0][i] * diff[1][1][i] - diff[1][0][i] * diff[0][1][i] + 0.00000000001;
	idiff[0][0][i] = diff[1][1][i] / jac[i] ;
	idiff[0][1][i] = - diff[0][1][i]/jac[i] ;
	idiff[1][0][i] = - diff[1][0][i]/jac[i] ;
	idiff[1][1][i] = diff[0][0][i]/jac[i] ;
      }
    else if (size() == 3)
      for(unsigned int i=0; i<idiff[0].length(); i++) {
	jac[i] = diff[0][0][i] * diff[1][1][i] * diff[2][2][i] 
	  - diff[0][0][i] * diff[1][2][i] * diff[2][1][i]
	  - diff[0][1][i] * diff[1][0][i] * diff[2][2][i]
	  - diff[0][2][i] * diff[1][1][i] * diff[2][0][i]
	  + diff[0][1][i] * diff[1][2][i] * diff[2][0][i] 
	  + diff[0][2][i] * diff[1][0][i] * diff[2][1][i] + 0.00000000001 ;
	idiff[0][0][i] = (diff[1][1][i] * diff[2][2][i] - diff[1][2][i] * diff[2][1][i]) / jac[i] ;
	idiff[1][1][i] = (diff[0][0][i] * diff[2][2][i] - diff[0][2][i] * diff[2][0][i]) / jac[i] ;
	idiff[2][2][i] = (diff[1][1][i] * diff[0][0][i] - diff[1][0][i] * diff[0][1][i]) / jac[i] ;
	idiff[0][1][i] = -(diff[0][1][i] * diff[2][2][i] - diff[2][1][i] * diff[0][2][i]) / jac[i] ;
	idiff[1][0][i] = -(diff[1][0][i] * diff[2][2][i] - diff[1][2][i] * diff[2][0][i]) / jac[i] ;
	idiff[0][2][i] = (diff[0][1][i] * diff[1][2][i] - diff[0][2][i] * diff[1][1][i]) / jac[i] ;
	idiff[2][0][i] = (diff[1][0][i] * diff[2][1][i] - diff[2][0][i] * diff[1][1][i]) / jac[i] ;
	idiff[1][2][i] = -(diff[0][0][i] * diff[1][2][i] - diff[0][2][i] * diff[1][0][i]) / jac[i] ;
	idiff[2][1][i] = -(diff[0][0][i] * diff[2][1][i] - diff[2][0][i] * diff[0][1][i]) / jac[i] ;
      }
    else {
      cerr << "no inverse in dimension higher than 3" << endl ;
      exit(1) ;
    }
  }



  /**
     computes the jacobian of *this by finite differences
  */
  void jacobian(_Vector<NUM> &res, vector<_real> &resol) const
  {
    std::vector<_VectorMap<NUM> > grad ;
    grad.resize(size()) ;
  
    for(unsigned int t=0; t<size(); t++) {
      gradient((*this)[t], grad[t], resol) ;
      //      grad[t] *= resol[t] ;
    }
    res.al(d) ;

    if (size() == 1) 
      for(unsigned int i=0; i<res.length(); i++) 
	res[i] = fabs(grad[0][0][i]) ;
    else  if (size() == 2) 
      for(unsigned int i=0; i<res.length(); i++) 
	res[i] = fabs(grad[0][0][i] * grad[1][1][i] - grad[1][0][i] * grad[0][1][i]) ;
    else if (size() == 3)
      for(unsigned int i=0; i<res.length(); i++) 
	res[i] = fabs(grad[0][0][i] * grad[1][1][i] * grad[2][2][i] 
		      - grad[0][0][i] * grad[1][2][i] * grad[2][1][i]
		      - grad[0][1][i] * grad[1][0][i] * grad[2][2][i]
		      - grad[0][2][i] * grad[1][1][i] * grad[2][0][i]
		      + grad[0][1][i] * grad[1][2][i] * grad[2][0][i] 
		      + grad[0][2][i] * grad[1][0][i] * grad[2][1][i]) ;
    else {
      cerr << "no jacobian in dimension higher than 3" << endl ;
      exit(1) ;
    }
  }


  void displacement(_Vector<NUM> &res) const
  {
    _VectorMap id ;
    id.idMesh(d) ;
    id -= *this ;
    id.norm(res) ;
  }

  /**
     computes the inverse jacobian of *this by finite differences
  */
  void invJacobian(_Vector<NUM> &res, vector<_real>&resol) const
  {
    std::vector<_VectorMap<NUM> > grad ;
    grad.resize(size()) ;
    
    for(unsigned int t=0; t<size(); t++) {
      gradient((*this)[t], grad[t], resol) ;
    }
    res.al(d) ;
    
    if (size() == 1) 
      for(unsigned int i=0; i<res.length(); i++) 
	res[i] = 1/fabs(grad[0][0][i]) ;
    else  if (size() == 2) 
      for(unsigned int i=0; i<res.length(); i++) 
	res[i] = 1/fabs(grad[0][0][i] * grad[1][1][i] - grad[1][0][i] * grad[0][1][i]) ;
    else if (size() == 3)
      for(unsigned int i=0; i<res.length(); i++) 
	res[i] = 1/fabs(grad[0][0][i] * grad[1][1][i] * grad[2][2][i] 
			- grad[0][0][i] * grad[1][2][i] * grad[2][1][i]
			- grad[0][1][i] * grad[1][0][i] * grad[2][2][i]
			- grad[0][2][i] * grad[1][1][i] * grad[2][0][i]
			+ grad[0][1][i] * grad[1][2][i] * grad[2][0][i] 
			+ grad[0][2][i] * grad[1][0][i] * grad[2][1][i]) ;
    else {
      cerr << "no jacobian in dimension higher than 3" << endl ;
      exit(1) ;
    }
  }

  /**
     computes the logarithm of the jacobian of *this by finite differences
  */
  void logJacobian(_Vector<NUM> &res, vector<_real> &resol) const
  {
    std::vector<_VectorMap<NUM> > grad ;
    grad.resize(size()) ;
  
    for(unsigned int t=0; t<size(); t++) {
      gradient((*this)[t], grad[t], resol) ;
    }
    res.al(d) ;
    if (size() == 1) 
      for(unsigned int i=0; i<res.length(); i++) 
	res[i] = log(fabs(grad[0][0][i])) ;
    else  if (size() == 2) 
      for(unsigned int i=0; i<res.length(); i++) 
	res[i] = log(fabs(grad[0][0][i] * grad[1][1][i] - grad[1][0][i] * grad[0][1][i])) ;
    else if (size() == 3)
      for(unsigned int i=0; i<res.length(); i++) 
	res[i] = log(fabs(grad[0][0][i] * grad[1][1][i] * grad[2][2][i] 
			  - grad[0][0][i] * grad[1][2][i] * grad[2][1][i]
			  - grad[0][1][i] * grad[1][0][i] * grad[2][2][i]
			  - grad[0][2][i] * grad[1][1][i] * grad[2][0][i]
			  + grad[0][1][i] * grad[1][2][i] * grad[2][0][i] 
			  + grad[0][2][i] * grad[1][0][i] * grad[2][1][i])) ;
    else {
      cerr << "no jacobian in dimension higher than 3" << endl ;
      exit(1) ;
    }
  }

  /**
     Discrete jacobian
  */
  void discreteJacobian(_Vector<NUM> &res0) const
  {
    res0.al(d) ;
    res0.zero() ;

    Ivector I ;
    std::vector<_real> r ;
  
    I.resize(size()) ;
    r.resize(size()) ;
    for(unsigned int c =0; c<d.length; c++) {
      std::vector<_real>::iterator ir =r.begin() ;
      std::vector<int>::iterator ii=I.begin();
      for(unsigned int i=0; i<I.size(); i++, ++ii, ++ir) {
	*ii = (int) floor((*this)[i][c]) ;
	if (*ii < d.getm(i)) {
	  *ii = d.getm(i) ;
	  *ir = 0 ;
	}
	else if (*ii >= d.getM(i)) {
	  *ii = d.getM(i)-1 ;
	  *ir = 1 ;
	}
	else 
	  *ir = (*this)[i][c] - *ii ;
      }

      int i0 = d.position(I) ;
      for (int k=0; k< twoPowerDim ; k++) {
	_real weight = 1 ;
	int j = 1, ii = i0 ;
	ir = r.begin() ;
	for (unsigned int i=0; i<size(); i++, ++ir) {
	  if (k & j) {
	    weight *= *ir ;
	    ii += d.getCum(i) ;
	  }
	  else {
	    weight *= 1 - *ir ;
	  }
	  j <<= 1 ;
	}
	res0[ii] += weight  ; 
      }
    }
  }

  
  // interpolations

  void pointSetInterp(const PointSet &p0, PointSet &q0) const
  {
    double prec = 1e-10 ;
    q0.al(p0.size(), p0.dim()) ;
    if (d.n == 1) {
      int offset = - d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0) ; 
      int iu0 ;  
      _real r0, u0, ri0;
      int c1 = d.getCum(0) ;
      c_iterator src0 = (*this)[0].begin(), srcI ;
      for (unsigned int k=0; k<p0.size(); k++) {
	ri0 = floor(p0[k][0] / prec) * prec ;
	u0 = minV(maxV(ri0, m0), M0-prec) ;
	iu0 = (int) floor(u0) ;
	srcI = src0  + iu0*c1 + offset ;
	r0 = u0 - iu0 ;
	q0[0][k] = (*(srcI)) * (1-r0)  
	  + (*(srcI +c1)) * r0 ; 
      }
    }
    else if (d.n == 2) {
      int offset = - d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0), m1 = d.getm(1), M1 = d.getM(1) ; 
      int iu0, iu1 ;  
      _real r0, u0, r1, u1, ri0, ri1;
      int c10 = d.getCum(0), c01 = d.getCum(1),
	c11 = c10 + c01 ;
      c_iterator src0 = (*this)[0].begin(), srcI ;
      for(unsigned int j=0; j<2; j++) {
	src0 = (*this)[j].begin() ;
	for (unsigned int k=0; k<p0.size(); k++) {
	  ri0 = floor(p0[k][0] / prec) * prec ;
	  ri1 = floor(p0[k][1] / prec) * prec ;
	  u0 = minV(maxV(ri0, m0), M0-prec) ;
	  u1 = minV(maxV(ri1, m1), M1-prec) ;
	  iu0 = (int) floor(u0) ;
	  iu1 = (int) floor(u1) ;
	  srcI = src0 +  iu1*c01 + iu0*c10 + offset ;
	  r0 = u0 - iu0 ;
	  r1 = u1- iu1 ;
	  q0[j][k] = (*(srcI)) * (1-r0)*(1-r1)  
	    + (*(srcI +c10)) * r0 * (1-r1) 
	    + (*(srcI +c01)) *  (1-r0) * r1  
	    + (*(srcI+c11)) * r0 * r1 ;
	}
      }
    }
    else if  (d.n == 3) {
      int offset = - d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0), m1 = d.getm(1), M1 = d.getM(1), 
	m2 = d.getm(2), M2 = d.getM(2) ;
      int iu0, iu1, iu2 ;  
      _real r0, u0, r1, u1, r2, u2, ri0, ri1, ri2;
      int c100 = d.getCum(0), c010 = d.getCum(1), c001 = d.getCum(2),
	c110 = c100 + c010, c101 = c100 + c001, c011 = c010 + c001, 
	c111 = c110 + c001 ;
      c_iterator src0 = (*this)[0].begin(), srcI ;
      for(unsigned int j=0; j<3; j++) {
	src0 = (*this)[j].begin() ;
	for (unsigned int k=0; k<p0.size(); k++) {
	  ri0 = floor(p0[k][0] / prec) * prec ;
	  ri1 = floor(p0[k][1] / prec) * prec ;
	  ri2 = floor(p0[k][2] / prec) * prec ;
	  u0 = minV(maxV(ri0, m0), M0-prec) ;
	  u1 = minV(maxV(ri1, m1), M1-prec) ;
	  u2 = minV(maxV(ri2, m2), M2-prec) ;
	  iu0 = (int) floor(u0) ;
	  iu1 = (int) floor(u1) ;
	  iu2 = (int) floor(u2) ;
	  srcI = src0 + iu2*c001 + iu1*c010 + iu0*c100 + offset ;
	  r0 = u0 - iu0 ;
	  r1 = u1- iu1 ;
	  r2 = u2- iu2 ;
	  q0[j][k] = (*(srcI)) * (1-r0)*(1-r1) * (1-r2) 
	    + (*(srcI +c100)) * r0 * (1-r1) * (1-r2)
	    + (*(srcI +c010)) *  (1-r0) * r1 * (1-r2) 
	    + (*(srcI+c110)) * r0 * r1 * (1-r2)
	    + (*(srcI+c001)) * (1-r0)*(1-r1) *  r2 
	    + (*(srcI+c101)) * r0 * (1-r1) * r2
	    + (*(srcI+c011)) *  (1-r0) * r1 * r2 
	    + (*(srcI+c111)) * r0 * r1 * r2 ;
	}
      }
    }
    else {
      cerr << "No implementation of point set interpolation for d > 3" << endl;
      exit(1) ;
    }
  }


  void multilinInterp(const _Vector<NUM> &src, _Vector<NUM> &res0) const
  {
    multilinInterpNEW(src, res0) ;
  }

  void multilinInterp(const _VectorMap<NUM> &src, _VectorMap<NUM> &res0) const
  {
    multilinInterpNEW(src, res0) ;
  }

  void multilinInterpDual(const _VectorMap<NUM> &src, _VectorMap<NUM> &res0) const
  {
    res0.al(src.d) ;
    for (unsigned int k=0; k<src.size(); k++)
      multilinInterpDual(src[k], res0[k]) ;
  }

  void multilinInterpNEW(const _Vector<NUM> &src, _Vector<NUM> &res0) const
  {
    _Vector<NUM> res ;
    res.al(d) ;
    double prec = 1e-10 ;

    if (d.n  == 1) {
      _iterator I ;
      int offset = - d.getCumMin(), d0=d.getCum(0), m0 = d.getm(0), M0 = d.getM(0) ;
      int phi0, iu0 ;  ;
      _real r0, u0, ri0 ;
      c_iterator i0 =(*this)[0].begin() ;
      for(I=res.begin(); I!=res.end(); ++I) {
	ri0 = floor(*i0 / prec) * prec ;
	u0 = minV(maxV(ri0, m0), M0-prec) ;
	iu0 = (int) floor(u0) ;
	phi0 = iu0*d0 + offset ;
	r0 = u0 - iu0 ;
	(*I) = src[phi0] * (1-r0) + src[phi0+1] * r0 ;
	i0++ ;
      }
    }
    else if (d.n == 2) {
      _iterator I ;
      int offset = - d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0), m1 = d.getm(1), M1 = d.getM(1) ;
      int iu0, iu1 ;  ;
      _real r0, u0, r1, u1, ri0, ri1;
      c_iterator i0 =(*this)[0].begin(), i1 =(*this)[1].begin() ;
      c_iterator src0 = src.begin(), srcI ;
      int c10 = d.getCum(0), c01 = d.getCum(1), c11 = c10 + c01 ;
      for(I=res.begin(); I!=res.end(); ++I) {
	ri0 = floor(*i0 / prec) * prec ;
	ri1 = floor(*i1 / prec) * prec ;

	u0 = minV(maxV(ri0, m0), M0-prec) ;
	u1 = minV(maxV(ri1, m1), M1-prec) ;
	iu0 = (int) floor(u0) ;
	iu1 = (int) floor(u1) ;

	srcI = src0 + iu1*c01 + iu0*c10 + offset ;
	r0 = u0 - iu0 ;
	r1 = u1-iu1 ;
	*I = (*srcI) * (1-r0)*(1-r1) 
	  + (*(srcI +  c10)) * r0 * (1-r1)
	  + (*(srcI + c01)) * r1 * (1-r0)
	  + (*(srcI + c11)) * r1 * r0 ;
	i0++ ;
	i1++ ;
      }
    } 
    else if (d.n == 3) {
      _iterator I ;
      int offset = - d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0), m1 = d.getm(1), M1 = d.getM(1), 
	m2 = d.getm(2), M2 = d.getM(2) ;
      int iu0, iu1, iu2 ;  ;
      _real r0, u0, r1, u1, r2, u2, ri0, ri1, ri2;
      c_iterator src0 = src.begin(), srcI ;
      c_iterator i0 =(*this)[0].begin(), i1 =(*this)[1].begin(), i2 =(*this)[2].begin() ;
      int c100 = d.getCum(0), c010 = d.getCum(1), c001 = d.getCum(2),
	c110 = c100 + c010, c101 = c100 + c001, c011 = c010 + c001, 
	c111 = c110 + c001 ;

      for(I=res.begin(); I!=res.end(); ++I) {
	ri0 = floor(*i0 / prec) * prec ;
	ri1 = floor(*i1 / prec) * prec ;
	ri2 = floor(*i2 / prec) * prec ;
	u0 = minV(maxV(ri0, m0), M0-prec) ;
	u1 = minV(maxV(ri1, m1), M1-prec) ;
	u2 = minV(maxV(ri2, m2), M2-prec) ;
	iu0 = (int) floor(u0) ;
	iu1 = (int) floor(u1) ;
	iu2 = (int) floor(u2) ;
	srcI = src0 + iu2*c001 + iu1*c010 + iu0*c100 + offset ;
	r0 = u0 - iu0 ;
	r1 = u1- iu1 ;
	r2 = u2- iu2 ;
	*I = (*(srcI)) * (1-r0)*(1-r1) * (1-r2) 
	  + (*(srcI +c100)) * r0 * (1-r1) * (1-r2)
	  + (*(srcI +c010)) *  (1-r0) * r1 * (1-r2) 
	  + (*(srcI+c110)) * r0 * r1 * (1-r2)
	  + (*(srcI+c001)) * (1-r0)*(1-r1) *  r2 
	  + (*(srcI+c101)) * r0 * (1-r1) * r2
	  + (*(srcI+c011)) *  (1-r0) * r1 * r2 
	  + (*(srcI+c111)) * r0 * r1 * r2 ;
	i0++ ;
	i1++ ;
	i2++ ;
      }
    } else {
      cout << "dim = " << d.n << "?" << endl ;
      multilinInterpOLD(src, res) ;
    }
    res0.copy(res) ;
  }

  void multilinInterpGradient(const _Vector<NUM> &src, _VectorMap<NUM> &res0) const
  {
    res0.al(d) ;
    double prec = 1e-10 ;

    if (d.n  == 1) {
    _Vector<NUM> res ;
    res.al(d) ;
      _iterator I ;
      int offset = - d.getCumMin(), d0=d.getCum(0), m0 = d.getm(0), M0 = d.getM(0) ;
      int phi0, iu0 ;  ;
      _real u0, ri0 ;
      c_iterator i0 =(*this)[0].begin() ;
      for(I=res.begin(); I!=res.end(); ++I) {
	ri0 = floor(*i0 / prec) * prec ;
	u0 = minV(maxV(ri0, m0), M0-prec) ;
	iu0 = (int) floor(u0) ;
	phi0 = iu0*d0 + offset ;
	(*I) = src[phi0+1] - src[phi0] ;
	i0++ ;
      }
      res0[0].copy(res) ;
    }
    else if (d.n == 2) {
      _Vector<NUM> res1, res2 ;
      res1.al(d) ;
      res2.al(d) ;
      _iterator I1, I2 ;
      int offset = - d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0), m1 = d.getm(1), M1 = d.getM(1) ;
      int iu0, iu1 ;  ;
      _real r0, u0, r1, u1, ri0, ri1;
      c_iterator i0 =(*this)[0].begin(), i1 =(*this)[1].begin() ;
      c_iterator src0 = src.begin(), srcI ;
      int c10 = d.getCum(0), c01 = d.getCum(1), c11 = c10 + c01 ;
      I2 = res2.begin() ;
      for(I1=res1.begin() ; I1!=res1.end(); ++I1) {
	ri0 = floor(*i0 / prec) * prec ;
	ri1 = floor(*i1 / prec) * prec ;

	u0 = minV(maxV(ri0, m0), M0-prec) ;
	u1 = minV(maxV(ri1, m1), M1-prec) ;
	iu0 = (int) floor(u0) ;
	iu1 = (int) floor(u1) ;
	srcI = src0 + iu1*c01 + iu0*c10 + offset ;
	r0 = u0 - iu0 ;
	r1 = u1-iu1 ;
	*I1 = - (*srcI) * (1-r1) 
	  + (*(srcI +  c10)) * (1-r1)
	  - (*(srcI + c01)) * r1
	  + (*(srcI + c11)) * r1  ;
	*I2 = -(*srcI) * (1-r0) 
	  - (*(srcI +  c10)) * r0 
	  + (*(srcI + c01)) * (1-r0)
	  + (*(srcI + c11)) *  r0 ;
	i0++ ;
	i1++ ;
	++I2;
      }
      res0[0].copy(res1) ;
      res0[1].copy(res2) ;
    } 
    else if (d.n == 3) {
      _Vector<NUM> res1, res2, res3 ;
      res1.al(d) ;
      res2.al(d) ;
      res3.al(d) ;
      _iterator I1, I2,I3 ;
      int offset = - d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0), m1 = d.getm(1), M1 = d.getM(1), 
	m2 = d.getm(2), M2 = d.getM(2) ;
      int iu0, iu1, iu2 ;  ;
      _real r0, u0, r1, u1, r2, u2, ri0, ri1, ri2;
      c_iterator src0 = src.begin(), srcI ;
      c_iterator i0 =(*this)[0].begin(), i1 =(*this)[1].begin(), i2 =(*this)[2].begin() ;
      int c100 = d.getCum(0), c010 = d.getCum(1), c001 = d.getCum(2),
	c110 = c100 + c010, c101 = c100 + c001, c011 = c010 + c001, 
	c111 = c110 + c001 ;

      for(I1=res1.begin(), I2=res2.begin(), I3=res3.begin(); I1!=res1.end(); ++I1, ++I2, ++I3) {
	ri0 = floor(*i0 / prec) * prec ;
	ri1 = floor(*i1 / prec) * prec ;
	ri2 = floor(*i2 / prec) * prec ;
	u0 = minV(maxV(ri0, m0), M0-prec) ;
	u1 = minV(maxV(ri1, m1), M1-prec) ;
	u2 = minV(maxV(ri2, m2), M2-prec) ;
	iu0 = (int) floor(u0) ;
	iu1 = (int) floor(u1) ;
	iu2 = (int) floor(u2) ;
	srcI = src0 + iu2*c001 + iu1*c010 + iu0*c100 + offset ;
	r0 = u0 - iu0 ;
	r1 = u1- iu1 ;
	r2 = u2- iu2 ;
	*I1 = -(*(srcI)) * (1-r1) * (1-r2) 
	  + (*(srcI +c100)) * (1-r1) * (1-r2)
	  - (*(srcI +c010)) * r1 * (1-r2) 
	  + (*(srcI+c110)) * r1 * (1-r2)
	  - (*(srcI+c001)) * (1-r1) *  r2 
	  + (*(srcI+c101)) * (1-r1) * r2
	  - (*(srcI+c011))  * r1 * r2 
	  + (*(srcI+c111)) * r1 * r2 ;
	*I2 = -(*(srcI)) * (1-r0) * (1-r2) 
	  - (*(srcI +c100)) * r0  * (1-r2)
	  + (*(srcI +c010)) *  (1-r0) * (1-r2) 
	  + (*(srcI+c110)) * r0 * (1-r2)
	  - (*(srcI+c001)) * (1-r0) *  r2 
	  - (*(srcI+c101)) * r0  * r2
	  + (*(srcI+c011)) *  (1-r0)  * r2 
	  + (*(srcI+c111)) * r0  * r2 ;
	*I3 = -(*(srcI)) * (1-r0)*(1-r1)  
	  - (*(srcI +c100)) * r0 * (1-r1)
	  - (*(srcI +c010)) *  (1-r0) * r1
	  - (*(srcI+c110)) * r0 * r1 
	  + (*(srcI+c001)) * (1-r0)*(1-r1)  
	  + (*(srcI+c101)) * r0 * (1-r1)
	  + (*(srcI+c011)) *  (1-r0) * r1 
	  + (*(srcI+c111)) * r0 * r1 ;
	i0++ ;
	i1++ ;
	i2++ ;
      }
      res0[0].copy(res1) ;
      res0[1].copy(res2) ;
      res0[2].copy(res3) ;
    } else {
      cout << "dim = " << d.n << endl ;
      cerr << "gradient interpolation not implemented" << endl ;
      exit(1) ;
    }
  }

  void multilinInterpGradient(const _VectorMap<NUM> &src, std::vector<_VectorMap<NUM> > &res0) const
  {
    res0.resize(d.n) ;
    for (unsigned int k=0; k<res0.size(); k++)
      multilinInterpGradient(src[k], res0[k]) ;
  }

  void multilinInterpDual(const _Vector<NUM> &src, _Vector<NUM> &res0) const
  {
    _Vector<NUM> res ;
    res.zeros(d) ;
    double prec = 1e-10 ;

    if (d.n  == 1) {
      c_iterator I ;
      int offset = - d.getCumMin(), d0=d.getCum(0), m0 = d.getm(0), M0 = d.getM(0) ;
      int phi0, iu0 ;  ;
      _real r0, u0, ri0 ;
      c_iterator i0 =(*this)[0].begin() ;
      for(I=src.begin(); I!=src.end(); ++I) {
	ri0 = floor(*i0 / prec) * prec ;
	u0 = minV(maxV(ri0, m0), M0-prec) ;
	iu0 = (int) floor(u0) ;
	phi0 = iu0*d0 + offset ;
	r0 = u0 - iu0 ;
	res[phi0] += (*I) * (1-r0) ;
	res[phi0+1] += (*I) * r0 ;
	i0++ ;
      }
    }
    else if (d.n == 2) {
      c_iterator I ;
      int offset = - d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0), m1 = d.getm(1), M1 = d.getM(1) ;
      int iu0, iu1 ;  ;
      _real r0, u0, r1, u1, ri0, ri1;
      c_iterator i0 =(*this)[0].begin(), i1 =(*this)[1].begin() ;
      _iterator res00 = res.begin(), resI ;
      int c10 = d.getCum(0), c01 = d.getCum(1), c11 = c10 + c01 ;
      for(I=src.begin(); I!=src.end(); ++I) {
	ri0 = floor(*i0 / prec) * prec ;
	ri1 = floor(*i1 / prec) * prec ;

	u0 = minV(maxV(ri0, m0), M0-prec) ;
	u1 = minV(maxV(ri1, m1), M1-prec) ;
	iu0 = (int) floor(u0) ;
	iu1 = (int) floor(u1) ;
	resI = res00 + iu1*c01 + iu0*c10 + offset ;
	r0 = u0 - iu0 ;
	r1 = u1-iu1 ;
	(*resI) += (*I) * (1-r0)*(1-r1) ;
	(*(resI +  c10)) += (*I) * r0 * (1-r1) ;
	(*(resI + c01)) += (*I) * r1 * (1-r0) ;
	(*(resI + c11)) += (*I) * r1 * r0 ;
	i0++ ;
	i1++ ;
      }
    } 
    else if (d.n == 3) {
      c_iterator I ;
      int offset = - d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0), m1 = d.getm(1), M1 = d.getM(1), 
	m2 = d.getm(2), M2 = d.getM(2) ;
      int iu0, iu1, iu2 ;  ;
      _real r0, u0, r1, u1, r2, u2, ri0, ri1, ri2;
      _iterator res00 = res.begin(), resI ;
      c_iterator i0 =(*this)[0].begin(), i1 =(*this)[1].begin(), i2 =(*this)[2].begin() ;
      int c100 = d.getCum(0), c010 = d.getCum(1), c001 = d.getCum(2),
	c110 = c100 + c010, c101 = c100 + c001, c011 = c010 + c001, 
	c111 = c110 + c001 ;

      for(I=src.begin(); I!=src.end(); ++I) {
	ri0 = floor(*i0 / prec) * prec ;
	ri1 = floor(*i1 / prec) * prec ;
	ri2 = floor(*i2 / prec) * prec ;
	u0 = minV(maxV(ri0, m0), M0-prec) ;
	u1 = minV(maxV(ri1, m1), M1-prec) ;
	u2 = minV(maxV(ri2, m2), M2-prec) ;
	iu0 = (int) floor(u0) ;
	iu1 = (int) floor(u1) ;
	iu2 = (int) floor(u2) ;
	resI = res00 + iu2*c001 + iu1*c010 + iu0*c100 + offset ;
	r0 = u0 - iu0 ;
	r1 = u1- iu1 ;
	r2 = u2- iu2 ;
	(*(resI)) += (*I) * (1-r0)*(1-r1) * (1-r2) ;
	(*(resI +c100)) += (*I)* r0 * (1-r1) * (1-r2) ;
	(*(resI +c010)) += (*I) *  (1-r0) * r1 * (1-r2) ;
	(*(resI+c110)) += (*I) * r0 * r1 * (1-r2) ;
	(*(resI+c001)) += (*I) * (1-r0)*(1-r1) *  r2 ;
	(*(resI+c101)) += (*I) * r0 * (1-r1) * r2 ;
	(*(resI+c011)) += (*I) *  (1-r0) * r1 * r2 ;
	(*(resI+c111)) += (*I) * r0 * r1 * r2 ;
	i0++ ;
	i1++ ;
	i2++ ;
      }
    } else {
      cout << "dim = " << d.n << endl ;
      multilinInterpOLD(src, res) ;
    }
    res0.copy(res) ;
  }

  void multilinInterpDualGradient(const _Vector<NUM> &src, _VectorMap<NUM> &res0) const
  {
    res0.al(d) ;
    double prec = 1e-10 ;

    if (d.n  == 1) {
      _Vector<NUM> res ;
      res.zeros(d) ;
      c_iterator I ;
      int offset = - d.getCumMin(), d0=d.getCum(0), m0 = d.getm(0), M0 = d.getM(0) ;
      int phi0, iu0 ;  ;
      _real r0, u0, ri0 ;
      c_iterator i0 =(*this)[0].begin() ;
      for(I=src.begin(); I!=src.end(); ++I) {
	ri0 = floor(*i0 / prec) * prec ;
	u0 = minV(maxV(ri0, m0), M0-prec) ;
	iu0 = (int) floor(u0) ;
	phi0 = iu0*d0 + offset ;
	r0 = u0 - iu0 ;
	res[phi0] -= (*I) ;
	res[phi0+1] += (*I) ;
	i0++ ;
      }
      res0[0].copy(res) ;
    }
    else if (d.n == 2) {
      _Vector<NUM> res1, res2 ;
      res1.zeros(d) ;
      res2.zeros(d) ;

      c_iterator I ;
      int offset = - d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0), m1 = d.getm(1), M1 = d.getM(1) ;
      int iu0, iu1 ;  ;
      _real r0, u0, r1, u1, ri0, ri1;
      c_iterator i0 =(*this)[0].begin(), i1 =(*this)[1].begin() ;
      _iterator res01 = res1.begin(), res02 = res2.begin(), resI1, resI2 ;
      int c10 = d.getCum(0), c01 = d.getCum(1), c11 = c10 + c01 ;
      for(I=src.begin(); I!=src.end(); ++I) {
	ri0 = floor(*i0 / prec) * prec ;
	ri1 = floor(*i1 / prec) * prec ;

	u0 = minV(maxV(ri0, m0), M0-prec) ;
	u1 = minV(maxV(ri1, m1), M1-prec) ;
	iu0 = (int) floor(u0) ;
	iu1 = (int) floor(u1) ;
	resI1 = res01 + iu1*c01 + iu0*c10 + offset ;
	resI2 = res02 + iu1*c01 + iu0*c10 + offset ;
	r0 = u0 - iu0 ;
	r1 = u1-iu1 ;
	(*resI1) -= (*I) * (1-r1) ;
	(*(resI1 +  c10)) += (*I) * (1-r1) ;
	(*(resI1 + c01)) -= (*I) * r1 ;
	(*(resI1 + c11)) += (*I) * r1 ;
	(*resI2) -= (*I) * (1-r0) ;
	(*(resI2 +  c10)) -= (*I) * r0 ;
	(*(resI2 + c01)) += (*I) * (1-r0) ;
	(*(resI2 + c11)) += (*I) * r0 ;
	i0++ ;
	i1++ ;
      }
      res0[0].copy(res1) ;
      res0[1].copy(res2) ;
    } 
    else if (d.n == 3) {
      _Vector<NUM> res1, res2, res3 ;
      res1.zeros(d) ;
      res2.zeros(d) ;
      res3.zeros(d) ;
      c_iterator I ;
      int offset = - d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0), m1 = d.getm(1), M1 = d.getM(1), 
	m2 = d.getm(2), M2 = d.getM(2) ;
      int iu0, iu1, iu2 ;  ;
      _real r0, u0, r1, u1, r2, u2, ri0, ri1, ri2;
      _iterator res01 = res1.begin(), resI1 ;
      _iterator res02 = res2.begin(), resI2 ;
      _iterator res03 = res3.begin(), resI3 ;
      c_iterator i0 =(*this)[0].begin(), i1 =(*this)[1].begin(), i2 =(*this)[2].begin() ;
      int c100 = d.getCum(0), c010 = d.getCum(1), c001 = d.getCum(2),
	c110 = c100 + c010, c101 = c100 + c001, c011 = c010 + c001, 
	c111 = c110 + c001 ;

      for(I=src.begin(); I!=src.end(); ++I) {
	ri0 = floor(*i0 / prec) * prec ;
	ri1 = floor(*i1 / prec) * prec ;
	ri2 = floor(*i2 / prec) * prec ;
	u0 = minV(maxV(ri0, m0), M0-prec) ;
	u1 = minV(maxV(ri1, m1), M1-prec) ;
	u2 = minV(maxV(ri2, m2), M2-prec) ;
	iu0 = (int) floor(u0) ;
	iu1 = (int) floor(u1) ;
	iu2 = (int) floor(u2) ;
	resI1 = res01 + iu2*c001 + iu1*c010 + iu0*c100 + offset ;
	resI2 = res02 + iu2*c001 + iu1*c010 + iu0*c100 + offset ;
	resI3 = res03 + iu2*c001 + iu1*c010 + iu0*c100 + offset ;
	r0 = u0 - iu0 ;
	r1 = u1- iu1 ;
	r2 = u2- iu2 ;
	(*(resI1)) -= (*I) * (1-r1) * (1-r2) ;
	(*(resI1 +c100)) += (*I)* (1-r1) * (1-r2) ;
	(*(resI1 +c010)) -= (*I) * r1 * (1-r2) ;
	(*(resI1+c110)) += (*I) * r1 * (1-r2) ;
	(*(resI1+c001)) -= (*I) * (1-r1) *  r2 ;
	(*(resI1+c101)) += (*I) * (1-r1) * r2 ;
	(*(resI1+c011)) -= (*I) * r1 * r2 ;
	(*(resI1+c111)) += (*I) * r0 * r1 * r2 ;

	(*(resI2)) -= (*I) * (1-r0) * (1-r2) ;
	(*(resI2 +c100)) -= (*I)* r0 * (1-r2) ;
	(*(resI2 +c010)) += (*I) *  (1-r0) * (1-r2) ;
	(*(resI2+c110)) += (*I) * r0 * (1-r2) ;
	(*(resI2+c001)) -= (*I) * (1-r0) *  r2 ;
	(*(resI2+c101)) -= (*I) * r0 * r2 ;
	(*(resI2+c011)) += (*I) *  (1-r0) * r2 ;
	(*(resI2+c111)) += (*I) * r0 * r2 ;

	(*(resI3)) -= (*I) * (1-r0)*(1-r1) ;
	(*(resI3 +c100)) -= (*I)* r0 * (1-r1) ;
	(*(resI3 +c010)) -= (*I) *  (1-r0) * r1 ;
	(*(resI3+c110)) -= (*I) * r0 * r1 ;
	(*(resI3+c001)) += (*I) * (1-r0)*(1-r1) ;
	(*(resI3+c101)) += (*I) * r0 * (1-r1) ;
	(*(resI3+c011)) += (*I) *  (1-r0) * r1 ;
	(*(resI3+c111)) += (*I) * r0 * r1 ;
	i0++ ;
	i1++ ;
	i2++ ;
      }
      res0[0].copy(res1) ;
      res0[1].copy(res2) ;
      res0[2].copy(res3) ;
    } else {
      cout << "dim = " << d.n << endl ;
      exit(-1) ;
    }
  }

  void multilinInterpNEW(const _VectorMap<NUM> &src, _VectorMap<NUM> &res0) const
  {
    _VectorMap<NUM> res;
    res.al(d) ;
    double prec = 1e-10 ;


    if (d.n  == 1) {
      _iterator I ;
      int offset = - d.getCumMin(), d0=d.getCum(0), m0 = d.getm(0), M0 = d.getM(0) ;
      int phi0, iu0 ;  ;
      _real r0, u0, ri0 ;
      c_iterator i0 =(*this)[0].begin() ;
      for(I=res[0].begin(); I!=res[0].end(); ++I) {
	ri0 = floor(*i0 / prec) * prec ;
	u0 = minV(maxV(ri0, m0), M0-prec) ;
	iu0 = (int) floor(u0) ;
	phi0 = iu0*d0 + offset ;
	r0 = u0 - iu0 ;
	(*I) = src[0][phi0] * (1-r0) + src[0][phi0+1] * r0 ;
	i0++ ;
      }
    }
    else if (d.n == 2) {
      _iterator I ;
      int offset = -d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0), m1 = d.getm(1), M1 = d.getM(1) ;
      int iu0, iu1 ;  ;
      _real r0, u0, r1, u1, ri0, ri1;
      int c10 = d.getCum(0), c01 = d.getCum(1), c11 = c10 + c01 ;
      for(unsigned int k=0; k<2; k++) {
	c_iterator src0 = src[k].begin(), srcI ;
	c_iterator i0 =(*this)[0].begin(), i1 =(*this)[1].begin() ;
	I = res[k].begin() ;
#ifdef _PARALLEL_
	int nb_chunks = 1 ; // omp_get_max_threads() ;
#pragma omp parallel  private(srcI,ri0,ri1,u0,u1,iu0,iu1,r0,r1) 
	//shared(I,src0,res,prec,M0,M1,offset,c01,c10,c11)
#pragma omp for
#else
	int nb_chunks = 1 ;
#endif
	for (int cc=0; cc<nb_chunks; cc++) {
	  int csz = (int) floor(res[k].length()/nb_chunks) ;
	  int cst = cc * csz ;
	  int cend ;
	  if (cc < nb_chunks-1)
	    cend = cst + csz -1 ;
	  else
	    cend = res[k].length() - 1 ;
	  NUM* tmp = new NUM[csz]  ;
	  for (int jj=cst; jj<= cend; jj++){
	    ri0 = floor((*this)[0][jj] / prec) * prec ;
	    ri1 = floor((*this)[1][jj] / prec) * prec ;
	    u0 = minV(maxV(ri0, m0), M0-prec) ;
	    u1 = minV(maxV(ri1, m1), M1-prec) ;
	    iu0 = (int) floor(u0) ;
	    iu1 = (int) floor(u1) ;
	    srcI = src0 + iu1*c01 + iu0*c10 + offset ;
	    /* 	  if (iu0 < 0 | iu1 < 0) */
	    /* 	    cout << m0 << " " << M0 << " " << m1 << " " << M1 <<" "<< *i0 << " " << *i1 << endl ; */
	    r0 = u0 - iu0 ;
	    r1 = u1-iu1 ;
	    tmp[jj-cst] = (*srcI) * (1-r0)*(1-r1) 
	      + (*(srcI +  c10)) * r0 * (1-r1)
	      + (*(srcI + c01)) * r1 * (1-r0)
	      + (*(srcI + c11)) * r1 * r0 ;
	    //	  i0++ ;
	    //	  i1++ ;
	    //	  ++I ;
	  }
	
	  for(int jj=cst; jj<=cend; jj++)
	    *(I+jj) = tmp[jj-cst] ;
	  delete [] tmp ;
	}
      }
    } 
    else if (d.n == 3) {
      _iterator I ;
      int offset = - d.getCumMin(), m0 = d.getm(0), M0 = d.getM(0), m1 = d.getm(1), M1 = d.getM(1), 
	m2 = d.getm(2), M2 = d.getM(2) ;
      int iu0, iu1, iu2 ;  ;
      _real r0, u0, r1, u1, r2, u2, ri0, ri1, ri2;
      int c100 = d.getCum(0), c010 = d.getCum(1), c001 = d.getCum(2),
	c110 = c100 + c010, c101 = c100 + c001, c011 = c010 + c001, 
	c111 = c110 + c001 ;

      for(unsigned int k=0; k<3; k++) {
	c_iterator src0 = src[k].begin(), srcI ;
	c_iterator i0 =(*this)[0].begin(), i1 =(*this)[1].begin(), i2 =(*this)[2].begin() ;
	for(I=res[k].begin(); I!=res[k].end(); ++I) {
	  ri0 = floor(*i0 / prec) * prec ;
	  ri1 = floor(*i1 / prec) * prec ;
	  ri2 = floor(*i2 / prec) * prec ;
	  u0 = minV(maxV(ri0, m0), M0-prec) ;
	  u1 = minV(maxV(ri1, m1), M1-prec) ;
	  u2 = minV(maxV(ri2, m2), M2-prec) ;
	  iu0 = (int) floor(u0) ;
	  iu1 = (int) floor(u1) ;
	  iu2 = (int) floor(u2) ;
	  srcI = src0 + iu2*c001 + iu1*c010 + iu0*c100 + offset ;
	  r0 = u0 - iu0 ;
	  r1 = u1- iu1 ;
	  r2 = u2- iu2 ;
	  *I = (*(srcI)) * (1-r0)*(1-r1) * (1-r2) 
	    + (*(srcI +c100)) * r0 * (1-r1) * (1-r2)
	    + (*(srcI +c010)) *  (1-r0) * r1 * (1-r2) 
	    + (*(srcI+c110)) * r0 * r1 * (1-r2)
	    + (*(srcI+c001)) * (1-r0)*(1-r1) *  r2 
	    + (*(srcI+c101)) * r0 * (1-r1) * r2
	    + (*(srcI+c011)) *  (1-r0) * r1 * r2 
	    + (*(srcI+c111)) * r0 * r1 * r2 ;
	  i0++ ;
	  i1++ ;
	  i2++ ;
	}
      }
    } else {
      cout << "dim = " << d.n << "?" << endl ;
      exit(-1) ;
    }
    res0.copy(res) ;
  }



  void multilinInterp(const _VectorMap<NUM> &src, std::vector<NUM> &phic, std::vector<NUM> &resc, Ivector &I, std::vector<_real> &r) const
  {
    std::vector<_real>::iterator ir =r.begin();
    typename std::vector<NUM>::iterator  ip = phic.begin(), irc = resc.begin() ;
    std::vector<int>::iterator ii=I.begin();
    for(unsigned int i=0; i<I.size(); i++, ++ii, ++ir, ++ip) {
      *ip = absMin((_real) absMax(*ip, (_real) d.getm(i)), d.getM(i)-0.00000001) ;
      *ii = (int) floor(*ip) ;
      *ir = *ip - *ii ; 
    }

    int i0 = d.position(I) ;
    for(irc = resc.begin(); irc != resc.end(); ++irc)
      *irc = 0 ;
    for (int k=0; k< twoPowerDim ; k++) {
      _real weight = 1 ;
      int j = 1, ii = i0 ;
      ir = r.begin() ;
      for (unsigned int i=0; i<phic.size(); i++, ++ir) {
	if (k & j) {
	  weight *= *ir ;
	  ii += d.getCum(i) ;
	}
	else {
	  weight *= 1 - *ir ;
	}
	j<<=1 ;
      }
      int c=0 ;
      for(irc = resc.begin(); irc != resc.end(); ++irc)
	*irc += weight * src[c++][ii] ; 
    }
  }

  /**
     multilinear interpolation
  */
  _real multilinInterp(const _Vector<NUM> &src, std::vector<NUM> &phic, Ivector &I, std::vector<_real> &r) const
  {
    unsigned int i = 0 ;
    std::vector<_real>::iterator ir =r.begin();
    typename vector<NUM>::iterator ip = phic.begin() ;
    for(std::vector<int>::iterator ii=I.begin(); ii != I.end(); ++ii, ++ir, ++i, ++ip) {
      *ip = absMin((_real) absMax(*ip, (_real) d.getm(i)), d.getM(i)-0.00000001) ;
      *ii = (int) floor(*ip) ;
      *ir = *ip - *ii ;
  }

  int i0 = src.d.position(I) ;
  NUM res = 0 ;
  for (int k=0; k< twoPowerDim ; k++) {
    _real weight = 1 ;
    int j = 1, ii = i0 ;
    ir =r.begin() ;
    for (unsigned int i=0; i<phic.size(); i++, ++ir) {
      if (k & j) {
	weight *= *ir ;
	ii += src.d.getCum(i) ;
      }
      else {
	weight *= 1 - *ir ;
      }
      j<<=1 ;
    }
    res += weight * src[ii] ; 
  }
  return res ; 
  }


  /**
     multilinear interpolation
  */
  void multilinInterpOLD(const _Vector<NUM> &src, _Vector<NUM> &res0) const
  {
    _Vector<NUM> res ;
    res.zeros(d) ;
    Ivector I ;
    vector<NUM> phic, r ;
    phic.resize(size()) ;
    I.resize(size()) ;
    r.resize(size()) ;

  for(unsigned int c =0; c<d.length; c++) {
    for (unsigned int j=0; j<size(); j++)
      phic[j] = (*this)[j][c] ;
    res[c] = multilinInterp(src, phic, I, r) ;
  }
  res0.copy(res) ;
  }



  /**
     dual of the multilinear interpolation
  */
  void multilinInterpDualOld(const _Vector<NUM> &src, _Vector<NUM> &res0) const
  {
    _Vector<NUM> res ;
    res.zeros(d) ;
    for (unsigned int c=0; c<res.size(); c++)
      res[c] = 0 ;

    Ivector I ;
    std::vector<_real> r ;
  
    I.resize(size()) ;
    r.resize(size()) ;
    for(unsigned int c =0; c<d.length; c++) {
      std::vector<_real>::iterator ir =r.begin() ;
      std::vector<int>::iterator ii=I.begin();
      for(unsigned int i=0; i<I.size(); i++, ++ii, ++ir) {
	*ii = (int) floor((*this)[i][c]) ;
	if (*ii < d.getm(i)) {
	  *ii = d.getm(i) ;
	  *ir = 0 ;
	}
	else if (*ii >= d.getM(i)) {
	  *ii = d.getM(i)-1 ;
	  *ir = 1 ;
	}
	else 
	  *ir = (*this)[i][c] - *ii ;
      }

      int i0 = src.d.position(I) ;
      for (int k=0; k< twoPowerDim ; k++) {
	_real weight = 1 ;
	int j = 1, ii = i0 ;
	ir = r.begin() ;
	for (unsigned int i=0; i<size(); i++, ++ir) {
	  if (k & j) {
	    weight *= *ir ;
	    ii += src.d.getCum(i) ;
	  }
	  else {
	    weight *= 1 - *ir ;
	  }
	  j <<= 1 ;
	}
	res[ii] += weight * src[c] ; 
      }
    }
    res0.copy(res) ;
  }

  _real multilinInterp(const _Vector<NUM> &src, std::vector<NUM> &phic) const
  {
    Ivector I ;
    std::vector<_real> r ;

    I.resize(phic.size()) ;
    r.resize(phic.size()) ;

    return multilinInterp(src, phic, I, r) ;
  }

  /** gradient computation and interpolation
   */
  void extractionGradient(const _Vector<NUM> &src, std::vector<NUM> &resc, const std::vector<NUM> &phic) const 
  {
    Ivector I, J, out ;
    std::vector<_real> r ;

    out.resize(phic.size()) ;
    I.resize(phic.size()) ;
    J.resize(phic.size()) ;
    r.resize(phic.size()) ;
    for(unsigned int i=0; i<I.size(); i++) {
      I[i] = (int) floor(phic[i]) ;
      out[i] = 0 ;
      if (I[i] < src.d.getm(i)) {
	I[i] = src.d.getm(i) ;
	r[i] = 0 ;
	out[i] = 1 ;
      }
      else if (I[i] >= src.d.getM(i)) {
	I[i] = src.d.getM(i)-1 ;
	r[i] = 1 ;
	out[i] = 1 ;
      }
      else 
	r[i] = phic[i] - I[i] ;
    }

    _real jumpThresh = 0.001 ;
    std::vector<int> eps ;
    eps.resize(phic.size()) ;
    resc.resize(phic.size()) ;
    for (unsigned int j=0; j<eps.size(); j++) {
      resc[j] = 0 ;
      J[j] = I[j] ;
      _real testw = 0 ; 
      for (unsigned int i=0; i<eps.size(); i++)
	eps[i] = 0 ;
      bool cont = true ;
      while (cont) {
	_real weight = 1 ;
	for (unsigned int i=0; i<eps.size(); i++) 
	  if (i != j) {
	    if (eps[i] == 0) {
	      weight *= 1 - r[i] ;
	      J[i] = I[i] ;
	    }
	    else {
	      weight *= r[i] ;
	      J[i] = I[i] + 1; 
	    }
	  }
      
	testw += weight ;
	int i0 = src.d.position(J) ;
	_real resi = src[i0] ;
	if (r[j] > jumpThresh && r[j] < 1 - jumpThresh) 
	  resc[j] += weight*(src[src.d.rPos(i0, j, 1)] - resi) ;
	else if (r[j] >= 1 -  jumpThresh) {
	  if (J[j] > d.getm(j)) {
	    _real u = (src[src.d.rPos(i0, j, 1)] -resi)* (resi - src[src.d.rPos(i0, j, -1)]) ;
	    if (u > 0)
	      resc[j] += weight*((src[src.d.rPos(i0, j, 1)] - src[src.d.rPos(i0, j, -1)])/2) ;
	  }
	  else if (out[j] == 0)
	    resc[j] += weight*(src[src.d.rPos(i0, j, 1)] -resi) ;
	}
	else {
	  if (J[j] <= d.getM(j)-2) {
	    _real u = (src[src.d.rPos(i0, j, 2)] - (src[src.d.rPos(i0, j, 1)])) 
	      * (src[src.d.rPos(i0, j, 1)] - resi) ;
	    if (u > 0)
	      resc[j] += weight*((src[src.d.rPos(i0, j, 2)] - resi)/2) ;
	  }
	  else if (out[j] == 0)
	    resc[j] += weight*(src[src.d.rPos(i0, j, 1)] - resi);
	}
	
	int k = eps.size() -1 ;
	while (k >= 0 && ((unsigned int) k==j || eps[k] == 1))
	  k-- ;
	if (k >= 0 && (unsigned int) k!= j) {
	  eps[k]=1 ;
	  for (unsigned int l=k+1; l<eps.size(); l++)
	    if (l!= j)
	      eps[l] = 0 ;
	}
	else 
	  cont = false ;
      }
    }
  }

  void extractionGradient(const _Vector<NUM> &src, _VectorMap<NUM> &res, const NUM mat[DIM_MAX][DIM_MAX+1]) const
  {
    affineMap(d, mat) ;
    extractionGradient(src, res) ;
  }

  void extractionGradient(const _Vector<NUM> &src, _VectorMap<NUM> &res) const
  {
    res.al(d) ;
    vector<NUM> phic, resc ;
    phic.resize(size()) ;
    resc.resize(size()) ;

    for(unsigned int c =0; c<d.length; c++) {
      for (unsigned int j=0; j<size(); j++)
	phic[j] = (*this)[j][c] ;
      extractionGradient(src, resc, phic) ;
      for (unsigned int j=0; j<size(); j++)
	res[j][c] = resc[j] ;
    }
  }



  void multilinInterpOLD(const _VectorMap<NUM> &src, _VectorMap<NUM> &res0) const
  {
    _VectorMap<NUM> res ;
    res.zeros(d) ;
    Ivector I ;
    vector<NUM> phic, resc;
    vector<_real> r ;
    phic.resize(size()) ;
    I.resize(size()) ;
    r.resize(size()) ;
    resc.resize(size()) ;

    for(unsigned int c =0; c<d.length; c++) {
      for (unsigned int j=0; j<size(); j++)
	phic[j] = (*this)[j][c] ;
      multilinInterp(src, phic, resc, I, r) ;
      for(unsigned int j=0; j<res.size(); j++)
	res[j][c] = resc[j] ;
    }
    res0.copy(res) ;
  }


  void rescale(const Domain &D, _VectorMap<NUM> &res) const {
    res.al(D) ; for(unsigned int i=0; i<size(); i++) (*this)[i].rescale(D, res[i]) ;}

  // arithmetic operations
  // VectorMap &operator = (const VectorMap &src){copy(src); return *this;}
  void operator += (const _VectorMap<NUM> &src) {for(unsigned int i=0; i<size(); i++) (*this)[i] += src[i]; }
  void operator -= (const _VectorMap<NUM> &src) {for(unsigned int i=0; i<size(); i++) (*this)[i] -= src[i]; }
  void operator *= (const _VectorMap<NUM> &src) {for(unsigned int i=0; i<size(); i++) (*this)[i] *= src[i]; }
  void operator *= (const _Vector<NUM> &src) {for(unsigned int i=0; i<size(); i++) (*this)[i] *= src; }
  void operator /= (const _Vector<NUM> &src) {for(unsigned int i=0; i<size(); i++) (*this)[i] /= src; }
  void operator += (NUM m) {for(unsigned int i=0; i<size(); i++) (*this)[i] += m; }
  void operator -= (NUM m) {for(unsigned int i=0; i<size(); i++) (*this)[i] -= m; }
  void operator *= (NUM m) {for(unsigned int i=0; i<size(); i++) (*this)[i] *= m; }
  void operator /= (NUM m) {for(unsigned int i=0; i<size(); i++) (*this)[i] /= m; }

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

    int foo = d.n ;
    ofs.write((char *) &foo, sizeof(int)) ;
    //ofs.close(); return ;
    for (unsigned int i=0; i<d.n; i++) {
      foo = d.getm(i) ; 
      ofs.write((char *) &foo, sizeof(int)) ;
    }

    for (unsigned int i=0; i<d.n; i++) {
      foo = d.getM(i) ; 
      ofs.write((char *) &foo, sizeof(int)) ;
    }

    for (unsigned int i=0; i<d.n; i++)
      ofs.write((char *) &((*this)[i][0]), d.length*sizeof(NUM)) ;
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
    
    int foo, N ;
    Ivector m, M ;
    ifs.read((char *) &N, sizeof(int)) ;
    m.resize(N) ;
    M.resize(N) ;
    for (int i=0; i<N; i++) {
      ifs.read((char *) &foo, sizeof(int)) ;
      m[i] = foo ;
    }

    for (int i=0; i<N; i++) {
      ifs.read((char *) &foo, sizeof(int)) ;
      M[i] = foo ;
    }

    d.create(m, M) ;
    al(d) ;

    for (int i=0; i<N; i++) 
      ifs.read((char *) &((*this)[i][0]), (*this)[i].size()*sizeof(NUM)) ;

    ifs.close() ;
  }

  /**
     save in binary format 
  */
  void write_grid(const char * path) const
  {
    if (d.n == 1) {
      Vector grid ;
      grid.copy((*this)[0]) ;
      grid.write_image(path) ;
    }
    else {
      _VectorMap id ;
      Vector idgrid, grid ;
      Ivector I ;
      double step = 5 ;
      d.putm(I) ;
      id.idMesh(d) ;
      idgrid.zeros(d) ;
      for (unsigned int i=0; i<d.length; i++)
	for (unsigned int k=0; k< d.n; k++) 
	  if (fabs(id[k][i] - step*floor(id[k][i]/step)) < 0.001)
	    idgrid[i] = 1 ;

      multilinInterp(idgrid, grid) ;
      for (unsigned int i=0; i<d.length; i++)
	grid[i] = 1-(grid[i]) ;
      grid.write_imagesc(path) ;
    }
  }
};



template <class NUM> void cropEmptyRegions(const _Vector<NUM> &src1, const _Vector<NUM> &src2, _Vector<NUM> &dest1, _Vector<NUM> &dest2, Domain &crDom)
{
  Ivector MIN, MAX, I, TMAX;
  src1.domain().putm(I) ;
  src1.domain().putM(MIN) ;
  src1.domain().putM(TMAX) ;
  src1.domain().putm(MAX) ;
  for(unsigned int i=0; i<src1.size(); i++) {
    if (fabs(src1[i]) > 1e-5 || fabs(src2[i]) > 1e-5)
      for (unsigned int k=0; k<I.size(); k++) {
	if (I[k] < MIN[k]) 
	  MIN[k] = I[k] ;
	if (I[k] > MAX[k])
	  MAX[k] = I[k] ;
      }
    src1.domain().inc(I) ;
  }
   
  for (unsigned int k=0; k<I.size(); k++) {
    MIN[k] -= 5 ;
    if (MIN[k] < 0)
      MIN[k] = 0 ;
    MAX[k] += 5 ;
    if (MAX[k] > TMAX[k])
      MAX[k] = TMAX[k] ;
    if (MAX[k] < MIN[k])
      MAX[k] = MIN[k] ;
  }
  Domain D(MIN, MAX) ;
  cout << "cropped domain " << endl << D << endl;
  crDom.copy(D) ;
  src1.crop(D, dest1) ;
  src2.crop(D, dest2) ;
}

/**
   action of an affine transformation (on the right)
*/
template <class NUM> void affineInterp(const _Vector<NUM> &src, _Vector<NUM> &res, const _real mat[DIM_MAX][DIM_MAX+1])
{
  _VectorMap<NUM> phi ;
  //double t0 ;
  //t0 = time(0) ;
  phi.affineMap(src.d, mat) ;
  //t0 = time(0) - t0 ;
  //cout << "affineMap " << t0 << endl ;
  phi.multilinInterp(src, res) ;
  //t0 = time(0) - t0 ;
  // cout << "interp "  << endl ;
}


template <class NUM>  void matrixProduct(const std::vector<_VectorMap<NUM> > &mat,  const _VectorMap<NUM> & src, _VectorMap<NUM> &res)
{
  res.zeros(src.d) ;
  for (unsigned int i=0; i<src.size(); i++) 
    for (unsigned int j=0; j< src.size(); j++) 
      //#pragma omp for schedule(static) ordered
      for (unsigned int c=0; c<src.d.length; c++)
	res[i][c] += mat[i][j][c] * src[j][c] ;
}

template <class NUM>  void matrixTProduct(const std::vector<_VectorMap<NUM> > &mat,  const _VectorMap<NUM> & src, _VectorMap<NUM> &res)
{
  res.zeros(src.d) ;
  for (unsigned int i=0; i<src.size(); i++) 
    for (unsigned int j=0; j< src.size(); j++) 
      //#pragma omp for  schedule(static) ordered
      for (unsigned int c=0; c<src.d.length; c++)
	res[i][c] += mat[j][i][c] * src[j][c] ;
}

template <class NUM>  void addProduct(const _VectorMap<NUM>  &src1, double a,  const _VectorMap<NUM> & src2, _VectorMap<NUM> &res)
{
  res.zeros(src2.d) ;
  //#pragma omp parallel
  for (unsigned int i=0; i<src2.size(); i++) 
       for (unsigned int c=0; c<src2.d.length; c++)
	res[i][c] = src1[i][c] + a * src2[i][c] ;
}

template <class NUM>  void addProduct(_VectorMap<NUM> &res, double a,  const _VectorMap<NUM> & src2)
{
  //#pragma omp parallel
  for (unsigned int i=0; i<src2.size(); i++) 
       for (unsigned int c=0; c<src2.d.length; c++)
	res[i][c] += a * src2[i][c] ;
}

template <class NUM>  void copyProduct(const NUM &a,  const _VectorMap<NUM> & src2, _VectorMap<NUM> &res)
{
  res.zeros(src2.d) ;
  //#pragma omp parallel
  for (unsigned int i=0; i<src2.size(); i++) 
       for (unsigned int c=0; c<src2.d.length; c++)
	res[i][c] = a * src2[i][c] ;
}

    

template <class NUM> void gradient(const _Vector<NUM> &src, _VectorMap<NUM> &res, vector<_real> &resol)
{
    res.al(src.d) ;
    Ivector I ;
    src.d.putm(I)  ;

    for(unsigned int c =0; c< src.length(); c++) {
      for(unsigned int j=0; j<res.size(); j++)
	if (I[j] > src.d.getm(j) && I[j] < src.d.getM(j)) 
      	  res[j][c] = (src[src.d.rPos(c, j, 1)] - src[src.d.rPos(c, j, -1)]) / (2*resol[j]) ;
	else if (I[j] == src.d.getm(j))  
      	  res[j][c] = (src[src.d.rPos(c, j, 1)] - src[c])/resol[j]  ;
	else if (I[j] == src.d.getM(j))  
      	  res[j][c] = (src[c] - src[src.d.rPos(c, j, -1)])/resol[j]  ;
      src.d.inc(I) ;
    }
}

template <class NUM> void gradientPlus(const _Vector<NUM> &src, _VectorMap<NUM> &res, vector<_real> &resol)
{
    res.al(src.d) ;
    Ivector I ;
    src.d.putm(I)  ;

    for(unsigned int c =0; c< src.length(); c++) {
      for(unsigned int j=0; j<res.size(); j++)
	if (I[j] < src.d.getM(j)) 
	  res[j][c] = (src[src.d.rPos(c, j, 1)] - src[c]) / resol[j] ;
	else 
	  res[j][c] = (src[c] - src[src.d.rPos(c, j, -1)])/resol[j]  ;
      src.d.inc(I) ;
    }
}

template <class NUM> void gradientMinus(const _Vector<NUM> &src, _VectorMap<NUM> &res, vector<_real> &resol)
{
    res.al(src.d) ;
    Ivector I ;
    src.d.putm(I)  ;

    for(unsigned int c =0; c< src.length(); c++) {
      for(unsigned int j=0; j<res.size(); j++)
	if (I[j] > src.d.getm(j)) 
	  res[j][c] = (src[c] - src[src.d.rPos(c, j, -1)]) / resol[j] ;
	else 
	  res[j][c] = (src[src.d.rPos(c, j, 1)] - src[c])/resol[j]  ;
      src.d.inc(I) ;
    }
}



template <class NUM> void divergence(const _VectorMap<NUM> &src, _Vector<NUM> &res, vector<_real> &resol)
{
    res.al(src.d) ;
    Ivector I ;
    src.d.putm(I)  ;

    for(unsigned int c =0; c< res.length(); c++) {
      res[c] = 0 ;
      for(unsigned int j=0; j<src.size(); j++)
	if (I[j] > src.d.getm(j) && I[j] < src.d.getM(j)) 
	  res[c] += (src[j][src.d.rPos(c, j, 1)] - src[j][src.d.rPos(c, j, -1)]) / (2) ;
	else if (I[j] == src.d.getm(j))  
	  res[c] += (src[j][src.d.rPos(c, j, 1)] - src[j][c])  ;
	else if (I[j] == src.d.getM(j))  
	  res[c] += (src[j][c] - src[j][src.d.rPos(c, j, -1)])  ;
      src.d.inc(I) ;
    }
}


typedef _VectorMap<_real> VectorMap;

/**
   Time dependent vector
*/
class TimeVector : public std::vector<Vector>
{
public:
  Domain d ;
  void resize(int s, const Domain &d_) {
    std::vector<Vector>::resize(s) ; for(unsigned int i=0; i<size(); i++) (*this)[i].al(d_) ; d.copy(d_) ;}
  void zeros(int s, const Domain &d_) {
    std::vector<Vector>::resize(s) ; for(unsigned int i=0; i<size(); i++) (*this)[i].zeros(d_) ; d.copy(d_) ;}
  void copy(const TimeVector &src) { resize(src.size(), src.d); d.copy(src.d); for(unsigned int i=0; i<size(); i++) (*this)[i].copy(src[i]) ;};
  unsigned int length() {return size() * d.length ;}

  /**
     sum of squares
  */
  _real norm2() const
  {
    _real res = 0 ;
    for (unsigned int i=0; i<size(); i++)
      res += (*this)[i].norm2() ;
    return res ;
  }

  _real norm2(unsigned int i1, unsigned int i2) const {
    double res = 0 ;
    for (unsigned int i=i1; i<=i2; i++)
      res += (*this)[i].norm2() ;
    return res ;
  }

  void operator += (TimeVector &src) {for(unsigned int i=0; i<size(); i++) (*this)[i] += src[i]; }
  void operator -= (TimeVector &src) {for(unsigned int i=0; i<size(); i++) (*this)[i] -= src[i]; }
  void operator *= (TimeVector &src) {for(unsigned int i=0; i<size(); i++) (*this)[i] *= src[i]; }
  void operator *= (Vector &src) {for(unsigned int i=0; i<size(); i++) (*this)[i] *= src; }
  void operator += (_real m) {for(unsigned int i=0; i<size(); i++) (*this)[i] += m; }
  void operator -= (_real m) {for(unsigned int i=0; i<size(); i++) (*this)[i] -= m; }
  void operator *= (_real m) {for(unsigned int i=0; i<size(); i++) (*this)[i] *= m; }
  void operator /= (_real m) {for(unsigned int i=0; i<size(); i++) (*this)[i] /= m; }
  _real sumScalProd(const TimeVector& tv, int i1, int i2) const {
    _real res = 0 ; for(int i=i1; i<=i2; i++) res += (*this)[i].sumProd(tv[i]) ; return res ;}
  _real sumScalProd(const TimeVector& tv) const {
      _real res = 0 ; for(unsigned int i=0; i<size(); i++) res += (*this)[i].sumProd(tv[i]) ; return res ;}
};

/**
   time dependent vector field and diffeomorphism
*/
class TimeVectorMap : public std::vector<VectorMap>
{
public:
  Domain d ;
  void resize(int s){std::vector<VectorMap>::resize(s) ;}
  void resize(int s, const Domain &d_) {
    std::vector<VectorMap>::resize(s) ; for(unsigned int i=0; i<size(); i++) (*this)[i].al(d_) ; d.copy(d_) ;}
  void zeros(int s, const Domain &d_) {
    std::vector<VectorMap>::resize(s) ; for(unsigned int i=0; i<size(); i++) (*this)[i].zeros(d_) ; d.copy(d_) ;}
  void copy(const TimeVectorMap &src) 
  { 
    resize(src.size(), src.d); 
    d.copy(src.d); 
    for(unsigned int i=0; i<size(); i++) {
      (*this)[i].copy(src[i]) ;
    }
  }
  unsigned int length() const {return size()*d.length ;}
  void operator += (TimeVectorMap &src) {for(unsigned int i=0; i<size(); i++) (*this)[i] += src[i]; }
  void operator -= (TimeVectorMap &src) {for(unsigned int i=0; i<size(); i++) (*this)[i] -= src[i]; }
  void operator *= (TimeVectorMap &src) {for(unsigned int i=0; i<size(); i++) (*this)[i] *= src[i]; }
  void operator += (_real m) {for(unsigned int i=0; i<size(); i++) (*this)[i] += m; }
  void operator -= (_real m) {for(unsigned int i=0; i<size(); i++) (*this)[i] -= m; }
  void operator *= (_real m) {for(unsigned int i=0; i<size(); i++) (*this)[i] *= m; }
  void operator /= (_real m) {for(unsigned int i=0; i<size(); i++) (*this)[i] /= m; }
  _real norm2() const {
    double res = 0 ;
    for (unsigned int i=0; i<size(); i++)
      res += (*this)[i].norm2() ;
    return res ;
  }

  _real norm2(unsigned int i1, unsigned int i2) const {
    double res = 0 ;
    for (unsigned int i=i1; i<=i2; i++)
      res += (*this)[i].norm2() ;
    return res ;
  }
};

// Fourier transform
void fourn(std::vector<_real>::iterator data, std::vector<int> nn, int isign) ;

#endif





