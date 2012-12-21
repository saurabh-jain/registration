#include "ImageMatching.h"
#ifndef _IMAGEMATCHINGAFFINE_
#define _IMAGEMATCHINGAFFINE_

#ifndef DIM_MAX
#define DIM_MAX 5 
#endif

static int _affine_time=10 ;
class BasicImageAffineEnergy{
public:
  virtual _real operator() (const Matrix& gamma) const {
    cout << "not implement in BasicImageAffineEnergy" << endl;
    return -1 ;
  }
  Vector Tp, Tg;
  Ivector MIN ;
  std::vector<_real> x0 ;
  std::vector<_real> sz ;
  virtual ~BasicImageAffineEnergy(){};
};






class ImageAffineEnergy: public BasicImageAffineEnergy{
public:
  void init(Vector &TP, Vector &TG, std::vector<_real> &SZ, std::vector<_real> &X0, Ivector &offset) {
    Tp.copy(TP) ;
    Tg.copy(TG) ;
    TpNorm = TP.norm2() ;
    TgNorm = TG.norm2() ;
    sz.resize(SZ.size()) ;
    for(unsigned int k=0; k<sz.size(); k++)
      sz[k] = SZ[k]  ;
    x0.resize(X0.size()) ;
    for(unsigned int k=0; k<x0.size(); k++)
      x0[k] = X0[k]  ;
    MIN.copy(offset) ;
  }
  _real operator() (const Matrix& gamma) const {
    Vector DI ;
    Matrix AT ;
    unsigned int N = gamma.nColumns() -1 ;
    _real det = gamma.integrate(AT, _affine_time), sqdet = sqrt(det) ;

    _real mat[DIM_MAX][DIM_MAX+1] ;
    for (unsigned int i=0; i<N; i++)
      for (unsigned int j=0; j<=N; j++)
	mat[i][j] = AT(i, j) ;

    affineInterp(Tg, DI, mat) ;

    _real res = (TpNorm * sqdet + TgNorm/sqdet - 2* DI.sumProd(Tp)*sqdet) /DI.length() ;
    return res ;
  }
  virtual ~ImageAffineEnergy(){};
  double TpNorm, TgNorm;
};


class ImageSegmentationEnergy: public BasicImageAffineEnergy{
public:
  void init(Vector &TP, Vector &TG, Vector &PS, std::vector<_real> &SZ, std::vector<_real> &X0, Ivector &offset, 
	    double lambda) {
    Tp.copy(TP) ;
    Tg.copy(TG);
    Ps.copy(PS) ;
    rho = 1/lambda ;
    sz.resize(SZ.size()) ;
    for(unsigned int k=0; k<sz.size(); k++)
      sz[k] = SZ[k]  ;
    x0.resize(X0.size()) ;
    for(unsigned int k=0; k<x0.size(); k++)
      x0[k] = X0[k]  ;
    MIN.copy(offset) ;
  }
  _real operator() (const Matrix& gamma0) const {
    Vector DI ;
    Matrix AT, gamma ;
    gamma.copy(gamma0);
    gamma *= -1 ;
    unsigned int N = gamma.nColumns() -1 ;
    gamma.integrate(AT, _affine_time) ;

    _real mat[DIM_MAX][DIM_MAX+1] ;
    for (unsigned int i=0; i<N; i++)
      for (unsigned int j=0; j<=N; j++)
	mat[i][j] = AT(i, j) ;

    affineInterp(Tp, DI, mat) ;
    Vector foo, foo2 ;
    foo.copy(Ps) ;
    foo *= rho ;
    foo += 1 ; 
    foo *= DI ;

    _real res = (foo.sumProd(DI) - 2* DI.sumProd(Tg)) /DI.length() ;
    return res ;
  }
  Vector Ps;
  double rho ;
  virtual ~ImageSegmentationEnergy(){};
};

#endif
