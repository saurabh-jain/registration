/**
   affineReg. h
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

#include "matrix.h"
/**
   affine registration between images; multigrid local exploration
*/
template <class AE>
class affineRegistration
 {
public:
  bool tryGradient ;
  affineRegistration(){tryGradient = true ;}

  _real localSearch(Matrix &gamma, std::vector<_real> &Tsize, std::vector<_real> &midPoint, int nb_iterAff, int type_group, AE &enerAff) {
    //    gamma.copy(gamma0) ;
    unsigned int N = gamma.nRows() - 1 ;
    
    unsigned int nbDir=0 ;
    int it = 0 ;
    _real step = 1, coeff = 0.5, gradientStep = 10, bestEn ;
    bool cont = true ;
    Matrix gammaOld, tmpMat,  gradMat ;
    std::vector<_real> energies ;
    _real en, oldEn ;
    std::vector<Matrix> Directions ;

    if (type_group == param_matching::TRANSLATION) {
      nbDir = N ;
      Directions.resize(nbDir) ;
      for(unsigned int i=0; i<Directions.size(); i++) {
	Directions[i].zeros(N+1, N+1) ;
	Directions[i](i, N) = Tsize[i] ;
      }
    } 
    else   if (type_group == param_matching::ROTATION) {
      nbDir = N + (N*(N-1))/2 ;
      Directions.resize(nbDir) ;
      int k=0 ;
      _real u = 1/sqrt(2) ;
      for(unsigned int i=0; i<N; i++) 
	for (unsigned int j=i+1; j<N; j++) {
	  Directions[k].zeros(N+1, N+1) ;
	  Directions[k](i, j) = u ;
	  Directions[k](j, i) = -u ;
	  Directions[k](i, N) = -u*midPoint[j] ;
	  Directions[k](j, N) = u*midPoint[i] ;
	  k++ ;
	}
      for(unsigned int i=0; i<N; i++) {
	Directions[k].zeros(N+1, N+1) ;
	Directions[k](i, N) = Tsize[i]/10 ;
	k++ ;
      }
    }
    else   if (type_group == param_matching::SIMILITUDE) {
      nbDir = N + (N*(N-1))/2 + 1;
      Directions.resize(nbDir) ;
      int k=0 ;
      _real u = 1/sqrt(2), v= 1/sqrt((_real)N) ;
      for(unsigned int i=0; i<N; i++) 
	for (unsigned int j=i+1; j<N; j++) {
	  Directions[k].zeros(N+1, N+1) ;
	  Directions[k](i, j) = u ;
	  Directions[k](j, i) = -u ;
	  Directions[k](i, N) = -u*midPoint[j] ;
	  Directions[k](j, N) = u*midPoint[i] ;
	  k++ ;
	}
      Directions[k].zeros(N+1, N+1) ;
      for(unsigned int i=0; i<N; i++) {
	Directions[k](i, i) = v ;
	Directions[k](i, N) = -v*midPoint[i] ;
      }
      k++ ;
      for(unsigned int i=0; i<N; i++) {
	Directions[k].zeros(N+1, N+1) ;
	Directions[k](i, N) = Tsize[i]/10 ;
	k++ ;
      }
    }
    else   if (type_group == param_matching::GENERAL) {
      nbDir = N + N*N;
      Directions.resize(nbDir) ;
      int k=0 ;
      for(unsigned int i=0; i<N; i++) 
	for (unsigned int j=0; j<N; j++) {
	  Directions[k].zeros(N+1, N+1) ;
	  Directions[k](i, j) = 1 ;
	  Directions[k](i,N) = -midPoint[j] ; 
	  k++ ;
	}
      for(unsigned int i=0; i<N; i++) {
	Directions[k].zeros(N+1, N+1) ;
	Directions[k](i, N) = Tsize[i]/10 ;
	k++ ;
      }
    }
    else   if (type_group == param_matching::SPECIAL) {
      nbDir = N + N*N - 1;
      Directions.resize(nbDir) ;
      int k=0 ;
      _real u= 1/sqrt(2.0) ;
      for(unsigned int i=0; i<N; i++) 
	for (unsigned int j=0; j<N; j++) 
	  if (i != j) {
	    Directions[k].zeros(N+1, N+1) ;
	    Directions[k](i, j) = 1 ;
	    Directions[k](i,N) = -midPoint[j] ; 
	    k++ ;
	  }
      for(unsigned int i=1; i<N; i++) {
	Directions[k].zeros(N+1, N+1) ;
	Directions[k](0, 0) = u ;
	Directions[k](i, i) = -u ;
	Directions[k](0, N) = -u*midPoint[0] ;
	Directions[k](i, N) = u*midPoint[i] ;
	k++ ;
      }
      for(unsigned int i=0; i<N; i++) {
	Directions[k].zeros(N+1, N+1) ;
	Directions[k](i, N) = Tsize[i]/10 ;
	k++ ;
      }
      //cout << k << " " << nbDir << endl; 
    }


    std::vector<_real>    grad, oldGrad, xGamma, xGammaOld, xGammaBest, diffxGamma, diffGrad, p ;
    grad.resize(nbDir) ;
    oldGrad.resize(nbDir) ;
    xGamma.resize(nbDir) ;
    xGammaBest.resize(nbDir) ;
    xGammaOld.resize(nbDir) ;
    diffxGamma.resize(nbDir) ;
    diffGrad.resize(nbDir) ;
    p.resize(nbDir) ;

    for (unsigned int k=0; k<nbDir; k++) {
      xGamma[k] = 0 ; 
    }

  
    energies.resize(2*nbDir + 1) ;
    en = enerAff(gamma) ;
    oldEn = en ;

    cout << "initial energy " << en << endl ;
    unsigned int istep = 0 ;
    std::vector<_real> stepSequence ;
    stepSequence.resize(1) ;
    stepSequence[0] = 0.5;// *(Tp.domain().minWidth()/100.0);
    while (stepSequence[istep] > 0.005) {
      istep++ ;
      stepSequence.resize(istep+1) ;
      stepSequence[istep] = stepSequence[istep-1]*coeff ;
    }
    while (stepSequence[istep]  < 1) {
      istep++ ;
      stepSequence.resize(istep+1) ;
      stepSequence[istep] = stepSequence[istep-1]/coeff ;
    }
    while (stepSequence[istep] > 0.005) {
      istep++ ;
      stepSequence.resize(istep+1) ;
      stepSequence[istep] = stepSequence[istep-1]*coeff ;
    }


    for (unsigned int i=0; i<nbDir; i++)
      oldGrad[i] = 0 ;
    Matrix H ;
    H.eye(nbDir) ;
    istep = 0 ;
    while(istep < stepSequence.size()) {
      gradientStep = 10 ;
      step = stepSequence[istep] ;
      cont = true ;
      it = 0 ;
      while(cont && it < nb_iterAff) {
	it ++ ;
	cont = false ;
	bestEn = en ;
	gammaOld.copy(gamma) ;
	for (unsigned int i=0; i<nbDir; i++)
	  xGammaOld[i] = xGamma[i] ;
	for (unsigned int i=0 ; i<nbDir; i++) {
	  gamma.copy(Directions[i]) ;
	  gamma *= step ;
	  gamma += gammaOld ;
	  energies[2*i] = enerAff(gamma) ;
	  if (energies[2*i] < bestEn){
	    bestEn = energies[2*i] ;
	    for (unsigned int j=0; j<nbDir ; j++)
	      xGammaBest[j] = xGammaOld[j] ;
	    xGammaBest[i] += step ;
	    if (energies[2*i] < 0.9999 * en)
	      cont = true ;
	  }
	      
	  gamma.copy(Directions[i]) ;
	  gamma *= -step ;
	  gamma += gammaOld ;
	  energies[2*i+1] = enerAff(gamma) ;
	  if (energies[2*i+1] < bestEn){
	    bestEn = energies[2*i+1] ;
	    for (unsigned int j=0; j<nbDir ; j++)
	      xGammaBest[j] = xGammaOld[j] ;
	    xGammaBest[i] -= step ;
	    if (energies[2*i+1] < 0.9999 * en)
	      cont = true ;
	  }
	  grad[i] = (energies[2*i] - energies[2*i+1])/(step*2) ;
	}

	if (cont == true) {
	  en = bestEn ;
	  cout << "Affine step= " << step << " Energy: " << en << " (" << oldEn << ")" << endl ;
	  oldEn = en ;
	  gamma.zero() ;
	  for (unsigned int j=0; j<nbDir ; j++) {
	    xGamma[j] = xGammaBest[j] ;
	    tmpMat.copy(Directions[j]) ;
	    tmpMat *= xGamma[j] ;
	    gamma += tmpMat ;
	  }
	}
	else {
	    cout << "Affine step= " << step << " Energy: " << en << endl ;
	  istep ++ ;
	  gamma.copy(gammaOld) ;
	} 
	if (it >= nb_iterAff) 
	  istep ++ ;
      }
    }

    if (tryGradient) 
      cont = true ;
    it  = 0 ;
    while (cont && it<nb_iterAff) {
      it ++ ;
      gammaOld.copy(gamma) ;
      for (unsigned int i=0; i<nbDir; i++)
	xGammaOld[i] = xGamma[i] ;
      for (unsigned int i=0 ; i<nbDir; i++) {
	gamma.copy(Directions[i]) ;
	gamma *= step ;
	gamma += gammaOld ;
	energies[2*i] = enerAff(gamma) ;
	      
	gamma.copy(Directions[i]) ;
	gamma *= -step ;
	gamma += gammaOld ;
	energies[2*i+1] = enerAff(gamma) ;
	  
	grad[i] = (energies[2*i] - energies[2*i+1])/(step*2) ;
      }
      _real sqGrad = 0 ;
      for(unsigned int i=0; i<nbDir; i++) {
	diffGrad[i] = grad[i] - oldGrad[i] ;
	oldGrad[i] = grad[i] ;
	sqGrad += grad[i] * grad[i] ;
      }
      sqGrad += 1e-10;
      sqGrad = sqrt(sqGrad) ;
      H.apply(grad, p) ;
      gradMat.zeros(N+1, N+1) ;
      for (unsigned int i=0; i<nbDir; i++) {
	xGammaOld[i] = xGamma[i] ;
	tmpMat.copy(Directions[i]) ;
	tmpMat *= p[i] ;
	gradMat += tmpMat ;
      }
      gamma.copy(gradMat) ;
      _real alpha = gradientStep/sqGrad ;
      gamma *= - alpha;
      gamma += gammaOld ;
      energies[2*nbDir] = enerAff(gamma) ;
      _real gradp = 0 ;
      for(unsigned int i=0; i<nbDir; i++)
	gradp += p[i] * grad[i] ;
	
      while (energies[2*nbDir] > en - 0.0001*alpha *gradp && gradientStep > 0.0000000001){
	gradientStep /= 1.5 ;
	gamma.copy(gradMat) ;
	alpha = gradientStep/sqGrad ;
	gamma *= -alpha;
	gamma += gammaOld ;
	energies[2*nbDir] = enerAff(gamma) ;
      }
      gradientStep *= 1.5 ;
      if (energies[2*nbDir] < en) {
	cout << "gradient descent energy: " << energies[2*nbDir] << endl ;
	en = energies[2*nbDir] ;
	for(unsigned int i=0; i<nbDir; i++) {
	  xGamma[i] -= p[i]* alpha ;
	  diffxGamma[i] = xGamma[i] - xGammaOld[i] ;
	}  
	_real rho = 0 ;
	for (unsigned int i=0; i<nbDir; i++)
	  rho += diffxGamma[i] * diffGrad[i] ;
	rho = 1/rho ;
	Matrix Z, newH ;
	Z.resize(nbDir, nbDir) ;
	for (unsigned int i=0; i<nbDir; i++)
	  for (unsigned int j=0; j<nbDir; j++)
	    Z(i,j) = -rho * diffxGamma[i] * diffGrad[j] ;
	for (unsigned int i=0; i<nbDir; i++)
	  Z(i,i) += 1 ;
	newH.matProduct(Z, H) ;
	H.matProductT(newH, Z) ;
	for (unsigned int i=0; i<nbDir; i++)
	  for (unsigned int j=0; j<nbDir; j++)
	    H(i,j) += rho * diffxGamma[i] * diffxGamma[j] ;
	//      H.Print() ;
      }
      else 
	cont = false ;
    }

    return en ;
  }
} ;
