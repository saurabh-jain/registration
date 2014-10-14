##EDIT THE FOLLOWING LINES ACCORDING TO YOUR SYSTEM CONFIGURATION
CC = g++

## where the binary directory will be created to store executables after make install
PROJECT=.

## include paths for fft3, magick++ and gsl
FFTW3_IPATH = -I/usr/local/include
IMAGICK_IPATH = -I/usr/local/include
GSL_IPATH = -I/usr/local/include/gsl

## library paths for fft3, magick++ and gsl
FFTW3_LPATH = -L/usr/local/lib
IMAGICK_LPATH = -L/usr/local/lib
GSL_LPATH = -L/usr/local/lib


## In principle, no change should be needed beyond this line
ARRAY_IPATH = -I$(PROJECT)/arrayClasses 
IPATH = $(ARRAY_IPATH) $(GSL_IPATH) $(FFTW3_IPATH) $(JAMA_IPATH) $(IMAGICK_IPATH)   
LPATH = $(GSL_LPATH) $(FFTW3_LPATH) $(IMAGICK_LPATH)   
CFLAGS =  -O3 -ffast-math -Wall 
#-arch x86_64 
#COMPILE.cpp = $(CC) $(IPATH)  $(CFLAGS) -c
COMPILE.cpp = $(CC) $(ARRAY_IPATH) `pkg-config --cflags fftw3 Magick++ gsl` -c
LIBPATH = -lfftw3 -lMagick++ -lstdc++ -lgsl -lgslcblas
LIBS = `pkg-config --cflags --libs fftw3 Magick++ gsl`
COMPILE.o = $(CC)  $(LIBS) 
#COMPILE.o = $(CC) $(IPATH)  $(CFLAGS) 

DIR.h = \
	arrayClasses \

SOURCES.h = \
	affineReg.h \
	arrayClasses/byteSwap.h \
	deformableObject.h \
	deformableImage.h \
	diffeomorphism.h \
	greedyImage.h \
	ImageEvolution.h \
	ImageMatching.h \
	ImageMatchingAffine.h \
	arrayClasses/Ivector.h \
	kernels.h \
	landmarkMatching.h \
	matchingBase.h \
	arrayClasses/matrix.h \
	morphingNew.h \
	optim.h \
	optimF.h\
	pointSet.h \
	pointSetEvolution.h \
	pointSetFunctors.h \
	pointSetMatching.h \
	PointSetMatchingAffine.h \
	param_matching.h \
	shooting.h \
	tensorField.h \
	arrayClasses/VectorBase.h\
	Vector.h \
	velocity.h


SOURCES.cpp = \
	deformableObjects.cpp \
	landmarkMatching.cpp \
	param_matching.cpp \
	pointSet.cpp \
	PointSetVectorTransforms.cpp \
	shooting.cpp \
	shootingSEG.cpp \
	shootingSym.cpp \
	velocity.cpp 

MAINS.cpp = \
	affine_registration.cpp \
	greedyImageMatching.cpp \
	initialMomentumLandmarkMatching.cpp \
	initialMomentumMatching.cpp \
	initialMomentumMatchingSYM.cpp \
	initialMomentumSegmentation.cpp \
	metamorphosis.cpp \
	momentumAverage.cpp \
	momentumCovMatrix.cpp \
	parallelTranslation.cpp \
	printDefaultParam.cpp \
	procrusteanAverage.cpp \
	shoot.cpp \
	tensorCurvature.cpp \
	velocityMatching.cpp 


MAINS = $(MAINS.cpp:%.cpp=%)
SOURCES.o = $(SOURCES.cpp:%.cpp=%.o)

#%.d: %.cpp
#	$(SHELL) -ec '$(CC) -M  $(IPATH)  $(CFLAGS) $< | sed '\"s/$*.o/& $@/g'\" > $@'
%.d: %.cpp $(SOURCES.h)
	$(SHELL) -ec '$(CC) -M  $(IPATH)  $(CFLAGS) $< > $@'

included: 
	include $(SOURCES.cpp:.cpp=.d)
	include $(MAINS.cpp:.cpp=.d)


%.o: %.cpp 
	$(COMPILE.cpp) $<

all: metamorphosis affine_registration initialMomentumMatching initialMomentumMatchingSYM initialMomentumSegmentation velocityMatching parallelTranslation procrusteanAverage momentumAverage shoot momentumCovMatrix initialMomentumLandmarkMatching printDefaultParam

preinstall:
	mkdir -p $(PROJECT)/include  
	cp -R ./arrayClasses $(PROJECT)/include

procrusteanAverage: $(SOURCES.o) procrusteanAverage.o
	$(COMPILE.o) procrusteanAverage.o -o procrusteanAverage $(SOURCES.o) 
printDefaultParam: $(SOURCES.o) printDefaultParam.o
	$(COMPILE.o) printDefaultParam.o -o printDefaultParam $(SOURCES.o) 
momentumAverage: $(SOURCES.o) momentumAverage.o
	$(COMPILE.o) momentumAverage.o -o momentumAverage $(SOURCES.o)
parallelTranslation: $(SOURCES.o) parallelTranslation.o $(SOURCES.h)
	$(COMPILE.o) parallelTranslation.o -o parallelTranslation $(SOURCES.o) 
metamorphosis: $(SOURCES.o) metamorphosis.o
	$(COMPILE.o) metamorphosis.o -o metamorphosis $(SOURCES.o) 
initialMomentumLandmarkMatching: $(SOURCES.o) initialMomentumLandmarkMatching.o
	$(COMPILE.o) initialMomentumLandmarkMatching.o -o initialMomentumLandmarkMatching $(SOURCES.o)
initialMomentumMatching: $(SOURCES.o) initialMomentumMatching.o
	$(COMPILE.o) initialMomentumMatching.o -o initialMomentumMatching $(SOURCES.o) 
initialMomentumMatchingSYM: $(SOURCES.o) initialMomentumMatchingSYM.o
	$(COMPILE.o) initialMomentumMatchingSYM.o -o initialMomentumMatchingSYM $(SOURCES.o) 
initialMomentumSegmentation: $(SOURCES.o) initialMomentumSegmentation.o
	$(COMPILE.o) initialMomentumSegmentation.o -o initialMomentumSegmentation $(SOURCES.o) 
velocityMatching: $(SOURCES.o) velocityMatching.o
	$(COMPILE.o) velocityMatching.o -o velocityMatching $(SOURCES.o)
greedyImageMatching: $(SOURCES.o) greedyImageMatching.o
	$(COMPILE.o) greedyImageMatching.o -o greedyImageMatching $(SOURCES.o) 
matchShootMG: $(SOURCES.o) matchShootMG.o
	$(COMPILE.o) matchShootMG.o -o matchShootMG $(SOURCES.o) 
shootingBatch: $(SOURCES.o) shootingBatch.o
	$(COMPILE.o) shootingBatch.o -o shootingBatch $(SOURCES.o) 
affine_registration: $(SOURCES.o) affine_registration.o
	$(COMPILE.o) affine_registration.o -o affine_registration $(SOURCES.o) 
testKernel: $(SOURCES.o) testKernel.o
	$(COMPILE.o) testKernel.o -o testKernel $(SOURCES.o) 
shoot: $(SOURCES.o) shoot.o
	$(COMPILE.o) shoot.o -o shoot $(SOURCES.o)
momentumCovMatrix: $(SOURCES.o) momentumCovMatrix.o
	$(COMPILE.o) momentumCovMatrix.o -o momentumCovMatrix $(SOURCES.o) 
tensorCurvature: $(SOURCES.o) tensorCurvature.o
	$(COMPILE.o) tensorCurvature.cpp -o tensorCurvature $(SOURCES.o) 

doc:
	doxygen matching.dox

clean:
	rm -f *.o *.d

tar: $(SOURCES.cpp) $(SOURCES.h)
	tar cvzf matching.tgz $(DIR.h) $(SOURCES.cpp) $(SOURCES.h) $(MAINS.cpp) Makefile.template matching.dox README.pdf

