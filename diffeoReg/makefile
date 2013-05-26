##EDIT THE FOLLOWING LINES ACCORDING TO YOUR SYSTEM CONFIGURATION
CC = gcc

## where the binary directory will be created to store executables after make install
PROJECT=.

## include paths for fft3, magick++ and gsl
FFTW3_IPATH = -I/opt/local/include
IMAGICK_IPATH = -I/opt/local/include/ImageMagick
GSL_IPATH = -I/opt/local/include/gsl

## library paths for fft3, magick++ and gsl
FFTW3_LPATH = -L/opt/local/lib
IMAGICK_LPATH = -L/opt/local/lib
GSL_LPATH = -L/opt/local/lib


## In principle, no change should be needed beyond this line
ARRAY_IPATH = -I$(PROJECT)/arrayClasses 
IPATH = $(ARRAY_IPATH) $(GSL_IPATH) $(FFTW3_IPATH) $(JAMA_IPATH) $(IMAGICK_IPATH)   
LPATH = $(GSL_LPATH) $(FFTW3_LPATH) $(IMAGICK_LPATH)   
CFLAGS =  -O3 -ffast-math -Wall 
#-arch x86_64 
COMPILE.cpp = $(CC) $(IPATH)  $(CFLAGS) -c
LIBPATH = -lfftw3 -lMagick++ -lstdc++ -lgsl -lgslcblas
COMPILE.o = $(CC) $(IPATH)  $(CFLAGS) 

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
	$(COMPILE.o) procrusteanAverage.o -o procrusteanAverage $(SOURCES.o) $(LPATH)  $(LIBPATH)
printDefaultParam: $(SOURCES.o) printDefaultParam.o
	$(COMPILE.o) printDefaultParam.o -o printDefaultParam $(SOURCES.o) $(LPATH)  $(LIBPATH)
momentumAverage: $(SOURCES.o) momentumAverage.o
	$(COMPILE.o) momentumAverage.o -o momentumAverage $(SOURCES.o) $(LPATH)  $(LIBPATH)
parallelTranslation: $(SOURCES.o) parallelTranslation.o $(SOURCES.h)
	$(COMPILE.o) parallelTranslation.o -o parallelTranslation $(SOURCES.o) $(LPATH)  $(LIBPATH)
metamorphosis: $(SOURCES.o) metamorphosis.o
	$(COMPILE.o) metamorphosis.o -o metamorphosis $(SOURCES.o) $(LPATH) $(LIBPATH)
initialMomentumLandmarkMatching: $(SOURCES.o) initialMomentumLandmarkMatching.o
	$(COMPILE.o) initialMomentumLandmarkMatching.o -o initialMomentumLandmarkMatching $(SOURCES.o) $(LPATH)  $(LIBPATH)
initialMomentumMatching: $(SOURCES.o) initialMomentumMatching.o
	$(COMPILE.o) initialMomentumMatching.o -o initialMomentumMatching $(SOURCES.o) $(LPATH)  $(LIBPATH)
initialMomentumMatchingSYM: $(SOURCES.o) initialMomentumMatchingSYM.o
	$(COMPILE.o) initialMomentumMatchingSYM.o -o initialMomentumMatchingSYM $(SOURCES.o) $(LPATH)  $(LIBPATH)
initialMomentumSegmentation: $(SOURCES.o) initialMomentumSegmentation.o
	$(COMPILE.o) initialMomentumSegmentation.o -o initialMomentumSegmentation $(SOURCES.o) $(LPATH)  $(LIBPATH)
velocityMatching: $(SOURCES.o) velocityMatching.o
	$(COMPILE.o) velocityMatching.o -o velocityMatching $(SOURCES.o) $(LPATH)  $(LIBPATH)
greedyImageMatching: $(SOURCES.o) greedyImageMatching.o
	$(COMPILE.o) greedyImageMatching.o -o greedyImageMatching $(SOURCES.o) $(LPATH)  $(LIBPATH)
matchShootMG: $(SOURCES.o) matchShootMG.o
	$(COMPILE.o) matchShootMG.o -o matchShootMG $(SOURCES.o) $(LPATH)  $(LIBPATH)
shootingBatch: $(SOURCES.o) shootingBatch.o
	$(COMPILE.o) shootingBatch.o -o shootingBatch $(SOURCES.o) $(LPATH)  $(LIBPATH)
affine_registration: $(SOURCES.o) affine_registration.o
	$(COMPILE.o) affine_registration.o -o affine_registration $(SOURCES.o) $(LPATH)  $(LIBPATH)
testKernel: $(SOURCES.o) testKernel.o
	$(COMPILE.o) testKernel.o -o testKernel $(SOURCES.o) $(LPATH)  $(LIBPATH)
shoot: $(SOURCES.o) shoot.o
	$(COMPILE.o) shoot.cpp -o shoot $(SOURCES.o) $(LPATH)  $(LIBPATH)
momentumCovMatrix: $(SOURCES.o) momentumCovMatrix.o
	$(COMPILE.o) momentumCovMatrix.o -o momentumCovMatrix $(SOURCES.o) $(LPATH)  $(LIBPATH)
tensorCurvature: $(SOURCES.o) tensorCurvature.o
	$(COMPILE.o) tensorCurvature.cpp -o tensorCurvature $(SOURCES.o) $(LPATH)  $(LIBPATH)

doc:
	doxygen matching.dox

clean:
	rm -f *.o *.d

tar: $(SOURCES.cpp) $(SOURCES.h)
	tar cvzf matching.tgz $(DIR.h) $(SOURCES.cpp) $(SOURCES.h) $(MAINS.cpp) Makefile.template matching.dox README.pdf

