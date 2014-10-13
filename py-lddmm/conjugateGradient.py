import numpy as np
import scipy as sp
import logging

# Class running noninear conjugate gradient
# opt is an optimizable class that must provide the following functions:
#   getVariable(): current value of the optimzed variable
#   objectiveFun(): value of the objective function
#   updateTry(direction, step, [acceptThreshold]) computes a temporary variable by moving the current one in the direction 'dircetion' with step 'step'
#                                                 the temporary variable is not stored if the objective function is larger than acceptThreshold (when specified)
#                                                 This function should not update the current variable
#   acceptVarTry() replace the current variable by the temporary one
#   getGradient(coeff) returns coeff * gradient; the result can be used as 'direction' in updateTry
#
# optional functions:
#   startOptim(): called before starting the optimization
#   startOfIteration(): called before each iteration
#   endOfIteration() called after each iteration
#   endOptim(): called once optimization is completed
#   dotProduct(g1, g2): returns a list of dot products between g1 and g2, where g1 is a direction and g2 a list of directions
#                       default: use standard dot product assuming that directions are arrays
#   addProd(g0, step, g1): returns g0 + step * g1 for directions g0, g1
#   copyDir(g0): returns a copy of g0
#   randomDir(): Returns a random direction
# optional attributes:
#   gradEps: stopping theshold for small gradient
#   gradCoeff: normalizaing coefficient for gradient.
#
# verb: for verbose printing
# TestGradient evaluate accracy of first order approximation (debugging)
# epsInit: initial gradient step

def cg(opt, verb = True, maxIter=1000, TestGradient = False, epsInit=10.):

    if (hasattr(opt, 'getVariable')==False | hasattr(opt, 'objectiveFun')==False | hasattr(opt, 'updateTry')==False | hasattr(opt, 'acceptVarTry')==False | hasattr(opt, 'getGradient')==False):
        logging.error('Error: required functions are not provided')
        return

    if hasattr(opt, 'startOptim'):
        opt.startOptim()

    if hasattr(opt, 'gradEps'):
        gradEps = opt.gradEps
    else:
        gradEps = 1.0

    if hasattr(opt, 'cgBurnIn'):
        cgBurnIn = opt.cgBurnIn
    else:
        cgBurnIn = -1
    
    if hasattr(opt, 'gradCoeff'):
        gradCoeff = opt.gradCoeff
    else:
        gradCoeff = 1.0

    if hasattr(opt, 'restartRate'):
        restartRate = opt.restartRate
    else:
        restartRate = 100

    eps = epsInit
    epsMin = 1e-10
    opt.converged = False

    obj = opt.objectiveFun()
    logging.info('iteration 0: obj = {0: .5f}'.format( obj))
    # if (obj < 1e-10):
    #     return opt.getVariable()


    skipCG = 0
    for it  in range(maxIter):
        if it % restartRate == 0:
            skipCG = 1 ;

        if hasattr(opt, 'startOfIteration'):
            opt.startOfIteration()
        grd = opt.getGradient(gradCoeff)

        if TestGradient:
            if hasattr(opt, 'randomDir'):
                dirfoo = opt.randomDir()
            else:
                dirfoo = np.random.normal(size=grd.shape)
            epsfoo = 1e-7
            objfoo = opt.updateTry(dirfoo, epsfoo, obj-1e10)
            if hasattr(opt, 'dotProduct'):
                [grdfoo] = opt.dotProduct(grd, [dirfoo])
            else:
                grdfoo = np.multiply(grd, dirfoo).sum()
            logging.info('Test Gradient: %.4f %.4f' %((objfoo - obj)/epsfoo, -grdfoo * gradCoeff ))

        if it == 0 or it == cgBurnIn:
            if hasattr(opt, 'dotProduct'):
                [grdOld2] = opt.dotProduct(grd, [grd])
            else:
                grdOld2 = np.multiply(grd, grd).sum()
            grd2= grdOld2
            grdTry = np.sqrt(max(1e-20, grdOld2))
            if hasattr(opt, 'copyDir'):
                oldDir = opt.copyDir(grd)
                grdOld = opt.copyDir(grd)
                dir0 = opt.copyDir(grd)
            else:
                oldDir = np.copy(grd)
                grdOld = np.copy(grd)
                dir0 = np.copy(grd)
            beta = 0
        else:
            if hasattr(opt, 'dotProduct'):
                [grd2, grd12] = opt.dotProduct(grd, [grd, grdOld])
            else:
                grd2 = np.multiply(grd, grd).sum()
                grd12 = np.multiply(grd, grdOld).sum()

            if skipCG == 0:
                beta = max(0, (grd2 - grd12)/grdOld2)
            else:
                beta = 0

            grdOld2 = grd2
            grdTry = np.sqrt(max(1e-20,grd2 + beta * grd12))

            if hasattr(opt, 'addProd'):
                dir0 = opt.addProd(grd, oldDir, beta)
            else:
                dir0 = grd + beta * oldDir

            if hasattr(opt, 'copyDir'):
                oldDir = opt.copyDir(dir0)
                grdOld = opt.copyDir(grd)
            else:
                oldDir = np.copy(dir0)
                grdOld = np.copy(grd)

        objTry = opt.updateTry(dir0, eps, obj)

        noUpdate = 0
        if objTry > obj:
            #fprintf(1, 'iteration %d: obj = %.5f, eps = %.5f\n', it, objTry, eps) ;
            epsSmall = 0.000001/(grdTry)
            #print 'Testing small variation, eps = {0: .10f}'.format(epsSmall)
            objTry0 = opt.updateTry(dir0, epsSmall, obj)
            if objTry0 > obj:
                if (skipCG == 1) | (beta < 1e-10):
                    logging.info('iteration {0:d}: obj = {1:.5f}, eps = {2:.5f}, beta = {3:.5f}, gradient: {4:.5f}'.format(it+1, obj, eps, beta, np.sqrt(grd2)))
                    logging.info('Stopping Gradient Descent: bad direction')
                    break
                else:
                    if verb:
                        logging.info('Disabling CG: bad direction')
                        skipCG = 1
                        noUpdate = 1
            else:
                while (objTry > obj) and (eps > epsMin):
                    eps = eps / 2
                    objTry = opt.updateTry(dir0, eps, obj)
                    #opt.acceptVarTry()

                    #print 'improve'
        ## reducing step if improves
        if noUpdate == 0:
            contt = 1
            while contt==1:
                objTry2 = opt.updateTry(dir0, .5*eps, objTry)
                if objTry > objTry2:
                    eps = eps / 2
                    objTry=objTry2
                else:
                    contt=0

       
        # increasing step if improves
            contt = 1
            #eps0 = eps / 4
            while contt==1:
                objTry2 = opt.updateTry(dir0, 1.25*eps, objTry)
                if objTry > objTry2:
                    eps *= 1.25
                    objTry=objTry2
                else:
                    contt=0

            #print obj+obj0, objTry+obj0
            if (np.fabs(obj-objTry) < .000001):
                if (skipCG==1) | (beta < 1e-10) :
                    logging.info('iteration {0:d}: obj = {1:.5f}, eps = {2:.5f}, beta = {3:.5f}, gradient: {4:.5f}'.format(it+1, obj, eps, beta, np.sqrt(grd2)))
                    logging.info('Stopping Gradient Descent: small variation')
                    opt.converged = True
                    break
                else:
                    if verb:
                        logging.info('Disabling CG: small variation')
                    skipCG = 1
                    eps = 1.0
            else:
                skipCG = 0

            opt.acceptVarTry()
            obj = objTry
            if verb | (it == maxIter):
                logging.info('iteration {0:d}: obj = {1:.5f}, eps = {2:.5f}, beta = {3:.5f}, gradient: {4:.5f}'.format(it+1, obj, eps, beta, np.sqrt(grd2)))

            if np.sqrt(grd2) <gradEps:
                logging.info('Stopping Gradient Descent: small gradient')
                opt.converged = True 
                break
            eps = 100*eps

            if hasattr(opt, 'endOfIteration'):
                opt.endOfIteration()

    if hasattr(opt, 'endOptim'):
        opt.endOptim()

    return opt.getVariable()
