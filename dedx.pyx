#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import os
import numpy
cimport numpy
from constants cimport *
cimport cython
cimport specfun
cimport spline

cdef extern from "math.h":
    double log(double x)
    double log10(double x)
    double sqrt(double x)
    double sin(double x)
    double cos(double x)
    double atan(double x)
    double exp(double x)
    double fabs(double x)

cdef extern from "complex.h":
    double creal(double complex z)
    double cimag(double complex z)
    double cabs(double complex z)
    double complex cexp(double complex z)
    double carg(double complex z)
    

# Good overview given in
# Weaver, B. A.; Westphal, A. J.
# NIM B187, 3, (2002) p. 285-301.
# Everything in this class SI units if not stated otherwise.
cdef class energyLoss:
    cdef:
        double rho, rho0Sh, cbarSh, aSh, kSh, delta0Sh, x0Sh, x1Sh, iEx, Z2OverA2, A2
        unsigned int Z2 
        double dE, dOmega2, gamma, A1, chic2, lnchia2effNum, lnchia2effDenom
        spline.Spline3D splineLLS, splineXLS
        double[:] rhoData, rho0ShData, cbarShData, x0ShData, x1ShData
        double[:] kShData, aShData, delta0ShData, iExData, Z2OverA2Data
        unsigned int[:] Z2Data

    def __init__(self, unsigned int matIdx, double rho = -1., unsigned int useSplineLS = 1):
        self.loadData()
        self.loadSpline(useSplineLS = useSplineLS)
        self.setMaterial(matIdx, rho = rho)
        self.setZero()

    # Loads material data. Kept in memory to have fast material switching.
    # Shouldn't take more than a couple mb.
    cpdef loadData(self):
        # Use tabulated data.
        # Standard density, z/a target, z target, Sternheimer 
        # density effect coefficients cbarSh x0Sh x1Sh kSh aSh delta0Sh,
        # mean excitation potential IEx, name/formula.
        # -- data taken from: 
        # http://pdg.lbl.gov/2009/AtomicNuclearProperties/expert.html
        # Files:
        # Properties6_Key.pdf
        # properties6.dat
        # Use extract.py to convert into more readable format if new
        # data available (and if format still applies...).      
        folder = 'resources/'
        # Unfortunately index does not directly correspond to Z, so
        # print Material. Use provided method.       
        self.rhoData = numpy.loadtxt(folder + 'rho.dat', dtype=numpy.double)
        self.rho0ShData = numpy.loadtxt(folder + 'rhoSternheimer.dat', dtype=numpy.double)
        self.cbarShData = numpy.loadtxt(folder + 'cbarSternheimer.dat', dtype=numpy.double) 
        self.x0ShData = numpy.loadtxt(folder + 'x0Sternheimer.dat', dtype=numpy.double)
        self.x1ShData = numpy.loadtxt(folder + 'x1Sternheimer.dat', dtype=numpy.double)
        self.kShData = numpy.loadtxt(folder + 'kSternheimer.dat', dtype=numpy.double)
        self.aShData = numpy.loadtxt(folder + 'aSternheimer.dat', dtype=numpy.double)
        self.delta0ShData = numpy.loadtxt(folder + 'delta0Sternheimer.dat', dtype=numpy.double)
        self.iExData = numpy.loadtxt(folder + 'iEx.dat', dtype=numpy.double)
        self.Z2OverA2Data = numpy.loadtxt(folder + 'zOverA.dat', dtype=numpy.double)    
        self.Z2Data = numpy.loadtxt(folder + 'z.dat', dtype=numpy.uint32)        
        
    # Loads the pre-calculated splines if desired. Currently only splines for LS correction.
    cpdef loadSpline(self, unsigned int useSplineLS = 1):
        cdef:
            double[:,:,:] splineLLSData
            double[:,:,:] splineXLSData   
            double maxZ1, minZ1, maxA1, minA1, maxG, minG
            unsigned int nZ1, nA1, nG
            double densScal, dZ1, dA1, dG, gamma, beta, Z1, A1
            unsigned int ii, jj, kk
            
        if useSplineLS==0:
            self.splineLLS = None
            self.splineXLS = None
        else:
            self.splineLLS = spline.Spline3D()
            self.splineXLS = spline.Spline3D()
            if os.path.exists('splineLLSData') and os.path.exists('splineXLSData'):
                self.splineLLS.load('splineLLSData')
                self.splineXLS.load('splineXLSData')
            else:
                print 'No spline data found. Creating...'
                # Values can be changed.
                # Set such that 100mb spline data.
                # Final energy loss is practically calculated
                # to full double precision compared to directly
                # calculating LS correction.
                nZ1 = 120; nA1 = 300; nG = 300;
                maxZ1 = 120.; minZ1 = 1.;
                maxA1 = 300.; minA1 = 1.;
                maxG = log(1000); minG = log(1.e-3);
                dZ1 = (maxZ1-minZ1)/(nZ1-1.)
                dA1 = (maxA1-minA1)/(nA1-1.)
                dG = (maxG-minG)/(nG-1.)
                splineLLSData = numpy.empty((nZ1,nA1,nG), dtype=numpy.double)
                splineXLSData = numpy.empty((nZ1,nA1,nG), dtype=numpy.double)
                for kk in range(nG):
                    print 'Progress: ' + str(<unsigned int> (0.5+1.*kk/(nG-1)*100.)) + '%'
                    gamma = exp(kk*dG+minG)+1
                    beta = sqrt(1.-1./gamma**2)
                    for jj in range(nA1):
                        A1 = jj*dA1+minA1
                        for ii in range(nZ1):
                            Z1 = ii*dZ1+minZ1
                            splineLLSData[ii,jj,kk] = _deltaLLS(Z1, A1, beta, gamma)
                            splineXLSData[ii,jj,kk] = _xLS(Z1, A1, beta, gamma)
                os.mkdir('splineLLSData')
                os.mkdir('splineXLSData')     
                self.splineLLS.make([minZ1,maxZ1,nZ1],[minA1,maxA1,nA1],[minG,maxG,nG],splineLLSData) 
                self.splineXLS.make([minZ1,maxZ1,nZ1],[minA1,maxA1,nA1],[minG,maxG,nG],splineXLSData)           
                self.splineLLS.save('splineLLSData')
                self.splineXLS.save('splineXLSData')        
                    
    # Material Index (see resources nameFormula.dat).
    # Because the data taken from particle data group
    # is a nice source, their material indices are used.      
    cpdef setMaterial(self, unsigned int matIdx, double rho = -1.):        
        cdef:    
            double densScal

        # Use default density if none is provided.    
        if rho < 0.:
            self.rho = self.rhoData[matIdx-1]*1.e3 # Tabulated data is in g/cm^3
        else:
            self.rho = rho
            
        # Density effect data origin goes back to 
        # Sternheimer ADNDT 30, 2, 1984, p. 261-271
        # Data can be obtained from particle data group as mentioned above.
        self.rho0Sh = self.rho0ShData[matIdx-1]*1.e3   # Tabulated data is in g/cm^3.
        self.cbarSh = self.cbarShData[matIdx-1]
        if self.cbarSh == 0.:
            # Sternheimer Phys. Rev. B3, 11 (1971)
            # Method for non-tabulated materials.
            # See dedxTemp.txt for remnants of implementation if necessary later.
            raise NotImplementedError('Tabulated data not complete. Calculation of density effect for \
                                       non-tabulated material not implemented yet.')
        # Scaling with density.
        densScal = self.rho/self.rho0Sh   
        self.cbarSh -= log(10)*log10(densScal)
        self.x0Sh = self.x0ShData[matIdx-1]
        self.x0Sh -= 0.5*log10(densScal)
        self.x1Sh = self.x1ShData[matIdx-1]
        self.x1Sh -= 0.5*log10(densScal)
        self.kSh = self.kShData[matIdx-1]
        self.aSh = self.aShData[matIdx-1]
        self.delta0Sh = self.delta0ShData[matIdx-1]

        self.iEx = self.iExData[matIdx-1]*elementary_charge    # Tabulated data is in eV.
        self.Z2OverA2 = self.Z2OverA2Data[matIdx-1]    
        self.Z2 = self.Z2Data[matIdx-1]        
        self.A2 = self.Z2/self.Z2OverA2        
        
    
    # Track the projectile through the specified material.            
    cpdef track(self, double Z1, double A1, double gamma, double thickness):
        cdef:
            unsigned int ii, jj, nSteps = 4        # Has to be (3n+1) for Simpson's rule later.
            double dx, gammaNTest1, gammaNTest2, dOmegadx
            double[:] gammaN
            double err = 1., acc = 1.e-12
            double k1, k2, k3, k4
            double rho = self.rho, A2 = self.A2, iEx = self.iEx
            double cbarSh = self.cbarSh, x0Sh = self.x0Sh, x1Sh = self.x1Sh
            double kSh = self.kSh, aSh = self.aSh, delta0Sh = self.delta0Sh
            unsigned int Z2 = self.Z2
            spline.Spline3D splineLLS = self.splineLLS, splineXLS = self.splineXLS
        # Track at least for 1fm, otherwise method messes up.
        if thickness<1.e-15:
            thickness = 1.e-15
        # First energy loss.
        # Runge-Kutta ODE solving of Bethe.
        for jj in range(16):
            # Make sure a rough estimate of error fulfills set accuracy.
            gammaNTest1 = gamma
            dx = thickness/(nSteps-1)
            k1 = _dGammadx(rho, Z1, Z2, A1, A2, gammaNTest1, iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS)
            k2 = _dGammadx(rho, Z1, Z2, A1, A2, gammaNTest1+0.5*dx*k1, iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS)     
            k3 = _dGammadx(rho, Z1, Z2, A1, A2, gammaNTest1+0.5*dx*k2, iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS)
            k4 = _dGammadx(rho, Z1, Z2, A1, A2, gammaNTest1+dx*k3, iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS)
            gammaNTest1 = gammaNTest1 + 1./6.*dx*(k1+2.*k2+2.*k3+k4)
              
            nSteps = 2*nSteps-1 
            gammaNTest2 = gamma
            dx = thickness/(nSteps-1)
            for ii in range(2):
                k1 = _dGammadx(rho, Z1, Z2, A1, A2, gammaNTest2, iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS)
                k2 = _dGammadx(rho, Z1, Z2, A1, A2, gammaNTest2+0.5*dx*k1, iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS)
                k3 = _dGammadx(rho, Z1, Z2, A1, A2, gammaNTest2+0.5*dx*k2, iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS)
                k4 = _dGammadx(rho, Z1, Z2, A1, A2, gammaNTest2+dx*k3, iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS)
                gammaNTest2 = gammaNTest2 + 1./6.*dx*(k1+2.*k2+2.*k3+k4)
              
            err = nSteps*fabs(gammaNTest2-gammaNTest1)/gammaNTest2
            if err<acc:
                break
        
        gammaN = numpy.empty(nSteps, dtype=numpy.double)
        gammaN[0] = gamma
        dx = thickness/(nSteps-1)
        for ii in range(1, nSteps):
            k1 = _dGammadx(rho, Z1, Z2, A1, A2, gammaN[ii-1], iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS)
            k2 = _dGammadx(rho, Z1, Z2, A1, A2, gammaN[ii-1]+0.5*dx*k1, iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS)
            k3 = _dGammadx(rho, Z1, Z2, A1, A2, gammaN[ii-1]+0.5*dx*k2, iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS)
            k4 = _dGammadx(rho, Z1, Z2, A1, A2, gammaN[ii-1]+dx*k3, iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS)
            gammaN[ii] = gammaN[ii-1] + 1./6.*dx*(k1+2.*k2+2.*k3+k4)
        
        # With energy loss data integrate fluctuation of energy loss.
        # Simpsons 3/8 rule (should be equivalent to Runge Kutta).
        dOmega2 = _dOmega2dx(rho, Z1, Z2, A1, A2, gammaN[0], splineXLS)
        for ii in range(1,nSteps-3,3):
            dOmega2 += 3.*_dOmega2dx(rho, Z1, Z2, A1, A2, gammaN[ii], splineXLS) + \
                       3.*_dOmega2dx(rho, Z1, Z2, A1, A2, gammaN[ii+1], splineXLS) + \
                       2.*_dOmega2dx(rho, Z1, Z2, A1, A2, gammaN[ii+2], splineXLS)
        dOmega2 += 3.*_dOmega2dx(rho, Z1, Z2, A1, A2, gammaN[nSteps-3], splineXLS) + \
                   3.*_dOmega2dx(rho, Z1, Z2, A1, A2, gammaN[nSteps-2], splineXLS) + \
                   _dOmega2dx(rho, Z1, Z2, A1, A2, gammaN[nSteps-1], splineXLS)
        dOmega2 *= 3./8.*dx

        # With energy loss data integrate angular straggling.
        # Simpsons 3/8 rule (should be equivalent to Runge Kutta).
        # It's kinda weird, because integrating sigma produces
        # wrong results; instead chia and chic have to be
        # accumulated in specific ways (see Lynch paper).
        chic2 = _chic2dx(rho, Z1, Z2, A1, A2, gammaN[0])
        for ii in range(1,nSteps-3,3):
            chic2 += 3.*_chic2dx(rho, Z1, Z2, A1, A2, gammaN[ii]) + \
                     3.*_chic2dx(rho, Z1, Z2, A1, A2, gammaN[ii+1]) + \
                     2.*_chic2dx(rho, Z1, Z2, A1, A2, gammaN[ii+2])
        chic2 += 3.*_chic2dx(rho, Z1, Z2, A1, A2, gammaN[nSteps-3]) + \
                 3.*_chic2dx(rho, Z1, Z2, A1, A2, gammaN[nSteps-2]) + \
                 _chic2dx(rho, Z1, Z2, A1, A2, gammaN[nSteps-1])
        chic2 *= 3./8.*dx
        
        lnchia2effNum = _lnchia2effdxNum(rho, Z1, Z2, A1, gammaN[0])

        for ii in range(1,nSteps-3,3):
            lnchia2effNum += 3.*_lnchia2effdxNum(rho, Z1, Z2, A1, gammaN[ii]) + \
                          3.*_lnchia2effdxNum(rho, Z1, Z2, A1, gammaN[ii+1]) + \
                          2.*_lnchia2effdxNum(rho, Z1, Z2, A1, gammaN[ii+2])
        lnchia2effNum += 3.*_lnchia2effdxNum(rho, Z1, Z2, A1, gammaN[nSteps-3]) + \
                         3.*_lnchia2effdxNum(rho, Z1, Z2, A1, gammaN[nSteps-2]) + \
                         _lnchia2effdxNum(rho, Z1, Z2, A1, gammaN[nSteps-1])
        lnchia2effNum *= 3./8.*dx
        
        lnchia2effDenom = _lnchia2effdxDenom(rho, Z2, A1)*dx*(nSteps-1.)

        self.dE += (gammaN[nSteps-1]-gammaN[0])*A1*u*c*c
        self.gamma = gammaN[nSteps-1]
        self.dOmega2 += dOmega2
        self.chic2 += chic2
        self.lnchia2effNum += lnchia2effNum
        self.lnchia2effDenom += lnchia2effDenom
        self.A1 = A1
               
    cpdef double eloss(self):
        return self.dE

    cpdef double sigmaE(self):
        return sqrt(self.dOmega2)

    cpdef double sigmaAng(self):
        return _sigmaAngStraggLy(self.chic2, exp(self.lnchia2effNum/self.lnchia2effDenom)) 
        
    cpdef double elossAMEV(self):
        # A1 is assumed to be the last used A1.
        return self.dE/(elementary_charge*1.e6*self.A1)

    cpdef double sigmaEAMEV(self):
        # A1 is assumed to be the last used A1.
        return sqrt(self.dOmega2)/(elementary_charge*1.e6*self.A1)

    cpdef double sigmaAngmrad(self):
        return _sigmaAngStraggLy(self.chic2, exp(self.lnchia2effNum/self.lnchia2effDenom))*1.e3   
    
    cpdef double getGamma(self):
        return self.gamma
 
    cpdef setZero(self):
        self.dE = 0.
        self.dOmega2 = 0.
        self.chic2 = 0.
        self.lnchia2effNum = 0.
        self.lnchia2effDenom = 0.
    
    cpdef double getRho(self):
        return self.rho

    cpdef printMaterials(self):
        cdef unsigned int ii = 1        
        file = open('resources/nameFormula.dat','r')
        for line in file:
            print ii, line.strip()
            ii += 1
        
            
    # print 'Material:', numpy.loadtxt(folder + 'nameFormula.dat')[matIdx-1]


#########################################################
# Everything in this part here belongs to the
# energy loss calculations.
#########################################################    
cpdef double _dGammadx(double rho, double Z1, unsigned int Z2, double A1, double A2, double gamma, double iEx, 
                       double cbarSh, double x0Sh, double x1Sh, double kSh, double aSh, double delta0Sh, spline.Spline3D splineLLS):
    cdef:
        double beta = sqrt(1.-1./gamma**2)
        double Z1eff = _effChargeH(Z1, Z2, A1, gamma)
    return _L(Z1eff, A1, beta, gamma, Z2, iEx, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh, splineLLS) * \
           _facDedx(rho, Z1eff, Z2, A2, beta)/(A1*u*c*c)
 
# Factor of de/dx in front of L.       
cpdef double _facDedx(double rho, double Z1, unsigned int Z2, double A2, double beta):
    return -elementary_charge**4*N_A*1.e3/(4.*pi*epsilon_0*epsilon_0*m_e*c*c)*rho*Z1**2*Z2/A2/beta**2

# All contributions to L together.
cpdef _L(double Z1, double A1, double beta, double gamma,
         unsigned int Z2, double I, double cbarSh, double x0Sh, double x1Sh, double kSh,
         double aSh, double delta0Sh, spline.Spline3D splineLLS):
    cdef:
        double betaGamma = beta*gamma    
    if splineLLS is None:
        return _L0WithoutDensityDelta(beta, betaGamma, I)*_facLBarkasJM(betaGamma, Z1, Z2) - \
               _deltaDensitySh(betaGamma, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh)*0.5 - \
               _deltaLShellBB(betaGamma, I, Z2) + _deltaLLS(Z1, A1, beta, gamma)
    else:
        return _L0WithoutDensityDelta(beta, betaGamma, I)*_facLBarkasJM(betaGamma, Z1, Z2) - \
               _deltaDensitySh(betaGamma, cbarSh, x0Sh, x1Sh, kSh, aSh, delta0Sh)*0.5 - \
               _deltaLShellBB(betaGamma, I, Z2) + splineLLS.interpolate(Z1, A1, log(gamma-1))

# Standard Bethe L without density correction.
cpdef double _L0WithoutDensityDelta(double beta, double betaGamma, double I):
    return log(2*m_e*c*c*betaGamma*betaGamma/I) - beta*beta

# Sternheimer et al.
# Atomic Data and Nuclear Data Tables
# Volume 30, Issue 2, March 1984, Pages 261--271
# See above.
cpdef double _deltaDensitySh(double betaGamma, double cbar, double x0, double x1, 
                             double k, double a, double delta0):
    cdef:
        double x
    x = log10(betaGamma)
    if x>=x1:
        return 2.*log(10.)*x - cbar
    elif x>=x0:
        return 2.*log(10.)*x - cbar + a*(x1-x)**k
    else:
        return delta0*10**(2*(x-x0))

# W.H. Barkas, M.J. Berger, NASA Report SP-3013 (1964).
# Fit for eta=beta*gamma>=0.13. Not trustworthy below.
cpdef double _deltaLShellBB(double betaGamma, double I, unsigned int Z2):
    cdef:
        double betaGammaInvSq, cShell
    betaGammaInvSq = 1./betaGamma/betaGamma
    cShell = (0.422377*betaGammaInvSq + 0.0304043*betaGammaInvSq*betaGammaInvSq -
              0.00038106*betaGammaInvSq*betaGammaInvSq*betaGammaInvSq)*1.e-6*(I/elementary_charge)**2 + \
             (3.858019*betaGammaInvSq - 0.1667989*betaGammaInvSq*betaGammaInvSq + 
              0.00157955*betaGammaInvSq*betaGammaInvSq*betaGammaInvSq)*1.e-9*(I/elementary_charge)**3
    return cShell/Z2
    
    
# J.D. Jackson and R.L. McCarthy
# Physical Review B6, 4131 (1972)
# V^2*F(V) values (index 1 till 10) from
# Weaver, B. A.; Westphal, A. J.
# NIM B187, 3, (2002) p. 285-301.
# Jackson/McCarthy Barkas correction not trustworthy for V<0.8. 
# For V>10 it is approximate, but gets small anyway. 
cpdef double _facLBarkasJM(double betaGamma, double Z1, unsigned int Z2):
    cdef:
        double V
        double* V2F = [0.3, 0.33, 0.32, 0.27, 0.23, 0.21, 0.19, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12]
        unsigned int ii
    V = betaGamma/alpha/sqrt(Z2)

    if V<10.:
        ii = 0
        while V>=ii:
            ii += 1
        return (1. + 2.*Z1/sqrt(Z2)/V/V*(V2F[ii]*(V-ii+1.) + (ii-V)*V2F[ii-1]))
    else:
        return (1. + 2.*Z1/sqrt(Z2)/V**2.5*sqrt(10)*V2F[10])
        

# F. Hubert et al.
# NIM B36 (1989) p. 357-363
# Fit to large dataset.
# Not sure if really trustworthy for small Z2<4.
cpdef double _effChargeH(double Z1, unsigned int Z2, double A1, double gamma):
    cdef:
        double D, B, C, x1, x2, x3, x4
        double E = (gamma-1)*u/elementary_charge*c*c*1.e-6    # In MeV.

    # Standard values for most elements.
    if Z1!=6 and Z1!=4:
        D = 1.164 + 0.2319*exp(-0.004302*Z2)
        B = 1.658
        C = 0.05170
        x1 = D + B*exp(-C*Z1)
        x2 = 8.144 + 0.09876*log(Z2)
        x3 = 0.3140 + 0.01072*log(Z2)
        x4 = 0.5218 + 0.02521*log(Z2)
    # Carbon values.
    elif Z1==6:
        D = 2.584
        B = 1.910
        C = 0.03958
        x1 = D + B*exp(-C*Z1)
        x2 = 6.933
        x3 = 0.2433
        x4 = 0.3969     
    # Beryllium values.
    elif Z1==4:
        D = 2.045
        B = 2.000
        C = 0.04369
        x1 = D + B*exp(-C*Z1)
        x2 = 7.000
        x3 = 0.2643
        x4 = 0.4171  
    return Z1*(1-x1*exp(-x2*E**x3*Z1**(-x4)))

# T.E. Pierce, M. Blann
# Phys. Rev. 173 (1968) 390
# Used by ATIMA. Crude Approximation.
cpdef double _effChargeP(double Z1, double beta):
    return Z1*(1.-exp(-0.95/alpha*beta/Z1**(2./3.)))

# Lindhard-Soerensen correction of energy loss.
# J. Lindhard, A.H. Soerensen
# Phys. Rev. A53 (1996) 2443
cpdef double _deltaLLS(double Z1, double A1, double beta, double gamma):
    
    cdef:
        double deltaLLS, eta, eta2i, r, rprime, sk, xikr, xiks, \
               FrOverGr, FsOverGs, GrOverGs, FintOverGintb0, Hk, \
               Hmk, Hmkm1, a0, a1, bn, an, anm1, bnsum, ansum, \
               deltak, deltakr, deltaks, deltamk, deltmkr, deltamks, \
               deltamkm1, deltamkm1r, deltamkm1s
        double complex lambdar, lambdas
        unsigned int nAB, nK, lp, lm, kk
        
    nAB = 35        # empirically chosen.
    nK = 35         # empirically chosen.
    deltaLLS = 0.
    eta = alpha*Z1/beta
    eta2i = 1./(eta**2)
    r = 1.18e-15*A1**(1./3.)
    rprime = r*m_e*c/hbar
       
    # Ultrarelativistic case.
    if gamma>10./rprime:
        deltaLLS = -log(beta*gamma*rprime)-0.2+0.5*beta*beta
    # Normal case.    
    else:
        deltak = 0.
        deltamkm1 = _deltamk(1, eta, beta, gamma, rprime)
        # Rest of series.
        for kk in range(1,nK):
            deltakm1 = deltak
            deltak = _deltak(kk, eta, beta, gamma, rprime)
            deltamk = deltamkm1
            deltamkm1 = _deltamk(kk+1, eta, beta, gamma, rprime)       
            deltaLLS += (eta2i*(kk*(kk-1.)/(2.*kk-1)*sin(deltak-deltakm1)**2 +
                                kk*(kk+1.)/(2.*kk+1)*sin(deltamk-deltamkm1)**2 +
                                kk/(4.*kk*kk-1.)*sin(deltak-deltamk)**2) -
                         1./kk)

        deltaLLS += beta*beta*0.5
    
    return deltaLLS


# Helper function to calculate deltak.
cdef double _deltak(unsigned int kk, double eta, double beta, double gamma, double rprime):
    cdef:
        double sk, xikr, xiks, FrOverGr, FsOverGs, GrOverGs, FintOverGint, \
               b0, a0, a1, ansum, bnsum, an, anm1, bn, bnm1
        double complex lambdar, lambdas
        unsigned int nn, ll
        unsigned int nAB = 35   #empirically chosen
    sk = sqrt(kk*kk -(eta*beta)**2)   
    xikr = 0.5*carg((kk-1.j*eta/gamma)/(sk-1.j*eta))
    xiks = 0.5*carg((kk-1.j*eta/gamma)/(-sk-1.j*eta))
    lambdar = cexp(1.j*(xikr-beta*gamma*rprime))*specfun.hyp1f1(sk+1.+1.j*eta,2*sk+1.,2.j*beta*gamma*rprime)
    lambdas = cexp(1.j*(xiks-beta*gamma*rprime))*specfun.hyp1f1(-sk+1.+1.j*eta,-2*sk+1.,2.j*beta*gamma*rprime)
    FrOverGr = sqrt((gamma-1.)/(gamma+1.))*creal(lambdar)/cimag(lambdar)
    FsOverGs = sqrt((gamma-1.)/(gamma+1.))*creal(lambdas)/cimag(lambdas)
    GrOverGs = (((<int> (-2.*sk+1)%2)*2-1)*
                exp(specfun.lnabscgamma(sk+1.+1.j*eta) - specfun.lnabscgamma(-sk+1.+1.j*eta) +
                    specfun.lnabsgamma(-2.*sk+1) - specfun.lnabsgamma(2.*sk+1))* 
                cimag(lambdar)/cimag(lambdas)*(2.*beta*gamma*rprime)**(2*sk)) 
    ll = kk
    b0 = 1.
    a0 = (2*kk+1)/(rprime*(gamma+1.)+1.5*beta*eta)*b0
    a1 = 0.5*(rprime*(-gamma+1.)-1.5*beta*eta)*b0
    ansum = a0; bnsum = b0; bn = b0; an = a1; anm1 = a0;
    for nn in range(1,nAB):
        bnm1 = bn
        bn = ((rprime*(gamma+1.)+1.5*beta*eta)*an - 0.5*beta*eta*anm1)/(2.*kk+2.*nn+1.)
        bnsum += bn
        ansum += an
        anm1 = an
        an = ((rprime*(-gamma+1.)-1.5*beta*eta)*bn + 0.5*beta*eta*bnm1)/(2.*nn+2.)
    FintOverGint = ansum/bnsum
    Hk = (FrOverGr-FintOverGint)/(FintOverGint-FsOverGs)*GrOverGs        
    deltakr = xikr-specfun.argctgamma(sk+1.+1.j*eta)+0.5*pi*(ll-sk)
    deltaks = xiks-specfun.argctgamma(-sk+1.+1.j*eta)+0.5*pi*(ll+sk)
    return atan((sin(deltakr)+Hk*sin(deltaks))/(cos(deltakr)+Hk*cos(deltaks)))

# Helper function to calculate deltamk.
cdef double _deltamk(unsigned int kk, double eta, double beta, double gamma, double rprime):
    cdef:
        double sk, xikr, xiks, FrOverGr, FsOverGs, GrOverGs, FintOverGint, \
               b0, a0, a1, ansum, bnsum, an, anm1, bn, bnm1
        double complex lambdar, lambdas
        unsigned int nn, ll
        unsigned int nAB = 35   #empirically chosen
    sk = sqrt(kk*kk-(eta*beta)**2)   
    xikr = 0.5*carg((-(kk+1.j*eta/gamma))/(sk-1.j*eta))
    xiks = 0.5*carg((-(kk+1.j*eta/gamma))/(-sk-1.j*eta))
    lambdar = cexp(1.j*(xikr-beta*gamma*rprime))*specfun.hyp1f1(sk+1.+1.j*eta,2*sk+1.,2.j*beta*gamma*rprime)
    lambdas = cexp(1.j*(xiks-beta*gamma*rprime))*specfun.hyp1f1(-sk+1.+1.j*eta,-2*sk+1.,2.j*beta*gamma*rprime)
    FrOverGr = sqrt((gamma-1.)/(gamma+1.))*creal(lambdar)/cimag(lambdar)
    FsOverGs = sqrt((gamma-1.)/(gamma+1.))*creal(lambdas)/cimag(lambdas)
    GrOverGs = (((<int> (-2.*sk+1)%2)*2-1)*
                exp(specfun.lnabscgamma(sk+1.+1.j*eta) - specfun.lnabscgamma(-sk+1.+1.j*eta) +
                    specfun.lnabsgamma(-2.*sk+1) - specfun.lnabsgamma(2.*sk+1))* 
                cimag(lambdar)/cimag(lambdas)*(2.*beta*gamma*rprime)**(2*sk)) 
    ll = kk-1
    b0 = 1.
    a0 = (2*kk+1)/(rprime*(-gamma+1.)-1.5*beta*eta)*b0
    a1 = 0.5*(rprime*(gamma+1.)+1.5*beta*eta)*b0
    ansum = a0; bnsum = b0; bn = b0; an = a1; anm1 = a0;
    for nn in range(1,nAB):
        bnm1 = bn
        bn = ((rprime*(-gamma+1.)-1.5*beta*eta)*an + 0.5*beta*eta*anm1)/(2.*kk+2.*nn+1.)
        bnsum += bn
        ansum += an
        anm1 = an
        an = ((rprime*(gamma+1.)+1.5*beta*eta)*bn - 0.5*beta*eta*bnm1)/(2.*nn+2.)
    FintOverGint = bnsum/ansum
    Hk = (FrOverGr-FintOverGint)/(FintOverGint-FsOverGs)*GrOverGs        
    deltakr = xikr-specfun.argctgamma(sk+1.+1.j*eta)+0.5*pi*(ll-sk)
    deltaks = xiks-specfun.argctgamma(-sk+1.+1.j*eta)+0.5*pi*(ll+sk)
    return atan((sin(deltakr)+Hk*sin(deltaks))/(cos(deltakr)+Hk*cos(deltaks)))

#########################################################
# Everything in this part here belongs to the
# energy loss fluctucation/straggling.
#########################################################
cpdef double _dOmega2dx(double rho, double Z1, unsigned int Z2, double A1, double A2, double gamma, spline.Spline3D splineXLS):
    cdef:
        double beta = sqrt(1.-1./gamma**2)
        double Z1eff = _effChargeH(Z1, Z2, A1, gamma)
    if splineXLS is None:
        return _facOmega2(rho, Z1eff, Z2, A2, gamma)*_xLS(Z1eff, A1, beta, gamma)
    else:
        return _facOmega2(rho, Z1eff, Z2, A2, gamma)*splineXLS.interpolate(Z1eff, A1, log(gamma-1))

cpdef double _facOmega2(double rho, double Z1, unsigned int Z2, double A2, double gamma):
    return elementary_charge**4*N_A*1.e3/(4.*pi*epsilon_0*epsilon_0)*rho*Z1**2*Z2/A2*gamma*gamma
 
cpdef double _xLS(double Z1, double A1, double beta, double gamma): 
    cdef:
        double deltak, deltakm1, deltakm2, deltamk, deltamkm1, deltamkm2, deltamkp1
        double eta, r, rprime, xLS = 0.
        unsigned int kk, nK = 35    #empirically chosen

    eta = alpha*Z1/beta
    r = 1.18e-15*A1**(1./3.)
    rprime = r*m_e*c/hbar
    
    if gamma>10./rprime:
        # 1.189 is empirical and not mentioned by LS.
        # Makes a smooth transition to ultra-relativistic regime.
        # However, as it's 2**0.25 I might have overlooked something in LS paper.
        xLS = 1.189*sqrt(1.+(alpha*Z1*exp(eul))**2)/(gamma*gamma*rprime*rprime*beta*beta)    
    else:
        deltak = 0.
        deltakm1 = 0.
        deltamk = 0.
        deltamkm1 = _deltamk(1, eta, beta, gamma, rprime)
        deltamkm2 = _deltamk(2, eta, beta, gamma, rprime)
        for kk in range(1,nK):
            deltakm2 = deltakm1
            deltakm1 = deltak
            deltak = _deltak(kk, eta, beta, gamma, rprime)
            deltamkp1 = deltamk
            deltamk = deltamkm1
            deltamkm1 = deltamkm2
            deltamkm2 = _deltamk(kk+2, eta, beta, gamma, rprime)
            xLS += kk*(2.*(kk-1.)/(2.*kk-1.)*sin(deltak-deltakm1)**2 +
                       2.*(kk+1.)/(2.*kk+1.)*sin(deltamk-deltamkm1)**2 +
                       2./(4.*kk*kk-1.)*sin(deltak-deltamk)**2 -
                       (kk-1.)*(kk-2.)/(2.*kk-1.)/(2.*kk-3.)*sin(deltak-deltakm2)**2 -
                       (kk+1.)*(kk+2.)/(2.*kk+1.)/(2.*kk+3.)*sin(deltamk-deltamkm2)**2 -
                       2.*(kk-1.)/(2.*kk-3.)/(4.*kk*kk-1.)*sin(deltak-deltamkp1)**2 -
                       (kk+1.)/(2.*kk+1.)*(1./(4.*kk*kk-1.)+1./(4.*(kk+1.)**2-1.))*sin(deltak-deltamkm1)**2 
                       )
        xLS /= eta*eta
    return xLS



#########################################################
# Everything in this part here belongs to the
# multiple coulomb scattering/angular straggling.
#########################################################
# Everything from
# Lynch, G.R. and Dahl, O.I.
# NIM B58 (1991) 6-10

# Effective charge Z1eff might be a good idea for angular
# straggling as well, but not sure, so not included.

# Final formula to calculate angular straggling.
# Most accurate Gaussian approximation... NOT!
# THIS SHIT DOESNT WORK.
# Not sure if I made a mistake, almost everything might
# be calculated wrong. HOWEVER, the final formula
# for sigma scales at best case linearly with thickness when toying around
# with different chi. It should roughly scale with the square root
# of the thickness. So the scaling is completely off.
# Maybe still a mistake on my part, but this would explain
# why nobody ever uses this formula, although it is fairly simple
# and "much more accurate".
cpdef double _sigmaAngStraggLy(double chic2, double chia2):
    cdef:
        double F = 0.99
        double Omega, v
    Omega = chic2/chia2
    v = 0.5*Omega/(1.-F)
    return sqrt(chic2/(1.+F*F)*(((1.+v)/v)*log(1.+v)-1.))

# Helper methods to calculate effective screening angle.
cpdef double _lnchia2effdxNum(double rho, double Z1, unsigned int Z2, double A2, double gamma):
    return rho*Z2*(Z2+1)/A2*log(_chia2(Z1, Z2, A2, gamma))

cpdef double _lnchia2effdxDenom(double rho, unsigned int Z2, double A2):
    return rho*Z2*(Z2+1)/A2

# Characteristic angle
cpdef double _chic2dx(double rho, double Z1, unsigned int Z2, double A1, double A2, double gamma):
    cdef:
        double beta = sqrt(1.-1./gamma**2)
        double p = gamma*A1*u*beta*c*c/(elementary_charge*1.e6) # In MeV/c
        double X = rho*1.e-1 # Not sure if factor correct, Lynch: gm/cm2. WHAT ARE GM? mg?
    return 0.157*(Z2*(Z2+1.)*X/A2)*(Z1/(p*beta))**2

# Screening angle
cpdef double _chia2(double Z1, unsigned int Z2, double A1, double gamma):
    cdef:
        double beta = sqrt(1.-1./gamma**2)
        double p = gamma*A1*u*beta*c*c/(elementary_charge*1.e6) # In MeV/c
    return 2.007e-5*Z2**(2./3.)*(1.+3.34*(Z2*Z1*alpha/beta)**2)/p**2

# Lynch formula with radiation length. For debugging.
cpdef double _sigmaAngStraggLyRad(double Z1, double A1, double thickness, double rad, double gamma):
    cdef:
        double beta = sqrt(1.-1./gamma**2)
        double p = gamma*A1*u*beta*c*c/(elementary_charge*1.e6) # In MeV/c
    return 13.6*sqrt(thickness/rad)/p/beta*Z1*(1+0.038*log(thickness/rad))
 
# Rossi formula. Inaccurate. Used by ATIMA. For debugging.     
# Originally Rossi uses 15.0 instead of 14.1, but ATIMA uses the latter.
# This is probably due to taking the value from the (different) Highland formula
# cited by the Particle Data Group. However the PDG actually cited it wrong. It
# should be 13.9. So, several errors involved here!
cpdef double _sigmaAngStraggRo(double Z1, double A1, double thickness, double rad, double gamma):
    cdef:
        double beta = sqrt(1.-1./gamma**2)
        double p = gamma*A1*u*beta*c*c/(elementary_charge*1.e6) # In MeV/c
    return 14.1*sqrt(thickness/rad)/p/beta*Z1   # 14.1 is used by ATIMA, but not by Rossi.

# Tsai radiation length calculation
# See particle data group.
cpdef double _radLengthTs(double rho, unsigned int Z2, double A2):
    cdef:
        double lrad, lradprime, f, a
    
    a = Z2*alpha
    f = a**2*(1./(1.+a**2) + 0.20206 - 0.0369*a**2 + 0.0083*a**4 - 0.002*a**6)
    
    if Z2>=4.5:
        lrad = log(Z2**(-1./3)*184.15)
        lradprime = log(Z2**(-2./3)*1194.)
    elif Z2>=3.5:
        lrad = 4.71
        lradprime = 5.924
    elif Z2>=2.5:
        lrad = 4.74
        lradprime = 5.805
    elif Z2>=1.5:
        lrad = 4.79
        lradprime = 5.621
    else:
        lrad = 5.31    
        lradprime = 6.144    
    return 1./(4.*alpha*r_e**2*N_A/A2*(Z2**2*(lrad-f)+Z2*lradprime))*1.e-3/rho

