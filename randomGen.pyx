import numpy
cimport numpy
cimport cython
from constants cimport * 
import mpmath
import specfun
import spline
import time

# GSL random number generation
cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng
    cdef gsl_rng_type* gsl_rng_mt19937    
    gsl_rng* gsl_rng_alloc(gsl_rng_type* T) nogil
    double gsl_rng_uniform(gsl_rng* r) nogil
    void gsl_rng_set(gsl_rng* r, unsigned long int seed) nogil
        
cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_gaussian_ziggurat(gsl_rng* r, double sigma) nogil

       
cdef gsl_rng* r = gsl_rng_alloc(gsl_rng_mt19937)
gsl_rng_set(r, <unsigned long> time.time()*256)
        
cdef void seed(unsigned long x):  
    gsl_rng_set(r, x)

cpdef double rand():
    return gsl_rng_uniform(r)

cpdef double randn():
    return gsl_ran_gaussian_ziggurat(r, 1.)

# Random number of cut exponential distribution. E.g. reaction
# position in a finite thickness target.
# cpdef double randce(double xmax, double lamb):#double ymax):
#     cdef double ymax = 1.-exp(-xmax*lamb)
#     return xmax*log(1.-(1.-ymax)*rand())/log(ymax)
cpdef double randce(double ymax):#double ymax):
    return log(1.-(1.-ymax)*rand())/log(ymax)

# Some standard c methods
cdef extern from "math.h":
    double exp(double x)
    double log(double x)
    double fabs(double x)
    double sin(double x)
    double cos(double x)
    double sqrt(double x)

# Schorr Computer Phys. Comm. 7 (1974) 215-224
cdef class Vavilov:

    cdef:
        double kappa, beta2, epspm, eps, omega, a, tm, tp, tq, S
        numpy.ndarray coeffspdf, coeffscdf, coeffsQuantile
        unsigned int N, M, useFastInv
        object invSpline
    
    # There is a fast inverse option useFastInv = 1 which just
    # pre-calculates a cubic spline for the inverse.
    # Fast inverse is probably not just faster but even more accurate.
    # Fourier series by Schorr suffers from heavy oscillations near 0 and 1
    # and bad convergence.
    def __init__(self, double kappa, double beta2, useFastInv = 1, M = 500):
        self.M = M
        self.useFastInv = useFastInv
        self.setParam(kappa, beta2)
        
    def setParam(self, double kappaIn, double beta2In):
        
        cdef:
            double kappa, beta2, tp, tm
            double acc = 1.e-6, epspm = 1.e-4, eps = 1.e-4    # Schorr: 5.e-4 should be 3 digit accurate.
            unsigned int N
            double omega, d
            
        if kappaIn<0.001:
            print 'Parameter kappa too small. Better use Landau distribution.'
            print 'Setting kappa = 0.001.'
            kappa = 0.001
        elif kappaIn>10.:
            print 'Parameter kappa too large. Better use Gaussian distribution.'
            print 'Setting kappa = 10.'
            kappa = 10.
        else:
            kappa = kappaIn
        
        if beta2In<0.:
            print 'Parameter beta2 too small.'
            print 'Setting beta2 = 0.'
            beta2 = 0.
        elif beta2In>1.:
            print 'Parameter beta2 too large.'
            print 'Setting beta2 = 1.'
            beta2 = 1.
        else:
            beta2 = beta2In
            
        self.kappa = kappa
        self.beta2 = beta2        
        self.tm = _calcTm(beta2, kappa, epspm)
        self.tp = _calcTp(beta2, kappa, epspm)
        self.omega = 2.*pi/(self.tp-self.tm)
        self.N = _calcN(beta2, kappa, self.omega, eps)
        self.coeffspdf = _calcCoeffsPDF(beta2, kappa, self.omega, self.N)        
        self.coeffscdf = _calcCoeffsCDF(self.coeffspdf, self.N)
        self.a = _calcA(self.tm, self.coeffscdf, self.omega, self.N)
        if self.useFastInv == 0:
            self.S = _calcS(kappa)
            self.tq = _calcTq(self.coeffscdf, self.tm, self.tp, self.omega, self.N, self.a, self.S)
            self.coeffsQuantile = _calcCoeffsQuantile(self.coeffscdf, kappa, beta2, self.S, self.omega, self.N, self.a, self.M, self.tm, self.tq)
        else:
            self.invSpline = _makeInvSpline(self.coeffscdf, self.tm, self.tp, self.omega, self.N, self.a, self.M)

    cpdef double pdf(self, double x):
        if x<self.tm:
            return 0.
        elif x>self.tp:
            return 0.
        else:
            return _gFourier(x, self.coeffspdf, self.omega, self.N)
    
    cpdef double cdf(self, double x):
        if x<self.tm:
            return 0.
        elif x>self.tp:
            return 1.
        else:
            return _GFourier(x, self.coeffscdf, self.omega, self.N, self.a)
        
    cpdef double quantile(self, double x):
        if self.useFastInv==1:
            if x<=0.:
                return self.tm
            elif x>=1.:
                return self.tp
            else:
                return self.invSpline.interpolate(x)
        else:
            if x<=0.:
                return self.tm
            elif x>=1.:
                return self.tp
            else:
                return _QsFourier(x, self.coeffsQuantile, self.M, self.tq, self.tm)   
            
    cpdef double rand(self):
        return self.quantile(rand())
    
    cpdef double mean(self):
        return eul-1.-log(self.kappa)-self.beta2

    cpdef double variance(self):
        return (1.-self.beta2*0.5)/self.kappa
    
    cpdef double std(self):
        return sqrt(self.variance())
    
    cpdef double mode(self):
        return _findMaxPDF(self.coeffspdf, self.tm, self.tp, self.omega, self.N)
    

    
cdef double _calcTm(double beta2, double kappa, double epspm):
    cdef:
        double xm, f, fd, fdd, add
        double acc = 1.e-12
    # Use approximate value proposed by Schorr, 
    # but do few (typically 2) Halley iterations.
    xm = 1.-beta2*(1.-eul)-1./kappa*log(epspm)
    for ii in range(100):
        f = (1.-beta2)*exp(-xm)-beta2*(log(fabs(xm))+mpmath.e1(xm)) - \
            1.+beta2*(1.-eul)+1./kappa*log(epspm)+xm
        fd = -(1.-beta2)*exp(-xm)-beta2*(1.-exp(-xm))/xm+1.
        fdd = (1.-beta2)*exp(-xm)+beta2/xm**2-exp(-xm)*(1./xm**2+1./xm)
        add = -2*f*fd/(2.*fd*fd-f*fdd)
        xm += add
        if fabs(add)<fabs(xm)*acc:
            break   
    return 1./xm*(1./kappa*log(epspm)-1.-beta2*eul-xm*log(kappa)+
                  exp(-xm)-(xm+beta2)*(log(xm)+mpmath.e1(xm)))       

    
cdef double _calcTp(double beta2, double kappa, double epspm):
    cdef:
        double f, fd, fdd, add, xp    
        double acc = 1.e-12
    # Initial guess from a (bad) fitting function. 
    # Don't know what else one could do. 
    # Works well for the set epspm ~= 0.0001.
    xp = 7.89896-10.5447*kappa**(-0.0733286)-0.603317*beta2**1.65352
    for ii in range(100):
        if xp>0.: # Not trusting that this doesn't happen. Fail safe.
            xp = -0.5
        f = (1.-beta2)*exp(-xp)-beta2*(log(fabs(xp))-mpmath.ei(-xp)) - \
            1.+beta2*(1.-eul)+1./kappa*log(epspm)+xp
        fd = -(1.-beta2)*exp(-xp)-beta2*(1.-exp(-xp))/xp+1.
        fdd = (1.-beta2)*exp(-xp)+beta2/xp**2-exp(-xp)*(1./xp**2+1./xp)
        add = -2*f*fd/(2.*fd*fd-f*fdd)
        xp += add
        if fabs(add)<fabs(xp)*acc:
            break    
    return 1./xp*(1./kappa*log(epspm)-1.-beta2*eul-xp*log(kappa)+
                  exp(-xp)-(xp+beta2)*(log(-xp)-mpmath.ei(-xp)))    
        

cdef unsigned int _calcN(double beta2, double kappa, double omega, double eps):
    cdef:
        double d, nDouble, f, fd, fdd, add
        double acc = 1.e-14
        unsigned int ii    
    d = 2./pi**2*(omega/kappa)**(beta2*kappa)*exp(kappa*(2.+beta2*eul))
    # Again a fit for rough starting value.
    nDouble = 3.78/omega+2.47*beta2+1.*kappa-1.33
    test = nDouble
    if nDouble<5:
        nDouble = 5.
    for ii in range(100):
        f = d*nDouble**(beta2*kappa)*exp(-0.5*pi*omega*nDouble)-eps
        fd = d*(beta2*exp(-0.5*pi*omega*nDouble)*kappa*nDouble**(beta2*kappa-1)-
                0.5*exp(-0.5*pi*omega*nDouble)*nDouble**(beta2*kappa)*omega*pi)
        fdd = d*(beta2*exp(-0.5*nDouble*omega*pi)*kappa*(-1.+beta2*kappa)*nDouble**(-2.+beta2*kappa) -
                 beta2*exp(-0.5*nDouble*omega*pi)*kappa*nDouble**(-1.+beta2*kappa)*omega*pi +
                 0.25*exp(-0.5*nDouble*omega*pi)*nDouble**(beta2*kappa)*omega*omega*pi*pi)
        add = -2*f*fd/(2.*fd*fd-f*fdd)
        nDouble += add
        if fabs(add)<fabs(nDouble)*acc:
            break  
    return <unsigned int> (nDouble+1.)


cdef numpy.ndarray[numpy.double_t] _calcCoeffsPDF(double beta2, double kappa, double omega, unsigned int N):
    cdef:
        numpy.ndarray[numpy.double_t] coeffspdfNumpy = numpy.empty(2*N)
        double* coeffspdf = &coeffspdfNumpy[0]
        double temp00, temp01, temp02, temp03, temp04, temp05
        unsigned int ii
    temp05 = exp(kappa*(1.+beta2*eul))
    for ii in range(1,N+1):
        temp00 = ii*omega/kappa
        temp01 = log(temp00)-mpmath.ci(temp00)
        temp02 = mpmath.si(temp00)
        temp03 = beta2*kappa*temp01 - ii*omega*temp02 - kappa*cos(temp00)
        temp04 = ii*omega*log(kappa) + ii*omega*temp01 + beta2*kappa*temp02 + kappa*sin(temp00)
        coeffspdf[2*(ii-1)] = temp05*exp(temp03)*cos(temp04)
        coeffspdf[2*(ii-1)+1] = -temp05*exp(temp03)*sin(temp04)  
    return coeffspdfNumpy

cdef numpy.ndarray[numpy.double_t] _calcCoeffsCDF(double[:] coeffspdf, unsigned int N):
    cdef:
        numpy.ndarray[numpy.double_t] coeffscdfNumpy = numpy.empty(2*N)
        double* coeffscdf = &coeffscdfNumpy[0]
        unsigned int ii
    for ii in range(N):
        coeffscdf[2*ii] = coeffspdf[2*ii]/(ii+1)
        coeffscdf[2*ii+1] = coeffspdf[2*ii+1]/(ii+1)
    return coeffscdfNumpy

cdef double _calcTq(double[:] coeffscdf, double tm, double tp, double omega, unsigned int N, double a, double S):
    cdef:
        double x0, x1, x2, fmS, acc = 1.e-12
        unsigned int ii
    # Find the value corresponding to the quantile.
    # Newton probably faster, but not sure if always converges.
    err = 1.; x0 = tm; x2 = tp; 
    for ii in range(100):
        x1 = (x2+x0)*0.5
        fmS = _GFourier(x1, coeffscdf, omega, N, a) - S
        if fmS<0.:
            x0 = x1
            err = (x2-x1)/x1
        elif fmS>0.:
            x2 = x1
            err = (x1-x0)/x1
        else:
            break   
        if err<acc:
            break 
    return (x2+x0)*0.5
    
cdef numpy.ndarray[numpy.double_t] _calcCoeffsQuantile(double[:] coeffscdf, double kappa, double beta2, double S, double omega, 
                                                       unsigned int N, double a, unsigned int M, double tm, double tq):
    cdef:
        numpy.ndarray[numpy.double_t] coeffsQuantileNumpy = numpy.empty(M)
        double* coeffsQ = &coeffsQuantileNumpy[0]
        unsigned int ii, nReq
        double h, temp, sum

    for ii in range(1,M+1):
        temp = ii*pi/S
        nReq = <unsigned int> (ii*(5*(1-tq/tm)+1.)+10)
        nReq -= (nReq % 3)
        h = (tq-tm)/nReq
        sum = cos(temp*_GFourier(tm, coeffscdf, omega, N, a))
        for jj in range(1,<unsigned int> nReq/3):
            sum += 3*cos(temp*_GFourier(tm+h*(3*jj-2), coeffscdf, omega, N, a)) + \
                   3*cos(temp*_GFourier(tm+h*(3*jj-1), coeffscdf, omega, N, a)) + \
                   2*cos(temp*_GFourier(tm+h*3*jj, coeffscdf, omega, N, a))
        sum += 3*cos(temp*_GFourier(tm+h*(nReq-2), coeffscdf, omega, N, a)) + \
               3*cos(temp*_GFourier(tm+h*(nReq-1), coeffscdf, omega, N, a)) + \
               cos(temp*_GFourier(tm+h*nReq, coeffscdf, omega, N, a))
        coeffsQ[ii-1] = 2./ii/pi*3./8.*h*sum
    return coeffsQuantileNumpy
        
     
cdef double _gFourier(double x, double[:] coeffspdf, double omega, unsigned int N):
    cdef:
        double sum = 0.
        unsigned int kk  
    for kk in range(N):
        sum += coeffspdf[2*kk]*cos((kk+1)*omega*x) + coeffspdf[2*kk+1]*sin((kk+1)*omega*x)
    return omega/pi*(0.5 + sum)


cdef double _GFourier(double x, double[:] coeffscdf, double omega, unsigned int n, double a):   
    cdef:
        double sum = 0.
        unsigned int kk
    for kk in range(n):
        sum += coeffscdf[2*kk]*sin((kk+1)*omega*x) - coeffscdf[2*kk+1]*cos((kk+1)*omega*x) 
    return 1/pi*(a + 0.5*omega*x + sum)
  
cdef double _calcA(double tMinus, double[:] coeffscdf, double omega, unsigned int n):  
    cdef:
        double sum = 0.
        unsigned int kk     
    for kk in range(n):
        sum += coeffscdf[2*kk]*sin((kk+1)*omega*tMinus) - coeffscdf[2*kk+1]*cos((kk+1)*omega*tMinus)
    return -0.5*omega*tMinus - sum    

cdef double _QsFourier(double x, double[:] coeffsQuantile, unsigned int m, double tq, double tm):   
    cdef:
        double sum = tm + (tq-tm)*x
        unsigned int kk
    for kk in range(m):
        sum += coeffsQuantile[kk]*sin((kk+1)*pi*x)
    return sum

cdef double _calcS(kappa):
    return 0.999 + 1.e-6*kappa

cdef object _makeInvSpline(double[:] coeffscdf, double tm, double tp, double omega, unsigned int N, double a, unsigned int M):

    cdef:
        numpy.ndarray[numpy.double_t] tVal = numpy.linspace(tm, tp, M)
        numpy.ndarray[numpy.double_t] GVal = numpy.empty(M)
        unsigned int ii
    for ii in range(M):
        GVal[ii] = _GFourier(tVal[ii], coeffscdf, omega, N, a)
    return spline.Spline1D(GVal, tVal)


cdef double _findMaxPDF(double[:] coeffspdf, double tm, double tp, double omega, unsigned int N):
    cdef:
        double x1, x2, x3, x4
        double f1, f2, f3, f4
        double grp1i = 1./(1. + (1.+sqrt(5.))*0.5)
        double acc = 1.e-9
        unsigned int ii
        double num, denom
    
    # A little bit of golden section search for initial stability, 
    # then parabolic interpolation for faster convergence.
    # See http://en.wikipedia.org/wiki/Golden_section_search.
    # Added parabolic interpolation http://linneus20.ethz.ch:8080/1_5_2.html.
    x1 = tm; x3 = tp; x2 = grp1i*(x3+x1)
    f1 = _gFourier(x1, coeffspdf, omega, N) 
    f2 = _gFourier(x2, coeffspdf, omega, N)   
    f3 = _gFourier(x3, coeffspdf, omega, N)
    for ii in range(5):
        x4 = x1 + (x3 - x2)
        f4 = _gFourier(x4, coeffspdf, omega, N)  
        if x4>x2:
            if f4>f2:
                x1 = x2; f1 = f2;
                x2 = x4; f2 = f4;
            else:
                x3 = x4; f3 = f4;        
        else:
            if f4>f2:
                x3 = x2; f3 = f2;
                x2 = x4; f2 = f4;
            else:
                x1 = x4; f1 = f4;
    for ii in range(100):
        num = ((x2-x1)**2*(f2-f3)-(x2-x3)**2*(f2-f1))
        denom = (x2-x1)*(f2-f3)-(x2-x3)*(f2-f1)
        if denom==0.:
            break
        x4 = x2 - 0.5*num/denom
        if fabs(x3-x1)<acc*fabs(x4):
            break
        f4 = _gFourier(x4, coeffspdf, omega, N)  
        if x4>x2:
            if f4>f2:
                x1 = x2; f1 = f2;
                x2 = x4; f2 = f4;
            else:
                x3 = x4; f3 = f4;        
        else:
            if f4>f2:
                x3 = x2; f3 = f2;
                x2 = x4; f2 = f4;
            else:
                x1 = x4; f1 = f4;
    return 0.5*(x3+x1)
        
        


