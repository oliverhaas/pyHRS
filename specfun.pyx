#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import  numpy
from constants cimport *
cimport cython

cdef extern from "mpfr.h":
    ctypedef struct mpfr_t:
        pass
    ctypedef unsigned int mpfr_rnd_t
    ctypedef unsigned int mpfr_prec_t
    mpfr_rnd_t MPFR_RNDN
    void mpfr_set_default_prec(mpfr_prec_t prec)
    void mpfr_init(mpfr_t x)
    void mpfr_set(mpfr_t rop, mpfr_t op, mpfr_rnd_t rnd)
    void mpfr_set_d(mpfr_t rop, double op, mpfr_rnd_t rnd)
    void mpfr_set_ui(mpfr_t rop, unsigned long int op, mpfr_rnd_t rnd)
    void mpfr_add(mpfr_t rop, const mpfr_t op1, const mpfr_t op2, mpfr_rnd_t rnd)
    void mpfr_sub(mpfr_t rop, const mpfr_t op1, const mpfr_t op2, mpfr_rnd_t rnd)
    void mpfr_mul(mpfr_t rop, const mpfr_t op1, const mpfr_t op2, mpfr_rnd_t rnd)
    void mpfr_div(mpfr_t rop, const mpfr_t op1, const mpfr_t op2, mpfr_rnd_t rnd)
    void mpfr_abs(mpfr_t rop, const mpfr_t op, mpfr_rnd_t rnd)
    int mpfr_cmp_d(const mpfr_t op1, double op2)
    double mpfr_get_d(const mpfr_t op, mpfr_rnd_t rnd)
    void mpfr_clear(mpfr_t x)
    
    
cdef extern from "math.h":
    double sqrt(double x)
    double exp(double x)
    double sin(double x)
    double tgamma(double x)
    double atan(double x)
    double atan2(double x, double y)
    double log(double x)
    double fabs(double x)
    double frexp(double x, int *exp)
    
cdef extern from "complex.h":
    double creal(double complex x)
    double cimag(double complex x)
    double cabs(double complex x)
    double complex csin(double complex x)
    double complex cexp(double complex x)
    double carg(double complex x)
    double complex clog(double complex x)
   
    
cdef double complex hyp1f1(double complex a, double b, double complex z):
    cdef:
        double acc = 1.e-15
        double complex sum, add, u1, u2, num, denom
        double errn, errnm1
        unsigned int jj, wp = 128
        double zabs = cabs(z), zreal = creal(z), zimag = cimag(z)
        double aabs = cabs(a), areal = creal(a)
        double babs = fabs(b)
        mpfr_t SRE, SIM, PRE, PIM, AREN, AIMN, BREN, ZRE, ZIM
        mpfr_t one, zero, DIV, T1, T2, T3
        
    # b is negative integer.
    if b<0. and (b%1)==0.:
        raise ValueError('b should not be a non-negative integer.')
    # Kummerer U / 2F0 representation for large z. See mpmath.
    # Currently not working probably due to limited double precision.
    # If needed use multiple precision arithmetic as below for
    # the standard series expansion to implement this here.
    elif False:#zabs>65.:
        if cimag(z)>=0:
            signiz = -1
        else:
            signiz = 1
        zi = 1./z
        d = a-b+1. 
        add = 1.
        sum = 1.
        for jj in range(1000):
            add *= -(a+jj)*(d+jj)/(jj+1.)*zi
            sum += add
            errnm1 = errn
            errn = cabs(add)/cabs(sum)
            if errn<acc and errnm1<acc:
                break
        if jj==999:
            print 'Asymptotic U/0F2 in hyp1f1 slow converging; result inaccurate.'
        u1 = z**(-a)*sum
        zi = -1./(z*cexp(1.j*pi*signiz))
        c = b-a
        e = 1.-a 
        add = 1.
        sum = 1.
        for jj in range(1000):
            add *= (c+jj)*(e+jj)/(jj+1.)*zi
            sum += add
            errnm1 = errn
            errn = cabs(add)/cabs(sum)
            if errn<acc and errnm1<acc:
                break        
        if jj==999:
            print 'Asymptotic U/0F2 in hyp1f1 slowly converging; result inaccurate.'
        u2 = (cexp(1.j*pi*signiz)*z)**(-c)*sum
        return cexp(lncgamma(b)-lncgamma(c)-signiz*a*pi*1.j)*u1 + \
               cexp(lncgamma(b)-lncgamma(a)+signiz*c*pi*1.j+z)*u2 
    # Precise series expansion.
    # Standard precision (even long double) is often not
    # enough for large z, so use mpfr library for multiple precision arithmetic.
    elif zabs>50.:      # zabs>15. recommended for accuracy, but slow. 
        mpfr_set_default_prec(wp)
        mpfr_init(SRE); mpfr_init(SIM); mpfr_init(PRE); mpfr_init(PIM);
        mpfr_init(ZRE); mpfr_init(ZIM); mpfr_init(AREN); mpfr_init(AIMN); 
        mpfr_init(BREN); mpfr_init(one); mpfr_init(DIV); mpfr_init(zero);
        mpfr_init(T1); mpfr_init(T2); mpfr_init(T3);
               
        mpfr_set_ui(zero, 0, MPFR_RNDN)
        mpfr_set_ui(one, 1, MPFR_RNDN)
        mpfr_set(SRE, one, MPFR_RNDN)
        mpfr_set(SIM, zero, MPFR_RNDN)
        mpfr_set(PRE, one, MPFR_RNDN)
        mpfr_set(PIM, zero, MPFR_RNDN)
        mpfr_set_d(AREN, creal(a), MPFR_RNDN)
        mpfr_set_d(AIMN, cimag(a), MPFR_RNDN)
        mpfr_set_d(BREN, b, MPFR_RNDN)
        mpfr_set_d(ZRE, creal(z), MPFR_RNDN)
        mpfr_set_d(ZIM, cimag(z), MPFR_RNDN)

        for jj in range(1,1000):
            mpfr_set_ui(DIV, jj, MPFR_RNDN)
            mpfr_div(PRE, PRE, BREN, MPFR_RNDN)
            mpfr_div(PIM, PIM, BREN, MPFR_RNDN)
            
            mpfr_mul(T1, PRE, AREN, MPFR_RNDN)
            mpfr_mul(T2, PIM, AIMN, MPFR_RNDN)
            mpfr_sub(T3, T1, T2, MPFR_RNDN)
            mpfr_mul(T1, PRE, AIMN, MPFR_RNDN)
            mpfr_mul(T2, PIM, AREN, MPFR_RNDN)
            mpfr_add(PIM, T1, T2, MPFR_RNDN)
            mpfr_set(PRE, T3, MPFR_RNDN)
            
            mpfr_mul(T1, PRE, ZRE, MPFR_RNDN)
            mpfr_mul(T2, PIM, ZIM, MPFR_RNDN)
            mpfr_sub(T3, T1, T2, MPFR_RNDN)
            mpfr_mul(T1, PRE, ZIM, MPFR_RNDN)
            mpfr_mul(T2, PIM, ZRE, MPFR_RNDN)
            mpfr_add(PIM, T1, T2, MPFR_RNDN)
            mpfr_set(PRE, T3, MPFR_RNDN)
            
            mpfr_div(PRE, PRE, DIV, MPFR_RNDN)
            mpfr_div(PIM, PIM, DIV, MPFR_RNDN)
            
            mpfr_add(SRE, SRE, PRE, MPFR_RNDN)
            mpfr_add(SIM, SIM, PIM, MPFR_RNDN)
              
            errn = abs(mpfr_get_d(PRE, MPFR_RNDN)/mpfr_get_d(SRE, MPFR_RNDN))
            errnm1 = abs(mpfr_get_d(PIM, MPFR_RNDN)/mpfr_get_d(SIM, MPFR_RNDN))
            if errn<acc and errnm1<acc:
                break
            mpfr_add(BREN, BREN, one, MPFR_RNDN)
            mpfr_add(AREN, AREN, one, MPFR_RNDN)
            
        if jj>=999:
            print 'hyp1f1 precise series slowly converging; result inaccurate.' 
            
        sum = mpfr_get_d(SRE, MPFR_RNDN)+1.j*mpfr_get_d(SIM, MPFR_RNDN)
        
        mpfr_clear(SRE); mpfr_clear(SIM); mpfr_clear(PRE); mpfr_clear(PIM);
        mpfr_clear(ZRE); mpfr_clear(ZIM); mpfr_clear(AREN); mpfr_clear(AIMN); 
        mpfr_clear(BREN); mpfr_clear(one); mpfr_clear(DIV); mpfr_clear(T1); 
        mpfr_clear(T2); mpfr_clear(T3);
 
        return sum
    # Standard series expansion.
    # Slow in extreme cases and possibly inaccurate.
    # But in easy cases fastest method.
    else:   
        sum = 1.
        errn = 1.
        errrnm1 = 1.
        add = 1.
        for jj in range(200):
            add *= ((a+jj)*z/(b+jj)/(jj+1.))
            sum += add
            errnm1 = errn
            errn = cabs(add)/cabs(sum)
            if errn<acc and errnm1<acc:
                break
#         if jj==199:
#             print 'hyp1f1 standard series slowly converging; result inaccurate.'
        return sum



# Complex version of true gamma.
cdef double complex ctgamma(double complex z):
    # Coefficients used by the GNU Scientific Library, Lanczos approximation.
    cdef:
        unsigned int n = 9, g = 7
        double* p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                     771.32342877765313, -176.61502916214059, 12.507343278686905,
                     -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
        double complex x, t
    # Reflection formula
    if creal(z) < 0.5:
        return pi/(csin(pi*z)*ctgamma(1-z))
    else:
        z -= 1
        x = p[0]
        for ii in range(1, n):
            x += p[ii]/(z+ii)
        t = z+g+0.5
        return sqrt(2*pi)*t**(z+0.5)*cexp(-t)*x

# Complex version of ln(gamma).
cdef double complex lncgamma(double complex z):
    # Coefficients used by the GNU Scientific Library, Lanczos approximation.
    cdef:
        unsigned int n = 9, g = 7
        double* p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                     771.32342877765313, -176.61502916214059, 12.507343278686905,
                     -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
        double complex x, t
    # Reflection formula
    if creal(z) < 0.5:
        return clog(pi/csin(pi*z))-lncgamma(1-z)
    else: 
        z -= 1
        t = (z+g+0.5)
        t -= (z+0.5)*clog(t)
        x = p[0]
        for ii in range(1, n):
            x += p[ii]/(z+ii)
        return -t+clog(sqrt(2*pi)*x)
    
# Real version of ln(abs(gamma)).
cdef double lnabsgamma(double z):
    # Coefficients used by the GNU Scientific Library, Lanczos approximation.
    cdef:
        unsigned int n = 9, g = 7
        double* p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                     771.32342877765313, -176.61502916214059, 12.507343278686905,
                     -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
        double x, t
    # Reflection formula
    if z < 0.5:
        return log(fabs(pi/sin(pi*z)))-lnabsgamma(1-z)
    else: 
        z -= 1
        t = (z+g+0.5)
        t -= (z+0.5)*log(t)
        x = p[0]
        for ii in range(1, n):
            x += p[ii]/(z+ii)
        return -t+log(sqrt(2*pi)*x)
    

# Sign of the lnabsgamma AFTER taking exp.
cdef double lngammasign(double x):
    if x < 0.:
        return ((<int> (x%2))*2-1)
    else: 
        return 1.

# Argument of complex true gamma.
cdef double argctgamma(double complex z):
    # Coefficients used by the GNU Scientific Library, Lanczos approximation.
    cdef:
        unsigned int n = 9, g = 7
        double* p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                     771.32342877765313, -176.61502916214059, 12.507343278686905,
                     -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
        double x, y, y2, T, B, denomi
        unsigned int ii

    # Reflection formula
    if creal(z) < 0.5:
        x = -carg(csin(pi*z))-argctgamma(1-z)
        if x<-pi:
            return 2*pi+x
        elif x>=pi:
            return -2*pi+x
        else:
            return x
    else:
        z -= 1.
        x = creal(z)
        y = cimag(z)
        y2 = y*y
        T = 0.
        B = p[0]
        for ii in range(1, n):
            denomi = 1./((x+ii)**2+y2)
            T += p[ii]*denomi
            B += p[ii]*(x+ii)*denomi
        T *= -y
        return (0.5*y*log((x+g+0.5)**2+y2)+(x+0.5)*atan2(y,(x+g+0.5))-y+atan2(T,B))
       

cdef double lnabscgamma(double complex z):
 
    # Coefficients used by the GNU Scientific Library, Lanczos approximation.
    cdef:
        unsigned int n = 9, g = 7
        double* p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                     771.32342877765313, -176.61502916214059, 12.507343278686905,
                     -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
        double x, y, y2, T, B, denomi
        unsigned int ii

    # Reflection formula
    if creal(z) < 0.5:
        return log(cabs(pi/csin(pi*z)))-lnabscgamma(1-z)
    else:
        z -= 1.
        x = creal(z)
        y = cimag(z)
        y2 = y*y
        T = 0.
        B = p[0]
        for ii in range(1, n):
            denomi = 1./((x+ii)**2+y2)
            T += p[ii]*denomi
            B += p[ii]*(x+ii)*denomi
        T *= -y
        return (0.5*(x+0.5)*log((x+g+0.5)**2+y**2)-y*atan2(y,(x+g+0.5))-
                (x+g+0.5)+log(sqrt(2*pi*(T*T+B*B))))  
        
        
cdef double exp1(double x):
 
    cdef:
        unsigned int ii
        double add, sum, acc = 1.e-12

    sum = -eul-log(x)
    add = x
    for ii in range(2, 1000):
        add *= -x*(ii-1.)/(ii*ii) 
        sum += add

    return sum
