#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

from constants cimport *

cdef extern from "math.h":
    double sqrt(double x)

cpdef double ekinE0ToP(double eKin, double e0):
    return sqrt((e0+eKin)**2-e0**2)

cpdef double ekinMToP(double eKin, double m):
    return sqrt((m*c**2+eKin)**2-m**2*c**4)

cpdef double pMToEkin(double p, double m):
    return sqrt(p**2+m**2*c**4)-m*c**2

cpdef double pE0ToEkin(double p, double e0):
    return sqrt(p**2+e0**2)-e0

cpdef double dEoETodpop(double dEoE, double eKin, double e0):
    cdef double temp = ekinE0ToP(eKin, e0)
    return (ekinE0ToP(eKin*(1.+dEoE), e0)-temp)/temp

cpdef double dpopTodEoE(double dpop, double eKin, double e0):
    cdef double temp = ekinE0ToP(eKin,e0)
    return -(pE0ToEkin(temp*(1.+dpop), e0)-temp)/temp

cpdef double aToMev(double A):
    return A*u/elementary_charge*c**2/1.e6

cpdef double ekinE0ToGamma(double ekin, double e0):
    return (ekin+e0)/e0

cpdef double brhoToEkin(double brho, double e0, unsigned int Z, double A):
    return pE0ToEkin(brho*Z*1.e-6*c,e0)/A