# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy
cimport numpy
cimport cython

cdef extern from "math.h":
    double log(double x)
    double fabs(double x)
    
# cdef inline unsigned int max(unsigned int x, unsigned int y):
#     if x>y:
#         return x
#     else: 
#         return y 
        
# Cubic spline class.
# See http://en.wikipedia.org/wiki/Spline_interpolation
cdef class Spline1D:
    cdef:
        double[:] x, y
        double[::1] k
        unsigned int nPoints, equidistant
        
    def __init__(self, double[:] x, double[:] y, unsigned int equidistant = 0):
        self.make(x, y, equidistant = equidistant)
    
    # Make the spline. Mainly calculates spline coefficients.
    cpdef make(self, double[:] x, double[:] y, unsigned int equidistant = 0):  
        self.equidistant = equidistant
        self.nPoints = self.y.shape[0]
        if self.equidistant == 0:
            self.x = x
            self.y = y
            self.k = _calcCoeffs1D(self.x, self.y, self.nPoints)
        else:
            self.x = x[:2]
            self.y = y
            self.k = _calcCoeffs1DEqui(self.x, self.y, self.nPoints)
            
    cpdef double interpolate(self, double xVal):
        if self.equidistant == 0:
            return _interpolate1D(xVal, self.x, self.y, self.k, self.nPoints)
        else:
            return _interpolate1DEqui(xVal, self.x, self.y, self.k, self.nPoints)
        
    cpdef numpy.ndarray getCoeffs(self):
        return numpy.asarray(self.k)
        

# Calculate cubic spline coefficients.
cdef double[::1] _calcCoeffs1D(double[:] x, double[:] y, unsigned int nPoints):
    cdef:
        double[::1] a = numpy.empty(nPoints, dtype=numpy.double)
        double[::1] b = numpy.empty(nPoints, dtype=numpy.double)
        double[::1] c = numpy.empty(nPoints, dtype=numpy.double)
        double[::1] d = numpy.empty(nPoints, dtype=numpy.double)
        double[::1] k = numpy.empty(nPoints, dtype=numpy.double)
        unsigned int ii
        double dxiNew, dxiOld, dyOld, dyNew  
    dxiNew = 1./(x[1]-x[0])    
    dyNew = y[1]-y[0]
    a[0] = 0.;  b[0] = 2.*dxiNew; c[0] = dxiNew; d[0] = 3.*dyNew*dxiNew*dxiNew;
    for ii in range(1,nPoints-1):
        dxiOld = dxiNew
        dxiNew = 1./(x[ii+1]-x[ii])
        dyOld = dyNew
        dyNew = y[ii+1]-y[ii]
        a[ii] = dxiOld
        b[ii] = 2.*(dxiNew+dxiOld)
        c[ii] = dxiNew
        d[ii] = 3.*(dyOld*dxiOld*dxiOld + dyNew*dxiNew*dxiNew)  
    a[nPoints-1] = dxiNew;  b[nPoints-1] = 2.*dxiNew; c[nPoints-1] = 0.; d[nPoints-1] = 3.*dyNew*dxiNew*dxiNew;
    _TDMASolve(a, b, c, d, k, nPoints)
    return k


cdef double _interpolate1D(double xVal, double[:] x, double[:] y, double[::1] k, unsigned int nPoints):
    cdef:
        unsigned int ii, n0, n2, n1, iterMax, n0test
        double t, onemt, dx, dy      
    iterMax = <unsigned int> (log(nPoints)*1.4427+5.)   # 1.4427 = 1./log(2.)
    n0 = 0; n2 = nPoints;
    # Binary search for the correct interval.
    for ii in range(iterMax):
        if (n2-n0)==1:
            break
        n1 = <unsigned int> (n2+n0)/2
        if xVal>x[n1]:
            n0 = n1
        else:
            n2 = n1
    dx = x[n2]-x[n0]
    dy = y[n2]-y[n0] 
    t = (xVal-x[n0])/dx
    onemt = 1-t
    return onemt*y[n0] + t*y[n2] + t*onemt*((k[n0]*dx-dy)*onemt + (-k[n2]*dx+dy)*t)

# Calculate cubic spline coefficients.
cdef double[::1] _calcCoeffs1DEqui(double[:] x, double[:] y, unsigned int nPoints):

    cdef:
        double[::1] a = numpy.empty(nPoints-2, dtype=numpy.double)
        double[::1] b = numpy.empty(nPoints-2, dtype=numpy.double)
        double[::1] c = numpy.empty(nPoints-2, dtype=numpy.double)
        double[::1] d = numpy.empty(nPoints-2, dtype=numpy.double)
        double[::1] k = numpy.empty(nPoints+2, dtype=numpy.double)
        unsigned int ii   
    k[1] = y[0]/6.  
    a[0] = 0.;  b[0] = 4.; c[0] = 1.; d[0] = y[1]-k[1];
    for ii in range(1,nPoints-3):
        a[ii] = 1.
        b[ii] = 4.
        c[ii] = 1.
        d[ii] = y[ii+1] 
    k[nPoints] = y[nPoints-1]/6.
    a[nPoints-3] = 1.;  b[nPoints-3] = 4.; c[nPoints-3] = 0.; d[nPoints-3] = y[nPoints-2]-k[nPoints];
    _TDMASolve(a, b, c, d, k[2:nPoints], nPoints-2)
    k[0] = 2.*k[1]-k[2] 
    k[nPoints+1] = 2.*k[nPoints]-k[nPoints-1]
    return k


cdef double _interpolate1DEqui(double xVal, double[:] x, double[:] y, double[::1] k, unsigned int nPoints):
    cdef:
        unsigned int ind
        double tabs, dxi, tred, res        
    dxi = (nPoints-1)/(x[1]-x[0])
    tred = (xVal-x[0])*dxi
    ind = <unsigned int> tred
    tabs = fabs(tred - ind)
    res = k[ind+1]*(4.-6.*tabs**2+3.*tabs**3)
    tabs = fabs(tred - ind - 1)
    res += k[ind+2]*(4.-6.*tabs**2+3.*tabs**3)
    tabs = fabs(tred - ind + 1)
    res += k[ind]*(2.-tabs)**3
    tabs = fabs(tred - ind - 2)
    res += k[ind+3]*(2.-tabs)**3
    return res

# Tridiagonal matrix solver.
# No pivoting, but should be okay. Modifies c and d.
# From http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm.
cdef void _TDMASolve(double[::1] a, double[::1] b, double[::1] c, double[::1] d, double[:] k, unsigned int n):
    cdef unsigned int ii, jj
    c[0] = c[0]/b[0]
    d[0] = d[0]/b[0]
    for ii in range(1,n-1):
        c[ii] /= (b[ii]-c[ii-1]*a[ii])
        d[ii] -= d[ii-1]*a[ii]
        d[ii] /= (b[ii]-c[ii-1]*a[ii]) 
    d[n-1] -= d[n-2]*a[n-1]
    d[n-1] /= (b[n-1]-c[n-2]*a[n-1])
    k[n-1] = d[n-1]
    for ii in range(n-1):
        jj = n-2-ii
        k[jj] = d[jj]-c[jj]*k[jj+1]

    

# Cubic spline class equidistant grid.
# Based on 
# Habermann and Kindermann 
# Multidimensional Spline Interpolation: Theory and Applications
# Computational Economics 30 2 2007 pp 153-169    
cdef class Spline2D:
    cdef:
        double[:] x, y, z
        double[::1] k
        unsigned int nx, ny, np
        
    def __init__(self, double[:] x, double[:] y, double[:] z):
        self.make(x, y, z)
    
    # Make the spline. Mainly calculates spline coefficients.
    cpdef make(self, double[:] x, double[:] y, double[:] z):  
        self.x = x[:2]
        self.y = y[:2]
        self.z = z
        self.nx = <unsigned int> x[2]
        self.ny = <unsigned int>  y[2]
        self.np = self.z.shape[0]
        self.k = _calcCoeffs2DEqui(self.x, self.y, self.z, self.nx, self.ny, self.np)
            
    cpdef double interpolate(self, double xVal, double yVal):
        return _interpolate2DEqui(xVal, yVal, self.x, self.y, self.k, self.nx, self.ny, self.np)
         
    cpdef numpy.ndarray getCoeffs(self):
        return numpy.asarray(self.k)

# Calculate cubic spline coefficients. Equidistant grid.
cdef double[::1] _calcCoeffs2DEqui(double[:] x, double[:] y, double[:] z, unsigned int nx, unsigned int ny, unsigned int np):
    cdef:
        unsigned int maxnxny = max(nx,ny)
        unsigned int maxnxnym2 = maxnxny-2
        unsigned int ii, jj, nxp2 = nx+2, nyp2 = ny+2, nxm2 = nx-2, nym2 = ny-2
        unsigned int nxm3 = nx-3, nym3 = ny-3, jjnxp2, jjnx
        double[::1] a = numpy.empty(maxnxnym2, dtype=numpy.double)
        double[::1] b = numpy.empty(maxnxnym2, dtype=numpy.double)
        double[::1] c = numpy.empty(maxnxnym2, dtype=numpy.double)
        double[::1] d = numpy.empty(maxnxnym2, dtype=numpy.double)
        double[::1] kTemp = numpy.empty(nxp2*ny, dtype=numpy.double)
        double[::1] k = numpy.empty(nxp2*nyp2, dtype=numpy.double)
    for ii in range(nxm3):
        a[ii] = 1.
        b[ii] = 4.        
    a[0] = 0.;  b[0] = 4.; a[nxm3] = 1.;  b[nxm3] = 4.;   
    for jj in range(ny):    
        for ii in range(nxm3):
            c[ii] = 1.   
        c[0] = 1.; c[nxm3] = 0.;
        jjnxp2 = jj*nxp2; jjnx = jj*nx
        kTemp[jjnxp2+1] = z[jjnx]/6.  
        d[0] = z[jjnx+1]-kTemp[jjnxp2+1]
        for ii in range(1,nxm3):
            d[ii] = z[jjnx+ii+1] 
        kTemp[jjnxp2+nx] = z[jjnx+nx-1]/6.
        d[nxm3] = z[jjnx+nx-2]-kTemp[jjnxp2+nx]
        _TDMASolve(a[:nxm2], b[:nxm2], c[:nxm2], d[:nxm2], kTemp[(jjnxp2+2):(jjnxp2+nx)], nxm2)
        kTemp[jjnxp2] = 2.*kTemp[jjnxp2+1]-kTemp[jjnxp2+2] 
        kTemp[jjnxp2+nx+1] = 2.*kTemp[jjnxp2+nx]-kTemp[jjnxp2+nx-1]
    for jj in range(nym3):
            a[jj] = 1.
            b[jj] = 4.
    a[0] = 0.;  b[0] = 4.; a[nym3] = 1.;  b[nym3] = 4.;
    for ii in range(nxp2):
        for jj in range(ny-3):
                c[jj] = 1.  
        c[0] = 1.; c[nym3] = 0.; 
        k[nxp2+ii] = kTemp[ii]/6.  
        d[0] = kTemp[nxp2+ii]-k[nxp2+ii];
        for jj in range(1,nym3):
            d[jj] = kTemp[(jj+1)*nxp2+ii] 
        k[ny*nxp2 + ii] = kTemp[(ny-1)*nxp2+ii]/6.
        d[nym3] = kTemp[nym2*nxp2+ii]-k[ny*nxp2+ii]
        _TDMASolve(a[:nym2], b[:nym2], c[:nym2], d[:nym2], k[(2*nxp2+ii):(ny*nxp2+ii):nxp2], nym2)
        k[ii] = 2.*k[nxp2+ii]-k[2*nxp2+ii] 
        k[(ny+1)*nxp2+ii] = 2.*k[ny*nxp2+ii]-k[(ny-1)*nxp2+ii]
    return k

cdef double _interpolate2DEqui(double xVal, double yVal, double[:] x, double[:] y, double[::1] k, 
                               unsigned int nx, unsigned int ny, unsigned int np):
    cdef:
        unsigned int indx, indy, ii, jj, nxp2 = nx+2
        double dxi, dyi, txred, tyred, res
        double[4] txabs
        double[4] tyabs
        double[4] fx
        double[4] fy   
    if xVal<x[0]: xVal = x[0]
    elif xVal>x[1]: xVal = x[1]
    if yVal<y[0]: yVal = y[0]
    elif yVal>y[1]: yVal = y[1]
    dxi = (nx-1)/(x[1]-x[0])
    dyi = (ny-1)/(y[1]-y[0])
    txred = (xVal-x[0])*dxi
    tyred = (yVal-y[0])*dyi
    indx = <unsigned int> txred
    indy = <unsigned int> tyred
    if indx>=nx-1: indx = nx-2
    if indy>=ny-1: indy = ny-2
    for ii in range(4):
        txabs[ii] = fabs(txred - indx + 1 - ii)
        tyabs[ii] = fabs(tyred - indy + 1 - ii)        
    fx[0] = (2.-txabs[0])**3
    fx[1] = (4.-6.*txabs[1]**2+3.*txabs[1]**3)
    fx[2] = (4.-6.*txabs[2]**2+3.*txabs[2]**3)
    fx[3] = (2.-txabs[3])**3   
    fy[0] = (2.-tyabs[0])**3
    fy[1] = (4.-6.*tyabs[1]**2+3.*tyabs[1]**3)
    fy[2] = (4.-6.*tyabs[2]**2+3.*tyabs[2]**3)
    fy[3] = (2.-tyabs[3])**3
    res = 0.
    for ii in range(4):
        for jj in range(4):
            res += k[(indy+jj)*nxp2+ii+indx]*fx[ii]*fy[jj]
    return res
 

# Cubic spline class.
# Based on the 1D Spline above and
# Habermann and Kindermann 
# Multidimensional Spline Interpolation: Theory and Applications
# Computational Economics 30 2 2007 pp 153-169    
cdef class Spline3D:

    # Make the spline. Mainly calculates spline coefficients.
    cdef void make(self, x, y, z, double[:,:,:] f):        
        self.x = numpy.array(x[:2])
        self.y = numpy.array(y[:2])
        self.z = numpy.array(z[:2])
        self.f = f
        self.nx = <unsigned int> x[2]
        self.ny = <unsigned int> y[2]
        self.nz = <unsigned int> z[2]
        self.np = self.f.shape[0]
        self.k = _calcCoeffs3DEqui(self.x, self.y, self.z, self.f, self.nx, self.ny, self.nz)
        
    # Load the spline.
    cdef void load(self, folderName):        
        self.x = numpy.load(folderName + '/x.npy')[:2]
        self.y = numpy.load(folderName + '/y.npy')[:2]
        self.z = numpy.load(folderName + '/z.npy')[:2]     
        self.k = numpy.load(folderName + '/k.npy')
        self.nx = <unsigned int> numpy.load(folderName + '/x.npy')[2]
        self.ny = <unsigned int> numpy.load(folderName + '/y.npy')[2]
        self.nz = <unsigned int> numpy.load(folderName + '/z.npy')[2]
        self.np = self.nx*self.ny*self.nz
        
    cdef void save(self, folderName):        
        numpy.save(folderName + '/x.npy', numpy.array([self.x[0], self.x[1], self.nx]))
        numpy.save(folderName + '/y.npy', numpy.array([self.y[0], self.y[1], self.ny]))
        numpy.save(folderName + '/z.npy', numpy.array([self.z[0], self.z[1], self.nz]))    
        numpy.save(folderName + '/k.npy', self.k)  
            
    cdef double interpolate(self, double xVal, double yVal, double zVal):
        return _interpolate3DEqui(xVal, yVal, zVal, self.x, self.y, self.z, self.k, self.nx, self.ny, self.nz)


# Calculate cubic spline coefficients. Equidistant grid.
cdef double[:,:,:] _calcCoeffs3DEqui(double[:] x, double[:] y, double[:] z, double[:,:,:] f, 
                                      unsigned int nx, unsigned int ny, unsigned int nz):
    cdef:
        unsigned int maxnxnynz = max(max(nx,ny),nz)
        unsigned int maxnxnynzm2 = maxnxnynz-2
        unsigned int ii, jj, kk
        unsigned int nxp2 = nx+2, nyp2 = ny+2, nzp2 = nz+2
        unsigned int nxm2 = nx-2, nym2 = ny-2, nzm2 = nz-2
        unsigned int nxm3 = nx-3, nym3 = ny-3, nzm3 = nz-3  
        double[::1] a = numpy.empty(maxnxnynzm2, dtype=numpy.double)
        double[::1] b = numpy.empty(maxnxnynzm2, dtype=numpy.double)
        double[::1] c = numpy.empty(maxnxnynzm2, dtype=numpy.double)
        double[::1] d = numpy.empty(maxnxnynzm2, dtype=numpy.double)
        double[:,:,:] kTemp1 = numpy.empty((nxp2,ny,nz), dtype=numpy.double)
        double[:,:,:] kTemp2 = numpy.empty((nxp2,nyp2,nz), dtype=numpy.double)
        double[:,:,:] k = numpy.empty((nxp2,nyp2,nzp2), dtype=numpy.double)
    for ii in range(nxm3):
        a[ii] = 1.
        b[ii] = 4.        
    a[0] = 0.;  b[0] = 4.; a[nxm3] = 1.;  b[nxm3] = 4.;   
    for kk in range(nz):
        for jj in range(ny):    
            for ii in range(nxm3):
                c[ii] = 1.   
            c[0] = 1.; c[nxm3] = 0.;
            kTemp1[1,jj,kk] = f[0,jj,kk]/6.  
            d[0] = f[1,jj,kk]-kTemp1[1,jj,kk]
            for ii in range(1,nxm3):
                d[ii] = f[ii+1,jj,kk] 
            kTemp1[nx,jj,kk] = f[nx-1,jj,kk]/6.
            d[nxm3] = f[nxm2,jj,kk]-kTemp1[nx,jj,kk]
            _TDMASolve(a[:nxm2], b[:nxm2], c[:nxm2], d[:nxm2], kTemp1[2:nx,jj,kk], nxm2)
            kTemp1[0,jj,kk] = 2.*kTemp1[1,jj,kk]-kTemp1[2,jj,kk] 
            kTemp1[nx+1,jj,kk] = 2.*kTemp1[nx,jj,kk]-kTemp1[nx-1,jj,kk]
           
    for jj in range(nym3):
        a[jj] = 1.
        b[jj] = 4.
    a[0] = 0.;  b[0] = 4.; a[nym3] = 1.;  b[nym3] = 4.;
    for kk in range(nz):
        for ii in range(nxp2):    
            for jj in range(nym3):
                c[jj] = 1.   
            c[0] = 1.; c[nym3] = 0.;
            kTemp2[ii,1,kk] = kTemp1[ii,0,kk]/6.  
            d[0] = kTemp1[ii,1,kk]-kTemp2[ii,1,kk]
            for jj in range(1,nym3):
                d[jj] = kTemp1[ii,jj+1,kk] 
            kTemp2[ii,ny,kk] = kTemp1[ii,ny-1,kk]/6.
            d[nym3] = kTemp1[ii,nym2,kk]-kTemp2[ii,ny,kk]
            _TDMASolve(a[:nym2], b[:nym2], c[:nym2], d[:nym2], kTemp2[ii,2:ny,kk], nym2)
            kTemp2[ii,0,kk] = 2.*kTemp2[ii,1,kk]-kTemp2[ii,2,kk] 
            kTemp2[ii,ny+1,kk] = 2.*kTemp2[ii,ny,kk]-kTemp2[ii,ny-1,kk]

    for kk in range(nzm3):
        a[kk] = 1.
        b[kk] = 4.
    a[0] = 0.;  b[0] = 4.; a[nzm3] = 1.;  b[nzm3] = 4.;
    for jj in range(nyp2):
        for ii in range(nxp2):    
            for kk in range(nzm3):
                c[kk] = 1.   
            c[0] = 1.; c[nzm3] = 0.;
            k[ii,jj,1] = kTemp2[ii,jj,0]/6.  
            d[0] = kTemp2[ii,jj,1]-k[ii,jj,1]
            for kk in range(1,nzm3):
                d[kk] = kTemp2[ii,jj,kk+1] 
            k[ii,jj,nz] = kTemp2[ii,jj,nz-1]/6.
            d[nzm3] = kTemp2[ii,jj,nzm2]-k[ii,jj,nz]
            _TDMASolve(a[:nzm2], b[:nzm2], c[:nzm2], d[:nzm2], k[ii,jj,2:nz], nzm2)
            k[ii,jj,0] = 2.*k[ii,jj,1]-k[ii,jj,2] 
            k[ii,jj,nz+1] = 2.*k[ii,jj,nz]-k[ii,jj,nz-1]
    return k

cdef double _interpolate3DEqui(double xVal, double yVal, double zVal, double[:] x, double[:] y, 
                               double[:] z, double[:,:,:] k, unsigned int nx, unsigned int ny, unsigned int nz):
    cdef:
        unsigned int indx, indy, indz, ii, jj, kk
        double dxi, dyi, dzi, txred, tyred, tzred, res
        double[4] txabs, tyabs, tzabs, fx, fy, fz   
    if xVal<x[0]: xVal = x[0]
    elif xVal>x[1]: xVal = x[1]
    if yVal<y[0]: yVal = y[0]
    elif yVal>y[1]: yVal = y[1]
    if zVal<z[0]: zVal = z[0]
    elif zVal>z[1]: zVal = z[1]
    dxi = (nx-1.)/(x[1]-x[0])
    dyi = (ny-1.)/(y[1]-y[0])
    dzi = (nz-1.)/(z[1]-z[0])
    txred = (xVal-x[0])*dxi
    tyred = (yVal-y[0])*dyi
    tzred = (zVal-z[0])*dzi
    indx = <unsigned int> txred
    indy = <unsigned int> tyred
    indz = <unsigned int> tzred
    if indx>=nx-1: indx = nx-2
    if indy>=ny-1: indy = ny-2
    if indz>=nz-1: indz = nz-2
    for ii in range(4):
        txabs[ii] = fabs(txred - indx + 1 - ii)
        tyabs[ii] = fabs(tyred - indy + 1 - ii)
        tzabs[ii] = fabs(tzred - indz + 1 - ii)      
    fx[0] = (2.-txabs[0])**3
    fx[1] = (4.-6.*txabs[1]**2+3.*txabs[1]**3)
    fx[2] = (4.-6.*txabs[2]**2+3.*txabs[2]**3)
    fx[3] = (2.-txabs[3])**3    
    
    fy[0] = (2.-tyabs[0])**3
    fy[1] = (4.-6.*tyabs[1]**2+3.*tyabs[1]**3)
    fy[2] = (4.-6.*tyabs[2]**2+3.*tyabs[2]**3)
    fy[3] = (2.-tyabs[3])**3
    
    fz[0] = (2.-tzabs[0])**3
    fz[1] = (4.-6.*tzabs[1]**2+3.*tzabs[1]**3)
    fz[2] = (4.-6.*tzabs[2]**2+3.*tzabs[2]**3)
    fz[3] = (2.-tzabs[3])**3
    
    res = 0.
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                res += k[indx+ii,indy+jj,indz+kk]*fx[ii]*fy[jj]*fz[kk]
    return res