#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import  numpy
cimport numpy
from constants cimport *
cimport cython
import matplotlib.pyplot as mpl
from matplotlib import rc


cdef extern from "math.h":
    double exp(double x)
    double sqrt(double x)
    double atan2(double y, double x)
    double fabs(double x)
    double sin(double x)
    double cos(double x)

cdef class FieldMap:
    
    cdef:
        object intObj
        double brho, chargeOverMass, v0Abs, dt, angIn
        unsigned int nSteps, nStepsPerCell
        double[:,:] particleRefData
        double[:,:] particleEnv1Data
        double[:,:] particleEnv2Data
        double[:,:] multipoleData
        double[:] x0Ref
        double[:] v0Ref
        
    def __init__(self, object intObj):
        self.set(intObj)
    
    cpdef set(self, object intObj):
        self.intObj = intObj     
            
    cpdef doRef(self, double[:] x0, double[:] v0, double[:] bScale, double angOut, 
                 unsigned int nStepsPerCell = 10, double chargeOverMass = elementary_charge/m_p):   
        cdef:
            double bScale1
            unsigned int ii 
            double[:] ddir = numpy.empty(2)
        
        self.v0Abs = sqrt(v0[0]*v0[0]+v0[1]*v0[1])
        self.brho = self.v0Abs/chargeOverMass/sqrt(1.-(self.v0Abs/c)**2)      
        self.x0Ref = x0
        self.v0Ref = v0
        self.angIn = atan2(v0[1], v0[0])
        self.chargeOverMass = chargeOverMass
        self.nStepsPerCell = nStepsPerCell 
        self.dt = min(self.intObj.getDx(),self.intObj.getDy())/self.v0Abs/self.nStepsPerCell
        
        bScale1 = _findBScale(x0, v0, bScale, angOut, self.dt, chargeOverMass, self.intObj)
        self.intObj.setBScalePerma(bScale1)             
        
        self.nSteps = _findNSteps(x0, v0, self.intObj, self.dt, self.chargeOverMass)
        
        self.particleRefData = numpy.empty((self.nSteps,4))
        self.particleRefData[0,0] = x0[0]
        self.particleRefData[0,1] = x0[1]
        self.particleRefData[0,2] = v0[0]
        self.particleRefData[0,3] = v0[1]
        _trackAndSave(self.particleRefData, self.intObj, self.dt, self.chargeOverMass, self.nSteps)
        self.multipoleData = numpy.empty((self.nSteps,6))
        for ii in range(self.nSteps):
            ddir[0] = self.particleRefData[ii,3]/self.v0Abs
            ddir[1] = -self.particleRefData[ii,2]/self.v0Abs
            self.intObj.interpolateD(self.particleRefData[ii,:2], ddir, self.multipoleData[ii,:])
#             for jj in range(1,6):
#                 self.multipoleData[ii,jj] /= self.multipoleData[ii,0]
#             self.multipoleData[ii,2]*= 2. 
#             self.multipoleData[ii,3]*= 6. 
#             self.multipoleData[ii,4]*= 24.
#             self.multipoleData[ii,5]*= 120. 
         
           
    cpdef doEnv(self, double dx, double da, double delta):   
        cdef:
            double gammar, chi, vFac1
        
        gammar = 1./sqrt(1.-(self.v0Abs/c)**2)
        chi = (delta+1.)*gammar*self.v0Abs
        vFac2 = chi/sqrt(1.+(chi/c)**2)/self.v0Abs
        chi = (-delta+1.)*gammar*self.v0Abs
        vFac1 = chi/sqrt(1.+(chi/c)**2)/self.v0Abs

        self.particleEnv1Data = numpy.empty((self.nSteps,4))
        self.particleEnv1Data[0,0] = self.x0Ref[0]-dx*cos(self.angIn+0.5*pi)
        self.particleEnv1Data[0,1] = self.x0Ref[1]-dx*sin(self.angIn+0.5*pi)
        self.particleEnv1Data[0,2] = self.v0Ref[0]*cos(da) + self.v0Ref[1]*sin(da)
        self.particleEnv1Data[0,2]*= vFac1
        self.particleEnv1Data[0,3] = -self.v0Ref[0]*sin(da) + self.v0Ref[1]*cos(da)
        self.particleEnv1Data[0,3]*= vFac1
        _trackAndSave(self.particleEnv1Data, self.intObj, self.dt, self.chargeOverMass, self.nSteps)

        self.particleEnv2Data = numpy.empty((self.nSteps,4))
        self.particleEnv2Data[0,0] = self.x0Ref[0]+dx*cos(self.angIn+0.5*pi)
        self.particleEnv2Data[0,1] = self.x0Ref[1]+dx*sin(self.angIn+0.5*pi)
        self.particleEnv2Data[0,2] = self.v0Ref[0]*cos(-da) + self.v0Ref[1]*sin(-da)
        self.particleEnv2Data[0,2]*= vFac2
        self.particleEnv2Data[0,3] = -self.v0Ref[0]*sin(-da) + self.v0Ref[1]*cos(-da)
        self.particleEnv2Data[0,3]*= vFac2
        _trackAndSave(self.particleEnv2Data, self.intObj, self.dt, self.chargeOverMass, self.nSteps)
    
    cpdef plotMultipoles(self, subPlotObj = None, scale = None, xlim = None):
        rc('text', usetex=True)
        rc('font', family='Helvetica')
        rc('xtick', labelsize=15) 
        rc('ytick', labelsize=15) 
        if scale is None:
            scale = [1., 1., 1., 1., 1., 1.]
        s = numpy.linspace(0,self.nSteps*self.v0Abs*self.dt,self.nSteps)
        if subPlotObj is None:
            figObj = mpl.figure()
            subPlotObj = figObj.add_subplot(111) 
        
        subPlotObj.plot(s, scale[0]*numpy.asarray(self.multipoleData[:,0]),'-', label='dipole ('+str(numpy.int(scale[0]))+'$\cdot$b$_1$)')
        subPlotObj.plot(s, scale[1]*numpy.asarray(self.multipoleData[:,1]),'--', label='quadrupole ('+str(numpy.int(scale[1]))+'$\cdot$b$_2$)')
        subPlotObj.plot(s, scale[2]*numpy.asarray(self.multipoleData[:,2]),'-.', label='sextupole ('+str(numpy.int(scale[2]))+'$\cdot$b$_3$)')
        subPlotObj.plot(s, scale[3]*numpy.asarray(self.multipoleData[:,3]),':', label='octupole ('+str(numpy.int(scale[3]))+'$\cdot$b$_4$)')
        subPlotObj.plot(s, scale[4]*numpy.asarray(self.multipoleData[:,4]),'-', label='decapole ('+str(numpy.int(scale[4]))+'$\cdot$b$_5$)')
        subPlotObj.plot(s, scale[5]*numpy.asarray(self.multipoleData[:,5]),'--', label='dodecapole ('+str(numpy.int(scale[5]))+'$\cdot$b$_6$)')
            
        subPlotObj.set_xlabel(r'$s\;\mathrm{in}\;\mathrm{m}$', fontsize=25)
        subPlotObj.set_ylabel(r'$b_n\;\mathrm{in}\;\mathrm{T}\mathrm{m}^{-n}$', fontsize=25)
        subPlotObj.legend(fancybox=True)
        if xlim is not None:
            subPlotObj.set_xlim(xlim)
        return subPlotObj
    
    cpdef plotRef(self, subPlotObj = None):
        rc('text', usetex=True)
        rc('font', family='Helvetica')
        rc('xtick', labelsize=15) 
        rc('ytick', labelsize=15) 
        if subPlotObj is None:
            figObj = mpl.figure()
            subPlotObj = figObj.add_subplot(111) 
        plot = subPlotObj.plot(self.particleRefData[:,1],self.particleRefData[:,0],'w')
        plot = subPlotObj.plot(self.particleRefData[:,1],self.particleRefData[:,0],'k--', label='reference particle')
        subPlotObj.set_xlabel(r'$z\;\mathrm{in}\;\mathrm{m}$', fontsize=25)
        subPlotObj.set_ylabel(r'$x\;\mathrm{in}\;\mathrm{m}$', fontsize=25)
        subPlotObj.legend(fancybox=True)
        return subPlotObj


    cpdef plotEnv(self, subPlotObj = None):
        rc('text', usetex=True)
        rc('font', family='Helvetica')
        rc('xtick', labelsize=15) 
        rc('ytick', labelsize=15) 
        if subPlotObj is None:
            figObj = mpl.figure()
            subPlotObj = figObj.add_subplot(111) 
        subPlotObj.plot(self.particleEnv1Data[:,1],self.particleEnv1Data[:,0],'k:')
        subPlotObj.plot(self.particleEnv2Data[:,1],self.particleEnv2Data[:,0],'k:', label='beam envelope')
        subPlotObj.set_xlabel(r'$z\;\mathrm{in}\;\mathrm{m}$', fontsize=25)
        subPlotObj.set_ylabel(r'$x\;\mathrm{in}\;\mathrm{m}$', fontsize=25)
        subPlotObj.legend(fancybox=True)
        return subPlotObj

cdef double _findBScale(double[:] x0, double[:] v0, double[:] bScale, double angOut, 
                        double dt, double chargeOverMass, object intObj):    
    cdef:
        double[2] x
        double[2] v
        double[:] vTemp
        double v0Abs, bScale0, bScale1, bScale2, angErr1, angErr0, angErr2       
    x[1] = x0[1]
    x[0] = x0[0]
    v[0] = v0[0]
    v[1] = v0[1]
    v0Abs = sqrt(v0[0]*v0[0]+v0[1]*v0[1])
    
    bScale2 = bScale[1]
    intObj.setBScale(bScale2)
    vTemp = _trackFinDir(x, v, intObj, dt, chargeOverMass)
    angErr2 = atan2(vTemp[0],vTemp[1])-angOut
    
    bScale0 = bScale[0]
    intObj.setBScale(bScale0)
    vTemp = _trackFinDir(x, v, intObj, dt, chargeOverMass)
    angErr0 = atan2(vTemp[0],vTemp[1])-angOut

    for ii in range(100):
        bScale1 = (bScale2+bScale0)*0.5
        intObj.setBScale(bScale1)
        vTemp = _trackFinDir(x, v, intObj, dt, chargeOverMass)
        angErr1 = atan2(vTemp[0],vTemp[1])-angOut
        if angErr1*angErr2 > 0.:
            bScale2 = bScale1
        else:
            bScale0 = bScale1
#         print bScale1, angErr1, angErr2
        if fabs(angErr1)<1.e-9:
            break
    if ii>=99:
        raise ValueError('WARNING: Angle search has not converged. Try different bScale interval.')
    bScale1 = (bScale2+bScale0)*0.5
    return bScale1


cdef void _trackAndSave(double[:,:] particleData, object intObj, double dt, double chargeOverMass, unsigned int nSteps):
    cdef:
        double by, ts1, vs0, vs1, vAbsi
        double chargeDtOverMassGammaHalf = 0.5*chargeOverMass*dt*sqrt(1.-(particleData[0,2]*particleData[0,2]+particleData[0,3]*particleData[0,3])/c**2)
    
    for ii in range(nSteps-1):
        by = intObj.interpolate(particleData[ii,0],particleData[ii,1])
        ts1 = chargeDtOverMassGammaHalf*by
        vs0 = particleData[ii,2] - particleData[ii,3]*ts1
        vs2 = particleData[ii,3] + particleData[ii,2]*ts1
        ts1 *= 2./(1. + ts1*ts1)  
        particleData[(ii+1),2] = particleData[ii,2] - vs2*ts1
        particleData[(ii+1),3] = particleData[ii,3] + vs0*ts1 
        particleData[(ii+1),0] = particleData[ii,0] + dt*particleData[(ii+1),2]
        particleData[(ii+1),1] = particleData[ii,1] + dt*particleData[(ii+1),3]       
    return 


cdef double[::1] _trackFinDir(double[:] x0, double[:] v0, object intObj, double dt, double chargeOverMass):
    cdef:
        double by, ts1, vs0, vs1, vAbsi
        double[2] xOld
        double[2] xNew
        double[2] vOld
        double[::1] vNew = numpy.empty(2)
        double[::1] xMinMax = intObj.getXMinMax()
        double[::1] zMinMax = intObj.getYMinMax()
        double chargeDtOverMassGammaHalf = 0.5*chargeOverMass*dt*sqrt(1-(v0[0]*v0[0]+v0[1]*v0[1])/c**2)
    
    vNew[0] = v0[0]
    vNew[1] = v0[1]  
    xNew[0] = x0[0]
    xNew[1] = x0[1]
#     mpl.figure()
    while not (xNew[0]<xMinMax[0] or xNew[0]>xMinMax[1] or xNew[1]>zMinMax[1]):# or xNew[1]<zExt[0] 
        xOld[0] = xNew[0] 
        xOld[1] = xNew[1]
        vOld[0] = vNew[0] 
        vOld[1] = vNew[1]
        by = intObj.interpolate(xOld[0],xOld[1])
        ts1 = chargeDtOverMassGammaHalf*by
        vs0 = vOld[0] - vOld[1]*ts1
        vs2 = vOld[1] + vOld[0]*ts1
        ts1 *= 2./(1. + ts1*ts1)  
        vNew[0] = vOld[0] - vs2*ts1
        vNew[1] = vOld[1] + vs0*ts1 
        xNew[0] = xOld[0] + dt*vNew[0]
        xNew[1] = xOld[1] + dt*vNew[1]     
#         mpl.plot(xNew[0],xNew[1],'.')
#     mpl.show()   
    return vNew


cdef unsigned int _findNSteps(double[:] x0, double[:] v0, object intObj, double dt, double chargeOverMass):
    cdef:
        double by, ts1, vs0, vs1, vAbsi
        double[2] xOld
        double[2] xNew
        double[2] vOld
        double[::1] vNew = numpy.empty(2)
        double[::1] xMinMax = intObj.getXMinMax()
        double[::1] zMinMax = intObj.getYMinMax()
        double chargeDtOverMassGammaHalf = 0.5*chargeOverMass*dt*sqrt(1-(v0[0]*v0[0]+v0[1]*v0[1])/c**2)
        unsigned int ii = 0
    
    vNew[0] = v0[0]
    vNew[1] = v0[1]  
    xNew[0] = x0[0]
    xNew[1] = x0[1]
    while not (xNew[0]<xMinMax[0] or xNew[0]>xMinMax[1] or xNew[1]>zMinMax[1]):# or xNew[1]<zExt[0] 
        xOld[0] = xNew[0] 
        xOld[1] = xNew[1]
        vOld[0] = vNew[0] 
        vOld[1] = vNew[1]
        by = intObj.interpolate(xOld[0],xOld[1])
        ts1 = chargeDtOverMassGammaHalf*by
        vs0 = vOld[0] - vOld[1]*ts1
        vs2 = vOld[1] + vOld[0]*ts1
        ts1 *= 2./(1. + ts1*ts1)  
        vNew[0] = vOld[0] - vs2*ts1
        vNew[1] = vOld[1] + vs0*ts1 
        xNew[0] = xNew[0] + dt*vNew[0]
        xNew[1] = xNew[1] + dt*vNew[1]        
        ii += 1
    return ii

cdef class GaussianWavelet:

    cdef:
        double[::1] tempFx      # Preallocate temporary arrays for interpolation
        double[::1] tempFy
        double[::1] xRedTemp, yRedTemp
        double[::1] x, y, z
        double[::1] xExt, yExt, zExt
        double[::1] xMinMax, yMinMax
        double S, dx, dy, dxi, dyi, bScale
        unsigned int nx, ny, np, dn, nxExt, nyExt, npExt, dnExt
        
    def __init__(self, x, y, double[::1] z, double S = 1.4):
        self.make(x, y, z, S = S)

    cpdef make(self, x, y, double[::1] z, double S = 1.4):  
        cdef:
            unsigned int ii, jj         
            unsigned int dn3Half     
        self.bScale = 1.
        self.x = numpy.array(x[:2])
        self.y = numpy.array(y[:2])
        self.z = z
        self.nx = <unsigned int> x[2]
        self.ny = <unsigned int> y[2]
        self.np = self.z.shape[0]
        self.dn = <unsigned int> (S*16.+1.)
        self.dn += (self.dn % 2)
        self.dnExt = <unsigned int> (S*2.+0.5)
        self.nxExt = self.nx+2*self.dn+2*self.dnExt
        self.nyExt = self.ny+2*self.dn+2*self.dnExt
        self.npExt = self.nxExt*self.nyExt
        self.S = S      
        self.zExt = numpy.zeros(self.npExt, dtype=numpy.double)
        dn3Half = self.dn + self.dnExt
        for ii in range(self.nx):
            for jj in range(self.ny):
                self.zExt[(jj+dn3Half)*self.nxExt+ii+dn3Half] = self.z[jj*self.nx+ii]
        for ii in range(self.nx):
            for jj in range(self.dnExt):
                self.zExt[(jj+self.dn)*self.nxExt+ii+dn3Half] = (self.zExt[dn3Half*self.nxExt+ii+dn3Half] -
                                                                 (self.zExt[(dn3Half+1)*self.nxExt+ii+dn3Half]-self.zExt[dn3Half*self.nxExt+ii+dn3Half])*(dn3Half-(jj+self.dn))
                                                                 )
                self.zExt[(jj+self.nyExt-dn3Half)*self.nxExt+ii+dn3Half] = (self.zExt[(self.nyExt-dn3Half-1)*self.nxExt+ii+dn3Half] +
                                                                            (self.zExt[(self.nyExt-dn3Half-1)*self.nxExt+ii+dn3Half]-self.zExt[(self.nyExt-dn3Half-2)*self.nxExt+ii+dn3Half])*
                                                                            ((jj+self.nyExt-dn3Half)-(self.nyExt-dn3Half-1))
                                                                            )
        for ii in range(self.dnExt):
            for jj in range(self.nyExt):
                self.zExt[jj*self.nxExt+ii+self.dn] = (self.zExt[jj*self.nxExt+dn3Half] -
                                                       (self.zExt[jj*self.nxExt+dn3Half+1]-self.zExt[jj*self.nxExt+dn3Half])*(dn3Half-(ii+self.dn))
                                                       )
                self.zExt[jj*self.nxExt+ii+self.nxExt-dn3Half] = (self.zExt[jj*self.nxExt+self.nxExt-dn3Half-1] +
                                                                  (self.zExt[jj*self.nxExt+self.nxExt-dn3Half-1]-self.zExt[jj*self.nxExt+self.nxExt-dn3Half-2])*
                                                                  ((ii+self.nxExt-dn3Half)-(self.nxExt-dn3Half-1))
                                                                  )
        self.dx = (self.x[1]-self.x[0])/(self.nx-1)
        self.dy = (self.y[1]-self.y[0])/(self.ny-1)
        self.dxi = 1./self.dx
        self.dyi = 1./self.dy
        self.xExt = self.x.copy()
        self.xExt[0] -= (self.dnExt+self.dn)*self.dx
        self.xExt[1] += (self.dnExt+self.dn)*self.dx
        self.yExt = self.y.copy()
        self.yExt[0] -= (self.dnExt+self.dn)*self.dy
        self.yExt[1] += (self.dnExt+self.dn)*self.dy
        self.xMinMax = self.x.copy()
        self.xMinMax[0] -= (self.dnExt+0.5*self.dn)*self.dx
        self.xMinMax[1] += (self.dnExt+0.5*self.dn)*self.dx
        self.yMinMax = self.y.copy()
        self.yMinMax[0] -= (self.dnExt+0.5*self.dn)*self.dy
        self.yMinMax[1] += (self.dnExt+0.5*self.dn)*self.dy
        self.tempFx = numpy.empty(self.dn, dtype=numpy.double)
        self.tempFy = numpy.empty(self.dn, dtype=numpy.double)
        self.xRedTemp = numpy.empty(self.dn, dtype=numpy.double)
        self.yRedTemp = numpy.empty(self.dn, dtype=numpy.double)
        
    cpdef double interpolate(self, double xVal, double yVal):
        if (xVal<=self.xMinMax[0] or xVal>=self.xMinMax[1] or yVal<=self.yMinMax[0] or yVal>=self.yMinMax[1]):
            return 0.
        else:
            return self.bScale*_gwint(xVal, yVal, self.xExt, self.yExt, self.zExt, self.S, self.dxi, self.dyi, self.dn, self.nxExt, self.tempFx, self.tempFy)    

    cpdef double[:] interpolateD(self, double[:] pos, double[:] ddir, double[:] derivatives):
        _gwintdtill5(pos[0], pos[1], self.xExt, self.yExt, self.zExt, self.S,
                     self.dxi, self.dyi, self.dn, self.nxExt, self.tempFx, self.tempFy, 
                     self.xRedTemp, self.yRedTemp, derivatives, ddir)
        return derivatives
                
    cpdef numpy.ndarray getXExt(self):
        return numpy.asarray(self.xExt)
    
    cpdef numpy.ndarray getYExt(self):
        return numpy.asarray(self.yExt)
    
    cpdef numpy.ndarray getXMinMax(self):
        return numpy.asarray(self.xMinMax)
    
    cpdef numpy.ndarray getYMinMax(self):
        return numpy.asarray(self.yMinMax)
    
    cpdef numpy.ndarray getX(self):
        return numpy.asarray(self.x)
    
    cpdef numpy.ndarray getY(self):
        return numpy.asarray(self.y)
    
    cpdef numpy.ndarray getZExt(self):
        return numpy.asarray(self.zExt)
    
    cpdef unsigned int getNxExt(self):
        return self.nxExt
    
    cpdef unsigned int getNyExt(self):
        return self.nyExt
    
    cpdef setBScale(self, double bScale):
        self.bScale = bScale
    
    cpdef setBScalePerma(self, double bScale):
        cdef:
            unsigned int ii, jj
        for ii in range(self.nx):
            for jj in range(self.ny):
                self.z[jj*self.nx+ii] *= bScale              
        for ii in range(self.nxExt):
            for jj in range(self.nyExt):
                self.zExt[jj*self.nxExt+ii] *= bScale
        self.bScale = 1.
        
    cpdef double getDx(self):
        return self.dx
    
    cpdef double getDy(self):
        return self.dy
    
    cpdef plot(self, xIn = None, yIn = None, object subPlotObj = None, unsigned int nMulti = 5):
        cdef:
            unsigned int nx, ny, np
            numpy.ndarray x, y, vals, extend
        nx = nMulti*self.nx; ny = nMulti*self.ny; np = nx*ny;
        if xIn is None:
            x = numpy.linspace(self.x[0],self.x[1],nx)
        else:
            x = numpy.linspace(xIn[0],xIn[1],nx)
        if yIn is None:
            y = numpy.linspace(self.y[0],self.y[1],ny)
        else:
            y = numpy.linspace(yIn[0],yIn[1],ny)
        vals = numpy.empty(np)
        for ii in range(nx):
            for jj in range(ny):
                vals[jj*nx+ii] = self.interpolate(x[ii],y[jj])
        extent = numpy.array([y[0], y[ny-1], x[0], x[nx-1]])
        rc('text', usetex=True)
        rc('font', family='Helvetica')
        rc('xtick', labelsize=15) 
        rc('ytick', labelsize=15) 
        if subPlotObj is  None:
            figObj = mpl.figure()
            subPlotObj = figObj.add_subplot(111) 
        img = subPlotObj.imshow(numpy.reshape(vals,(ny,nx)).transpose(), extent=extent, origin='lower')
        subPlotObj.set_xlabel(r'$z\;\mathrm{in}\;\mathrm{m}$', fontsize=25)
        subPlotObj.set_ylabel(r'$x\;\mathrm{in}\;\mathrm{m}$', fontsize=25)
        cbar = mpl.colorbar(img, shrink=0.75)
        cbar.set_label(r'$B_y\;\mathrm{in}\;\mathrm{T}$', fontsize=25, rotation=90)
        subPlotObj.set_xlim(extent[:2])
        subPlotObj.set_ylim(extent[2:])
        return subPlotObj
        
# See Berz book.
cdef double _gwint(double xVal, double yVal, double[:] x, double[:] y, double[:] z, double S,
                   double dxi, double dyi, unsigned int dn, unsigned int nx, double[::1] fx, double[::1] fy): 
    cdef:
        unsigned int ii, jj
        double res = 0.
        double Ssqrt2inv = S*sqrt2inv
        double xValRed, yValRed
        unsigned int indx0, indy0
        
    indx0 = (<unsigned int> ((xVal-x[0])*dxi)) - dn/2 + 1
    indy0 = (<unsigned int> ((yVal-y[0])*dyi)) - dn/2 + 1
    xValRed = (xVal-x[0])*dxi - indx0
    yValRed = (yVal-y[0])*dyi - indy0
    for ii in range(dn):
        fx[ii] = _gnd((xValRed-ii)/Ssqrt2inv)/Ssqrt2inv
        fy[ii] = _gnd((yValRed-ii)/Ssqrt2inv)/Ssqrt2inv
    for ii in range(dn):
        for jj in range(dn):
            res += z[(indy0+jj)*nx+(indx0+ii)]*fx[ii]*fy[jj]
    return res
    
                          
cdef double _gnd(double x):
    return pi2invsqrt*exp(-0.5*x*x)


cdef _gwintdtill5(double xVal, double yVal, double[:] x, double[:] y, double[:] z, double S,
                  double dxi, double dyi, unsigned int dn, unsigned int nx, double[::1] fx, double[::1] fy, 
                  double[::1] xRedTemp, double[::1] yRedTemp, double[:] res, double[:] u):#In): 
    cdef:
        unsigned int ii, jj
        double xValRed, yValRed, temp
        unsigned int indx0, indy0
#         double[:] u = uIn.copy()
#         double uAbsi = 1./sqrt(u[0]*u[0]+u[1]*u[1])
#     
#     u[0] *= uAbsi
#     u[1] *= uAbsi
    sig = S*sqrt2inv 
    sigi = 1./sig; sigi2 = sigi*sigi; 
    sigi3 = sigi2*sigi; sigi4 = sigi3*sigi; sigi5 = sigi4*sigi;
    
    indx0 = (<unsigned int> ((xVal-x[0])*dxi)) - dn/2 + 1
    indy0 = (<unsigned int> ((yVal-y[0])*dyi)) - dn/2 + 1
    xValRed = (xVal-x[0])*dxi - indx0
    yValRed = (yVal-y[0])*dyi - indy0
    for ii in range(dn):
        xRedTemp[ii] = (xValRed-ii)*sigi
        yRedTemp[ii] = (yValRed-ii)*sigi
        fx[ii] = _gnd(xRedTemp[ii])*sigi
        fy[ii] = _gnd(yRedTemp[ii])*sigi
    for ii in range(6):
        res[ii] = 0.
    for ii in range(dn):
        for jj in range(dn):
            t = u[0]*xRedTemp[ii] + u[1]*yRedTemp[jj]
            temp = z[(indy0+jj)*nx+(indx0+ii)]*fx[ii]*fy[jj]
            res[0] += temp
            res[1] += -t*temp*sigi
            res[2] += (t**2-1.)*sigi2*temp
            res[3] += (-t**3+3.*t)*sigi3*temp
            res[4] += (t**4-6.*t**2+3.)*sigi4*temp  
            res[5] += (-t**5+10.*t**3-15.*t)*sigi5*temp
            
            
            