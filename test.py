import transportMap
import numpy
import scipy
import scipy.linalg
import timeit
import matplotlib.pyplot

nParticles = 1e+4;
dx = 0.016; da = .00938; dy = .0166; db = .00904; drho = 0.025;
detectResSigma = 0.#0.0002/2.3548


particlesIn = numpy.ascontiguousarray(numpy.vstack(( (scipy.rand(nParticles)*2-1)*dx,
                                                     (scipy.rand(nParticles)*2-1)*da,
                                                     (scipy.rand(nParticles)*2-1)*dy,
                                                     (scipy.rand(nParticles)*2-1)*db,
                                                     (scipy.rand(nParticles)*2-1)*drho,
                                                    )
                                                   ).transpose()
                                      )


tmo = transportMap.TransportMap("D:/Dropbox/AG Petri/HRS/workspace/tier1_10thb.txt", reduceToOrder = 8)
firstOrderRes = 1/tmo.getTransportMapValues()[4,0]*numpy.sqrt(detectResSigma**2 + detectResSigma**2*tmo.getTransportMapValues()[0,0]**2)
print 2.3548*firstOrderRes
particlesOut = tmo.traceParticles(particlesIn)

particlesRecon =  numpy.ascontiguousarray(
                                          numpy.vstack( (particlesIn[:,0]+detectResSigma*scipy.randn(nParticles),
                                          particlesOut[:,1],
                                          particlesIn[:,2],
                                          particlesOut[:,3],
                                          particlesOut[:,0]+detectResSigma*scipy.randn(nParticles))
                                                       ).transpose()
                                          )

tmo = transportMap.TransportMap("D:/Dropbox/AG Petri/HRS/workspace/tier1_10th.txt", reduceToOrder = 1, memoryOrder = 1)
tmo.partiallyInverse(None)
matplotlib.pyplot.hist((tmo.reconstruct(particlesRecon)[:,0]-particlesIn[:,4])*1e+4,100, histtype='step', normed = True)
tmo = transportMap.TransportMap("D:/Dropbox/AG Petri/HRS/workspace/tier1_10th.txt", reduceToOrder = 3, memoryOrder = 3)
tmo.partiallyInverse(None)
matplotlib.pyplot.hist((tmo.reconstruct(particlesRecon)[:,0]-particlesIn[:,4])*1e+4,100, histtype='step', normed = True)

# x = numpy.linspace(-3*firstOrderRes,3*firstOrderRes,100)
# matplotlib.pyplot.plot(x*1e+4,1/1e+4/numpy.sqrt(2*numpy.pi)/firstOrderRes*numpy.exp(-0.5*x**2/firstOrderRes**2),'r-')

matplotlib.pyplot.show(block=True)
# print tmo.getPartialInverseValues().dot(tmo.getTransportMapValues())

# print timeit.timeit('transportMap.TransportMap("D:/Dropbox/AG Petri/HRS/workspace/tier1_10th.txt", reduceToOrder = 1)','from __main__ import transportMap', number=1)/1*1000

# print timeit.timeit('tmo.partiallyInverse(None)','from __main__ import tmo', number=1)/1*1000
# tmo = transportMap.TransportMap("D:/Dropbox/AG Petri/HRS/workspace/tier1_10th.txt", reduceToOrder = 2)
# print timeit.timeit('tmo.partiallyInverse(None)','from __main__ import tmo', number=1)/1*1000


# transportMapObj = transportMap.TransportMap("D:/Dropbox/AG Petri/HRS/workspace/tier1_10th.txt")
# particlesIn = scipy.rand(nParticles,6)
# print timeit.timeit('transportMapObj.traceParticles(particlesIn)','from __main__ import transportMapObj; from __main__ import particlesIn', number=1)/1*1000


        
# order = 10
# elements = 5
# combinations = 0
# array = numpy.zeros(order,dtype=numpy.uint32)
# fullArray = numpy.zeros((1024,elements),dtype=numpy.uint32)
# fullArrayIdx = 0
# index = 0
# 
# 
# transportMap.testRComb(fullArray)
# print fullArray
    
    
# "D:/Dropbox/AG Petri/HRS/workspace/tier1_10th.txt"