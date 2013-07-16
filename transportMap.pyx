#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import scipy
import scipy.linalg
import scipy.special
import numpy
cimport numpy
cimport cython

cdef extern from "math.h":
    double pow(double x, double y)


cdef class TransportMap:

    cdef:
        numpy.ndarray transportMapValues, exp, expCounts, partialInverseValues, changeCoords
        unsigned int order, rowsCount
    '''
    Constructor.
    
    Input:
           - x:                      Comment.

                                       
    '''   
    def __init__(self, filename, reduceToOrder = None, memoryOrder = None):
        
        cdef:
            unsigned int transportMapRowsCount
            numpy.ndarray sumTransportMapExp
            numpy.ndarray transportMapExpIn = numpy.genfromtxt(filename, delimiter = (72,1,1,1,1,1,1), usecols = range(1,7), dtype = numpy.uint32, skip_footer = 1)
            numpy.ndarray transportMapValuesIn = numpy.genfromtxt(filename, delimiter = (1,14,14,14,14,14,14), usecols = range(1,6), skip_footer = 1)
            

        
        sumTransportMapExp = numpy.sum(transportMapExpIn, axis=1)
        if reduceToOrder != None:
            transportMapValuesIn = transportMapValuesIn[sumTransportMapExp <= reduceToOrder,:]
            transportMapExpIn = transportMapExpIn[sumTransportMapExp <= reduceToOrder,:]
            
        cdef:
            numpy.ndarray[numpy.double_t, ndim=2] transportMapValuesTempBuff = transportMapValuesIn     
            numpy.ndarray[numpy.uint32_t, ndim=2] transportMapExpTempBuff = transportMapExpIn   
        
        self.order = numpy.amax(sumTransportMapExp)

        transportMapRowsCount = transportMapExpTempBuff.shape[0]   
        if self.order < reduceToOrder:
            print 'Remark: Given order larger than map order.'
        
        if memoryOrder != None:
            self.order = memoryOrder
        else:
            self.order = min( (self.order, reduceToOrder) )
         
        transportMapRowsCount -= numpy.sum(transportMapExpTempBuff[:,4]>0)
        transportMapValuesTempBuff = transportMapValuesTempBuff[transportMapExpTempBuff[:,4]<1,:]
        transportMapExpTempBuff = transportMapExpTempBuff[transportMapExpTempBuff[:,4]<1,:]
        transportMapExpTempBuff = numpy.delete(transportMapExpTempBuff, 4, axis=1)
        
        self.expCounts = numpy.uint32(numpy.round(scipy.special.binom(range(5,5+self.order),range(1,self.order+1))))    
        self.rowsCount = <unsigned int> numpy.sum(self.expCounts)
        self.transportMapValues = numpy.zeros((self.rowsCount, 5), dtype = numpy.double)
        self.exp = numpy.zeros((self.rowsCount, 5), dtype = numpy.uint32)  
        
        exponentsToOrder(self.exp, self.order)
        
        
        cdef:
            unsigned int *transportMapExpTemp = <unsigned int *> transportMapExpTempBuff.data
            numpy.ndarray[numpy.uint32_t, ndim=2] expBuff = self.exp
            unsigned int *exp = <unsigned int *> expBuff.data
            double *transportMapValuesTemp = <double *> transportMapValuesTempBuff.data
            numpy.ndarray[numpy.double_t, ndim=2] transportMapValuesBuff = self.transportMapValues
            double *transportMapValues = <double *> transportMapValuesBuff.data
            int idx
        
        for ii in range(transportMapRowsCount):
            idx = -1
            for kk in range(self.rowsCount):
                for jj in range(5):
                    if transportMapExpTemp[5*ii+jj]!=exp[5*kk+jj]:
                        break
                    if jj == 4:
                        idx = kk
                if idx>-1:
                    break

            for jj in range(5):
                transportMapValues[5*idx+jj] = transportMapValuesTemp[5*ii+jj]
        
        
    '''
    Discription
    
    '''    

        
    def traceParticles(self, numpy.ndarray[numpy.double_t, ndim=2] particlesInNumpy):
        
        cdef:
            double *particlesIn = <double *> particlesInNumpy.data
            numpy.ndarray[numpy.double_t, ndim=2] transportMapValuesNumpy = self.transportMapValues
            double *transportMapValues = <double *> transportMapValuesNumpy.data
            numpy.ndarray[numpy.uint32_t, ndim=2] expNumpy = self.exp
            unsigned int *exp = <unsigned int *> expNumpy.data
            
            double temp
            unsigned int ii, jj
            unsigned int particleCount = particlesInNumpy.shape[0], rowsCount = self.rowsCount
            
            numpy.ndarray[numpy.double_t, ndim=2] particlesOutNumpy = numpy.zeros((particleCount,5), dtype = numpy.double)
            double *particlesOut = <double *> particlesOutNumpy.data
            
        for ii in range(particleCount):
            for jj in range(rowsCount):
                temp = 1.
                for kk in range(5):
                    if exp[5*jj+kk] != 0:
                        temp *= pow(particlesIn[5*ii+kk],exp[5*jj+kk])
                for kk in range(5):
                    particlesOut[5*ii+kk] += temp*transportMapValues[5*jj+kk]
            
        
        return particlesOutNumpy

    def partiallyInverse(self, numpy.ndarray keepCoordsASD):

        self.changeCoords = numpy.zeros((3,2), numpy.uint32)
        self.partialInverseValues = self.transportMapValues.copy()
        
        cdef:
            numpy.ndarray[numpy.uint32_t, ndim=2] expBuff = self.exp
            unsigned int *exp = <unsigned int *> expBuff.data
            numpy.ndarray[numpy.uint32_t] expCountsBuff = self.expCounts
            unsigned int *expCounts = <unsigned int *> expCountsBuff.data
            numpy.ndarray[numpy.double_t, ndim=2] partialInverseValuesNumpy = self.partialInverseValues
            double *partialInverseValues = <double *> partialInverseValuesNumpy.data
            unsigned int order = self.order, ii, jj, kk
            unsigned int rowsCount = self.rowsCount
            numpy.ndarray[numpy.double_t, ndim=2] equationSystemNumpy = numpy.zeros((rowsCount,rowsCount), dtype=numpy.double)
            double *equationSystem = <double *> equationSystemNumpy.data
            numpy.ndarray firstOrder = self.transportMapValues[:5,:].transpose()
            numpy.ndarray[numpy.uint32_t, ndim=2] changeCoordsNumpy = self.changeCoords
            unsigned int *changeCoords = <unsigned int *> changeCoordsNumpy.data
            numpy.ndarray[numpy.double_t, ndim=2] temporaryMapValuesNumpy = numpy.zeros((rowsCount,5), dtype=numpy.double)
            double *temporaryMapValues = <double *> temporaryMapValuesNumpy.data
            numpy.ndarray[numpy.double_t] temporarySolutionVecNumpy = numpy.zeros(rowsCount)
            double *temporarySolutionVec = <double *> temporarySolutionVecNumpy.data
            numpy.ndarray[numpy.double_t] temporaryRHSNumpy = numpy.zeros(rowsCount)
            double *temporaryRHS = <double *> temporaryRHSNumpy.data
            unsigned int *startExp = [0,0,0,0,0]
            double *startValues = [0,0,0,0,0]


        ############## Write automatic pivot stuff later ##############
        for ii in range(5):
            for jj in range(5):
                pass
        changeCoords[0] = 4; changeCoords[2] = 1; changeCoords[4] = 3;
        changeCoords[1] = 0; changeCoords[3] = 1; changeCoords[5] = 3;
        ###############################################################
             
        for ii in range(3):
            for jj in range(rowsCount):
                 
                if exp[5*jj+changeCoords[2*ii]] == 0:
                    equationSystem[jj*rowsCount+jj] = 1.
                 
                else:
                    for kk in range(5):
                        if kk != changeCoords[2*ii]:
                            startExp[kk] = exp[5*jj+kk]
                        else:
                            startExp[kk] = 0
                    expandToOrder(equationSystem, partialInverseValues, &changeCoords[2*ii], startExp, 
                                  jj, 1., exp, rowsCount, exp[5*jj+changeCoords[2*ii]], order)
    
            temporaryRHSNumpy[changeCoords[2*ii]] = 1. 
            temporarySolutionVecNumpy = scipy.linalg.solve(equationSystemNumpy, temporaryRHSNumpy)
            temporaryRHSNumpy[changeCoords[2*ii]] = 0.
            # Correct until here for first order.
            
            
            for jj in range(rowsCount):
                temporaryMapValues[5*jj+changeCoords[2*ii+1]] = temporarySolutionVecNumpy[jj]
            
            for jj in range(rowsCount):
                if exp[5*jj+changeCoords[2*ii]] == 0:
                    for kk in range(5):
                        if kk!=changeCoords[2*ii+1]:
                            temporaryMapValues[5*jj+kk] += partialInverseValues[5*jj+kk]
                else:     
                    for kk in range(5):
                        if kk != changeCoords[2*ii]:
                            startExp[kk] = exp[5*jj+kk]                    
                        else:
                            startExp[kk] = 0
                        if kk != changeCoords[2*ii+1]:
                            startValues[kk] = partialInverseValues[5*jj+kk]
                        else:
                            startValues[kk] = 0
                    expandToOrderRevamp(temporaryMapValues, &changeCoords[2*ii], startExp, 
                                        startValues, exp, rowsCount, exp[5*jj+changeCoords[2*ii]], order)

            # Reset/set arrays for next iteration. 
            for jj in range(5*rowsCount):
                partialInverseValues[jj] = temporaryMapValues[jj]          
                temporaryMapValues[jj] = 0.
            for jj in range(rowsCount**2):
                equationSystem[jj] = 0.   
            
        
    def reconstruct(self, numpy.ndarray[numpy.double_t, ndim=2] particlesInNumpy):
        cdef:
            double *particlesIn = <double *> particlesInNumpy.data
            numpy.ndarray[numpy.double_t, ndim=2] partialInverseValuesNumpy = self.partialInverseValues
            double *partialInverseValues = <double *> partialInverseValuesNumpy.data
            numpy.ndarray[numpy.uint32_t, ndim=2] expNumpy = self.exp
            unsigned int *exp = <unsigned int *> expNumpy.data
            
            double temp
            unsigned int ii, jj
            unsigned int particleCount = particlesInNumpy.shape[0], rowsCount = self.rowsCount
            
            numpy.ndarray[numpy.double_t, ndim=2] particlesOutNumpy = numpy.zeros((particleCount,5), dtype = numpy.double)
            double *particlesOut = <double *> particlesOutNumpy.data
            
            
        for ii in range(particleCount):
            for jj in range(rowsCount):
                temp = 1
                for kk in range(5):
                    if exp[5*jj+kk] != 0:
                        temp *= pow(particlesIn[5*ii+kk],exp[5*jj+kk])
                for kk in range(5):
                    particlesOut[5*ii+kk] += temp*partialInverseValues[5*jj+kk]
            
        
        return particlesOutNumpy


           
    def getTransportMapValues(self):
        return self.transportMapValues
    
    def getPartialInverseValues(self):
        return self.partialInverseValues
    
    def getExp(self):
        return self.exp
    
    def getChangeCoords(self):
        return self.changeCoords

        
cdef expandToOrder(double *equationSystem, double *transportMap, unsigned int *changeCoords, unsigned int *array, unsigned int equationSystemColumn,
                   double value, unsigned int *exp, unsigned int rowsCount, unsigned int remainingExp, unsigned int order):
    cdef:
        unsigned int sum = 0       
        unsigned int *passOnArray = [0,0,0,0,0]
        double passOnValue
        int idx
    
    if remainingExp > 0:
        for ii in range(rowsCount):
            sum = 0
            for jj in range(5):
                passOnArray[jj] = array[jj] + exp[5*ii+jj]  
                sum += passOnArray[jj]
            passOnValue = value*transportMap[5*ii+changeCoords[1]]
            if sum>order or passOnValue==0.:
                continue        
            expandToOrder(equationSystem, transportMap, changeCoords, passOnArray, equationSystemColumn,
                          passOnValue, exp, rowsCount, remainingExp-1, order)
        
    elif remainingExp == 0:

        for ii in range(5):
            sum += array[ii]
        if sum <= order:
            idx = -1
            for jj in range(rowsCount):
                for kk in range(5):
                    if array[kk]!=exp[5*jj+kk]:
                        break
                    if kk == 4:
                        idx = jj
                if idx>-1:
                    break
            equationSystem[rowsCount*idx+equationSystemColumn] += value


                    
cdef expandToOrderRevamp(double *partialInverse, unsigned int *changeCoords, unsigned int *array,
                         double *values, unsigned int *exp, unsigned int rowsCount, unsigned int remainingExp, unsigned int order):
    cdef:
        unsigned int sum = 0       
        unsigned int *passOnArray = [0,0,0,0,0]
        double *passOnValues = [0,0,0,0,0]
        int idx
         
    if remainingExp > 0:
        for ii in range(rowsCount):
            sum = 0
            for jj in range(5):
                passOnArray[jj] = array[jj] + exp[5*ii+jj]  
                sum += passOnArray[jj]
            if sum>order:
                continue
            for jj in range(5):
                passOnValues[jj] = values[jj]*partialInverse[5*ii+changeCoords[1]] 
            if passOnValues[0]==0. and passOnValues[1]==0. and passOnValues[2]==0. and passOnValues[3]==0. and passOnValues[4]==0.:
                continue
            expandToOrderRevamp(partialInverse, changeCoords, passOnArray,
                                passOnValues, exp, rowsCount, remainingExp-1, order)
         
    elif remainingExp == 0:
 
        for ii in range(5):
            sum += array[ii]
        if sum <= order:
            idx = -1
            for jj in range(rowsCount):
                for kk in range(5):
                    if array[kk]!=exp[5*jj+kk]:
                        break
                    if kk == 4:
                        idx = jj
                if idx>-1:
                    break
            for ii in range(5):
                if ii!=changeCoords[1]:
                    partialInverse[5*idx+ii] += values[ii]

   
def exponentsToOrder(fullArrayIn, orderIn):
    cdef:
        unsigned int elements = 5, order = <unsigned int> orderIn, fullArrayIdxReal = 0 
        numpy.ndarray[numpy.uint32_t, ndim=2] fullArrayBuff = fullArrayIn
        unsigned int *fullArray = <unsigned int *> fullArrayBuff.data
        numpy.ndarray[numpy.uint32_t] arrayBuff = numpy.zeros(order, dtype=numpy.uint32)
        unsigned int *array = <unsigned int *> arrayBuff.data        
        unsigned int *fullArrayIdx = &fullArrayIdxReal

    for ii in range(1, 1+order):
        rComb(array, ii, elements, ii, fullArray, fullArrayIdx)

cdef rComb(unsigned int *array, unsigned int arrayLength, unsigned int elements, unsigned int order, unsigned int *fullArray, unsigned int *fullArrayIdx):
    cdef:
        unsigned int ii
    if order > 0:
        for ii in range(elements):
            array[order-1] = ii         
            rComb(array, arrayLength, <unsigned int> (array[order-1]+1), <unsigned int> (order-1), fullArray, fullArrayIdx)
    if order == 0:
        for ii in range(arrayLength):
            fullArray[5*fullArrayIdx[0] + array[ii]] += 1
        fullArrayIdx[0] = fullArrayIdx[0]+1
        
        
