#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import os
import numpy
cimport numpy
cimport cython

cdef extern from "math.h":
    double pow(double x, double y)
    double fabs(double x)


cdef class TransferMap:

    cdef:
        numpy.ndarray transferMapValues, transferMapExp, partialInverseValues
        unsigned int transferMapOrder, transferMapRowsCount, nCoordsRHS, nCoordsLHS
    '''
    Constructor.
    
    Input:
           - x:                      Comment.

                                       
    '''   
    def __init__(self, filename, unsigned int reduceToOrder = 0, unsigned int memoryOrder = 0):
        
        self.importCosyInf(filename, reduceToOrder = reduceToOrder, memoryOrder = memoryOrder)

 
    cpdef importCosyInf(self, filename, unsigned int reduceToOrder = 0, unsigned int memoryOrder = 0):
       
        self.nCoordsRHS = 5
        self.nCoordsLHS = 5    
                
        cdef:
            unsigned int transferMapRowsCount, transferMapExpCount
            numpy.ndarray sumTransferMapExp
            numpy.ndarray transferMapExpIn = numpy.genfromtxt(filename, delimiter = (72,1,1,1,1,1,1), usecols = range(1,7), dtype = numpy.uint32, skip_footer = 1)
            numpy.ndarray transferMapValuesIn = numpy.genfromtxt(filename, delimiter = (1,14,14,14,14,14,14), usecols = range(1,6), skip_footer = 1)
               
        if numpy.sum(transferMapExpIn[0,:]) == 0:
            transferMapExpIn = transferMapExpIn[1:]
            transferMapValuesIn = transferMapValuesIn[1:]
            
        sumTransferMapExp = numpy.sum(transferMapExpIn, axis=1)
        if reduceToOrder>0:
            transferMapValuesIn = transferMapValuesIn[sumTransferMapExp <= reduceToOrder,:]
            transferMapExpIn = transferMapExpIn[sumTransferMapExp <= reduceToOrder,:]
            
        cdef:
            numpy.ndarray[numpy.double_t, ndim=2] transferMapValuesTempBuff = transferMapValuesIn     
            numpy.ndarray[numpy.uint32_t, ndim=2] transferMapExpTempBuff = transferMapExpIn   
        
        self.transferMapOrder = numpy.amax(sumTransferMapExp)

        transferMapRowsCount = transferMapExpTempBuff.shape[0]   
        if self.transferMapOrder < reduceToOrder:
            print 'Remark: Given order larger than map order.'
        
        if memoryOrder>0:
            self.transferMapOrder = memoryOrder
        else:
            self.transferMapOrder = min( (self.transferMapOrder, reduceToOrder) )
         
        transferMapRowsCount -= numpy.sum(transferMapExpTempBuff[:,4]>0)
        transferMapValuesTempBuff = transferMapValuesTempBuff[transferMapExpTempBuff[:,4]<1,:]
        transferMapExpTempBuff = transferMapExpTempBuff[transferMapExpTempBuff[:,4]<1,:]
        transferMapExpTempBuff = numpy.delete(transferMapExpTempBuff, 4, axis=1)
         
        self.transferMapRowsCount = _nExponentsCombinations(self.transferMapOrder, self.nCoordsRHS)
        self.transferMapValues = numpy.zeros((self.transferMapRowsCount, self.nCoordsLHS), dtype = numpy.double)
        self.transferMapExp = numpy.zeros((self.transferMapRowsCount, self.nCoordsRHS), dtype = numpy.uint32)  
        
        _expCombinations(self.transferMapExp, self.transferMapOrder, self.nCoordsRHS)
             
        cdef:
            unsigned int *transferMapExpTemp = <unsigned int *> transferMapExpTempBuff.data
            unsigned int[:,:] transferMapExp = self.transferMapExp
            double *transferMapValuesTemp = <double *> transferMapValuesTempBuff.data
            double[:,:] transferMapValues = self.transferMapValues
            int idx
        
        for ii in range(transferMapRowsCount):
            idx = -1
            for kk in range(self.transferMapRowsCount):
                for jj in range(5):
                    if transferMapExpTemp[5*ii+jj]!=transferMapExp[kk,jj]:
                        break
                    if jj == 4:
                        idx = kk
                if idx>-1:
                    break

            for jj in range(5):
                transferMapValues[idx,jj] = transferMapValuesTemp[5*ii+jj]
    
    # Export transfer map to numpy binary format.
    cpdef exportTransferMap(self, folderName):
        try:
            os.makedirs(folderName)
        except:
            pass
        finally:
            numpy.save(folderName + '/transferMapValues.npy', self.transferMapValues)
            numpy.save(folderName + '/transferMapExp.npy', self.transferMapExp)
    
    # Import transfer map to numpy binary format.    
    cpdef importTransferMap(self, folderName):
        self.transferMapValues = numpy.load(folderName + '/transferMapValues.npy')
        self.transferMapExp = numpy.load(folderName + '/transferMapExp.npy')
        self.transferMapRowsCount = len(self.transferMapExp)
        self.nCoordsRHS = len(self.transferMapExp[0])
        self.nCoordsLHS = len(self.transferMapValues[0])
        
    # Trace particles through transfer map.
    cpdef numpy.ndarray traceParticles(self, double[:,:] particlesIn):
        
        cdef:
            double[:,:] transferMapValues= self.transferMapValues
            unsigned int[:,:] transferMapExp = self.transferMapExp 
            double temp
            unsigned int ii, jj, kk
            unsigned int particleCount = len(particlesIn), transferMapRowsCount = self.transferMapRowsCount
            unsigned int nCoordsRHS = self.nCoordsRHS, nCoordsLHS = self.nCoordsLHS
            numpy.ndarray particlesOutNumpy = numpy.zeros((particleCount,nCoordsLHS), dtype = numpy.double)
            double[:,:] particlesOut = particlesOutNumpy
              
        for ii in range(particleCount):
            for jj in range(transferMapRowsCount):
                temp = 1.
                for kk in range(nCoordsRHS):
                    if transferMapExp[jj,kk] > 0:
                        temp*= particlesIn[ii,kk]**transferMapExp[jj,kk]
                for kk in range(nCoordsLHS):
                    particlesOut[ii,kk]+= temp*transferMapValues[jj,kk]
        return particlesOutNumpy

    # Exchange coordinates from RHS to LHS. Be careful choosing good pivot elements or
    # use provided method.
    cpdef exchangeCoords(self, indLHSIn, indRHSIn):
        cdef:            
            unsigned int ii, nExchanges = len(indLHSIn)
            unsigned int[:] indLHS = numpy.array(indLHSIn, dtype=numpy.uint32)
            unsigned int[:] indRHS = numpy.array(indRHSIn, dtype=numpy.uint32)    
        for ii in range(nExchanges):
            _exchangeCoord(indLHS[ii], indRHS[ii], self.nCoordsLHS, self.nCoordsRHS, self.transferMapRowsCount, 
                           self.transferMapOrder, self.transferMapExp, self.transferMapValues)
    
    # Find best pivot elements for provided coordinate exchanges.
    cpdef exchangeCoordsPivot(self, indLHSIn, indRHSIn):
        cdef:
            unsigned int ii, jj
            unsigned int[:] indLHS = numpy.array(indLHSIn, dtype=numpy.uint32)
            unsigned int[:] indRHS = numpy.array(indRHSIn, dtype=numpy.uint32)    
            double[:,:] firstOrderMap = self.transferMapValues[:self.nCoordsRHS, :self.nCoordsLHS].copy()
        for ii in range(self.nCoordsRHS):
            for jj in range(self.nCoordsLHS):
                firstOrderMap[ii,jj] = fabs(firstOrderMap[ii,jj])
        raise NotImplemented('Choose pivot manually for now.')
    
    # This can be used to remove some coordinates. Note that RHS can almost never (meaning at best in first order)
    # be removed without doing approximations. So RHS coordinates removing not recommended.
    cpdef removeCoords(self, indLHSIn, indRHSIn):
        cdef:
            unsigned int ii, jj, kk, transferMapRowsCountNew
            unsigned int[:] indLHS = numpy.array(indLHSIn, dtype=numpy.uint32)
            unsigned int[:] indRHS = numpy.array(indRHSIn, dtype=numpy.uint32)
            unsigned int[:,:] transferMapExp = self.transferMapExp
            double[:,:] transferMapValues = self.transferMapValues
        _insertionSort(indLHS)
        _insertionSort(indRHS)
        for ii in range(len(indLHS)):
            _removeCoordLHS(indLHS[ii]-ii, self.nCoordsLHS, transferMapValues)
            self.nCoordsLHS -= 1
            self.transferMapValues = self.transferMapValues[:,:self.nCoordsLHS]
        for ii in range(len(indRHS)):
            transferMapRowsCountNew = _nExponentsCombinations(self.transferMapOrder, self.nCoordsRHS-1)
            _removeCoordRHS(indRHS[ii]-ii, self.nCoordsRHS, transferMapRowsCountNew, transferMapExp, transferMapValues)
            self.nCoordsRHS -= 1
            self.transferMapRowsCount = transferMapRowsCountNew
            self.transferMapExp = self.transferMapExp[:self.transferMapRowsCount,:self.nCoordsRHS]
            self.transferMapValues = self.transferMapValues[:self.transferMapRowsCount,:]

    # Replace one coordinate with identity. Usually used to get the momentum delta
    # on both sides instead of path length difference l on the LHS.
    cpdef setCoordToIdentity(self, unsigned int indLHS, unsigned int indRHS):
        cdef:
            unsigned int ii, jj, kk, sum
            unsigned int[:,:] transferMapExp = self.transferMapExp
            double[:,:] transferMapValues = self.transferMapValues
            
        for ii in range(self.transferMapRowsCount):
            transferMapValues[ii,indLHS] = 0.
        transferMapValues[indRHS, indLHS] = 1.
    
    # Split a coordinate on the RHS which consists of two summands.
    # Usually used to split delta_0+delta'.
    cpdef splitCoord(self, unsigned int indRHS):  
        cdef:
            unsigned int ii, jj, kk, ll, nn, transferMapRowsCountOld
            unsigned int[:,:] transferMapExpOld
            double[:,:] transferMapValuesOld
            unsigned int[:,:] transferMapExp
            double[:,:] transferMapValues           
            int idx
        transferMapRowsCountOld = self.transferMapRowsCount
        transferMapExpOld = self.transferMapExp 
        transferMapValuesOld = self.transferMapValues
        
        self.nCoordsRHS += 1
        self.transferMapRowsCount = _nExponentsCombinations(self.transferMapOrder, self.nCoordsRHS)
        self.transferMapExp = numpy.zeros((self.transferMapRowsCount, self.nCoordsRHS), dtype = numpy.uint32)  
        self.transferMapValues = numpy.zeros((self.transferMapRowsCount, self.nCoordsLHS), dtype = numpy.double)
        _expCombinations(self.transferMapExp, self.transferMapOrder, self.nCoordsRHS)
        transferMapExp = self.transferMapExp
        transferMapValues = self.transferMapValues
        for ii in range(transferMapRowsCountOld):
            idx = -1
            for kk in range(self.transferMapRowsCount):
                for jj in range(self.nCoordsRHS-1):
                    if transferMapExpOld[ii,jj]!=transferMapExp[kk,jj]:
                        break
                    if jj==(self.nCoordsRHS-2) and transferMapExp[kk,self.nCoordsRHS-1]==0:
                        idx = kk
                if idx>-1:
                    break

            for jj in range(self.nCoordsLHS):
                transferMapValues[idx,jj] = transferMapValuesOld[ii,jj]     
                
        for ii in range(transferMapRowsCountOld):   
            if transferMapExpOld[ii,indRHS]==0:
                continue
            else:
                nn = transferMapExpOld[ii,indRHS]
                for kk in range(nn+1):
                    for jj in range(self.transferMapRowsCount):
                        idx = -1
                        for ll in range(self.nCoordsRHS):
                            if ll==indRHS:
                                if kk!=transferMapExp[jj,indRHS]:
                                    break
                            elif ll==(self.nCoordsRHS-1):
                                if transferMapExp[jj,ll]==(nn-kk):
                                    idx = jj
                            else:
                                if transferMapExpOld[ii,ll]!=transferMapExp[jj,ll]:
                                    break
                        if idx>-1:
                            break
                    temp = _binomCoeff(nn,kk)
                    for jj in range(self.nCoordsLHS):
                        transferMapValues[idx,jj] = temp*transferMapValuesOld[ii,jj]  
    
    # Multiplies whole map by a factor. Usually used to multiply 
    # by -1 to later subtract from another map.
    cpdef multiplyTransferMapValues(self, double factor):
        self.transferMapValues *= factor
    
    # Merge a given transfer Map to the existing one on the RHS.
    # Number of coordinates on the LHS have to be identical.
    cpdef mergeTransferMaps(self, TransferMap transferMap2):
        cdef:
            unsigned int[:,:] transferMapExp
            unsigned int[:,:] transferMapExpOld1 = self.transferMapExp
            unsigned int[:,:] transferMapExpOld2 = transferMap2.getTransferMapExp()
            double[:,:] transferMapValues
            double[:,:] transferMapValuesOld1 = self.transferMapValues
            double[:,:] transferMapValuesOld2 = transferMap2.getTransferMapValues()
            unsigned int nCoordsRHS1 = self.nCoordsRHS
            unsigned int nCoordsRHS2 = transferMap2.getNCoordsRHS()
            unsigned int transferMapRowsCount1 = self.transferMapRowsCount
            unsigned int transferMapRowsCount2 = transferMap2.getTransferMapRowsCount()
            unsigned int ii, jj, kk
            int idx       
        self.nCoordsRHS = nCoordsRHS1 + nCoordsRHS2
        self.transferMapOrder = max(self.transferMapOrder, transferMap2.getTransferMapOrder())  
        self.transferMapRowsCount = _nExponentsCombinations(self.transferMapOrder, self.nCoordsRHS)
        self.transferMapExp = numpy.zeros((self.transferMapRowsCount, self.nCoordsRHS), dtype = numpy.uint32)  
        self.transferMapValues = numpy.zeros((self.transferMapRowsCount, self.nCoordsLHS), dtype = numpy.double)
        _expCombinations(self.transferMapExp, self.transferMapOrder, self.nCoordsRHS)
        transferMapExp = self.transferMapExp
        transferMapValues = self.transferMapValues
        # Sort in values of transfer map 1.
        for ii in range(transferMapRowsCount1):
            idx = -1
            for kk in range(self.transferMapRowsCount):
                for jj in range(self.nCoordsRHS):
                    if jj<nCoordsRHS1: 
                        if transferMapExpOld1[ii,jj]!=transferMapExp[kk,jj]:
                            break
                    elif jj<(self.nCoordsRHS-1):
                        if transferMapExp[kk,jj]!=0:
                            break
                    else:
                        if transferMapExp[kk,jj]==0:
                            idx = kk
                if idx>-1:
                    break
            for jj in range(self.nCoordsLHS):
                transferMapValues[idx,jj] = transferMapValuesOld1[ii,jj]     
        # Sort in values of transfer map 2.
        for ii in range(transferMapRowsCount2):
            idx = -1
            for kk in range(self.transferMapRowsCount):
                for jj in range(self.nCoordsRHS):
                    if jj<nCoordsRHS1: 
                        if transferMapExp[kk,jj]!=0:
                            break
                    elif jj<(self.nCoordsRHS-1):
                        if transferMapExp[kk,jj]!=transferMapExpOld2[ii,jj-nCoordsRHS1]:
                            break
                    else:
                        if transferMapExp[kk,jj]==transferMapExpOld2[ii,jj-nCoordsRHS1]:
                            idx = kk
                if idx>-1:
                    break
            for jj in range(self.nCoordsLHS):
                transferMapValues[idx,jj] = transferMapValuesOld2[ii,jj]     
    
    # Merge two coordinates on the RHS if they are supposed to be the same.
    # Second given index will be removed.
    cpdef mergeCoords(self, unsigned int indRHS1, unsigned int indRHS2):
        cdef:
            unsigned int ii, jj, transferMapRowsCountNew
            unsigned int[:,:] transferMapExp = self.transferMapExp
            double[:,:] transferMapValues = self.transferMapValues
            int idx
        
        for ii in range(self.transferMapRowsCount):
            if transferMapExp[ii,indRHS2]==0:
                continue
            else:
                idx = -1
                for jj in range(self.transferMapRowsCount):
                    for kk in range(self.nCoordsRHS):
                        if kk==indRHS1: 
                            if transferMapExp[ii,indRHS2]!=transferMapExp[jj,indRHS1]:
                                break
                        elif kk!=indRHS2:
                            if transferMapExp[ii,kk]!=transferMapExp[jj,kk]:
                                break
                        if kk==(self.nCoordsRHS-1):
                            idx = jj
                    if idx>-1:
                        break
                for kk in range(self.nCoordsLHS):
                    transferMapValues[idx,kk] += transferMapValues[ii,kk]  
        transferMapRowsCountNew = _nExponentsCombinations(self.transferMapOrder, self.nCoordsRHS-1)
        _removeCoordRHS(indRHS2, self.nCoordsRHS, transferMapRowsCountNew, transferMapExp, transferMapValues)
        self.nCoordsRHS -= 1
        self.transferMapRowsCount = transferMapRowsCountNew
        self.transferMapExp = self.transferMapExp[:self.transferMapRowsCount,:self.nCoordsRHS]
        self.transferMapValues = self.transferMapValues[:self.transferMapRowsCount,:]

        
    # Print aberrations scaled by given phase space.
    cpdef printAberrations(self, phaseSpaceIn):
        cdef:
            double[:] phaseSpace = numpy.array(phaseSpaceIn, dtype=numpy.double)
            unsigned int[:,:] transferMapExp = self.transferMapExp
            double[:,:] transferMapValues = self.transferMapValues
            double[:,:] aberrations = transferMapValues.copy()
            unsigned int ii, jj       
        for ii in range(self.transferMapRowsCount):
            temp = 1.
            for jj in range(self.nCoordsRHS):
                if transferMapExp[ii,jj]>0:
                    temp *= phaseSpace[jj]**transferMapExp[ii,jj]
            for jj in range(self.nCoordsLHS):
                aberrations[ii,jj] *= temp
        numpy.set_printoptions(precision=3)
        for ii in range(self.transferMapRowsCount):
            print numpy.asarray(transferMapExp[ii]), numpy.asarray(aberrations[ii])                             
        numpy.set_printoptions(precision=8)
        
    cpdef printTransferMap(self):
        self.printAberrations(numpy.ones(self.nCoordsRHS, dtype=numpy.double))
        
    cpdef numpy.ndarray getTransferMapValues(self):
        return self.transferMapValues
    
    cpdef numpy.ndarray getTransferMapExp(self):
        return self.transferMapExp
    
    cpdef unsigned int getNCoordsRHS(self):
        return self.nCoordsRHS
    
    cpdef unsigned int getNCoordsLHS(self):
        return self.nCoordsLHS
    
    cpdef unsigned int getTransferMapRowsCount(self):
        return self.transferMapRowsCount
    
    cpdef unsigned int getTransferMapOrder(self):
        return self.transferMapOrder


'''
Helper Functions from here on. Usually to improve speed.
'''
    
# Helper function to remove a coordinate on the LHS.
cdef void _removeCoordLHS(unsigned int indLHS, unsigned int nCoordsLHS, double[:,:] transferMapValues):
    cdef unsigned int jj
    for jj in range(indLHS,nCoordsLHS-1):
        transferMapValues[:,jj] = transferMapValues[:,jj+1]

# Helper function to remove a coordinate on the RHS.       
cdef void _removeCoordRHS(unsigned int indRHS, unsigned int nCoordsRHS, unsigned int transferMapRowsCountNew,
                          unsigned int[:,:] transferMapExp, double[:,:] transferMapValues):
    cdef: 
        unsigned int ii, jj
    jj = 0
    for ii in range(transferMapRowsCountNew):
        while transferMapExp[jj,indRHS]>0:
            jj += 1
        transferMapValues[ii,:] = transferMapValues[jj,:]
        transferMapExp[ii,:] = transferMapExp[jj,:]
        jj += 1
    for ii in range(indRHS,nCoordsRHS-1):
        transferMapExp[:,ii] = transferMapExp[:,ii+1]

# Simple insertion sort helper function. 
cdef void _insertionSort(unsigned int[:] array):
    cdef unsigned int temp, ii, jj
    for ii in range(1,len(array)):
        temp = array[ii]
        jj = ii
        while jj>0 and array[jj-1]>temp:
            temp = array[jj] 
            array[jj] = array[jj-1]
            jj-= 1
        array[jj] = temp


cdef void _exchangeCoord(unsigned int indLHS, unsigned int indRHS, unsigned int nCoordsLHS, unsigned int nCoordsRHS, unsigned int transferMapRowsCount, 
                         unsigned int transferMapOrder, unsigned int[:,:] transferMapExp, double[:,:] transferMapValues):
    
    cdef:       
        unsigned int jj, kk
        double[:,:] equationSystem = numpy.zeros((transferMapRowsCount,transferMapRowsCount), dtype=numpy.double)
        double[:,:] tempMapValues = numpy.zeros((transferMapRowsCount,nCoordsLHS), dtype=numpy.double)
        double[:] tempSolutionVec
        double[:] tempRHS = numpy.zeros(transferMapRowsCount)
        unsigned int[:] startExp = numpy.zeros(nCoordsRHS, dtype=numpy.uint32)
        double[:] startValues = numpy.zeros(nCoordsLHS, dtype=numpy.double)
         
    for jj in range(transferMapRowsCount):           
        if transferMapExp[jj,indRHS] == 0:
            equationSystem[jj,jj] = 1.              
        else:
            for kk in range(nCoordsRHS):
                if kk != indRHS:
                    startExp[kk] = transferMapExp[jj,kk]
                else:
                    startExp[kk] = 0
            _fillEquationSystemRecursive(equationSystem, transferMapValues, indLHS, nCoordsRHS, startExp, 
                                         jj, 1., transferMapExp, transferMapRowsCount, transferMapExp[jj,indRHS], transferMapOrder)
    tempRHS[indRHS] = 1. 
    tempSolutionVec = numpy.linalg.solve(equationSystem, tempRHS)
    tempRHS[indRHS] = 0.
    for jj in range(transferMapRowsCount):
        tempMapValues[jj,indLHS] = tempSolutionVec[jj]
    for jj in range(transferMapRowsCount):
        if transferMapExp[jj,indRHS] == 0:
            for kk in range(nCoordsLHS):
                if kk != indLHS:
                    tempMapValues[jj,kk] += transferMapValues[jj,kk]
        else:     
            for kk in range(nCoordsRHS):
                if kk != indRHS:
                    startExp[kk] = transferMapExp[jj,kk]                    
                else:
                    startExp[kk] = 0
            for kk in range(nCoordsLHS):
                if kk != indLHS:
                    startValues[kk] = transferMapValues[jj,kk]
                else:
                    startValues[kk] = 0
            _substituteRecursive(tempMapValues, indLHS, nCoordsLHS, nCoordsRHS, startExp, startValues, transferMapExp, 
                                 transferMapRowsCount, transferMapExp[jj,indRHS], transferMapOrder)        
    for jj in range(transferMapRowsCount):
        for kk in range(nCoordsLHS):
            transferMapValues[jj,kk] = tempMapValues[jj,kk]  
            tempMapValues[jj,kk] = 0.
    
    return

                    
# Helper function to fill the equation system occurring in the coordinate exchange.         
# Recursive and and optimized for speed, although not very readable.               
cdef _fillEquationSystemRecursive(double[:,:] equationSystem, double[:,:] transferMapValues, unsigned int indLHS, unsigned int nCoordsRHS, 
                                  unsigned int[:] array, unsigned int equationSystemColumn, double value, unsigned int[:,:] transferMapExp, 
                                  unsigned int transferMapRowsCount, unsigned int remainingExp, unsigned int transferMapOrder):
    cdef:
        unsigned int sum = 0       
        unsigned int[:] passOnArray = numpy.zeros(nCoordsRHS, dtype=numpy.uint32)
        double passOnValue
        int idx    
    if remainingExp > 0:
        for ii in range(transferMapRowsCount):
            sum = 0
            for jj in range(nCoordsRHS):
                passOnArray[jj] = array[jj] + transferMapExp[ii,jj]  
                sum += passOnArray[jj]
            passOnValue = value*transferMapValues[ii,indLHS]
            if sum>transferMapOrder or passOnValue==0.:
                continue        
            _fillEquationSystemRecursive(equationSystem, transferMapValues, indLHS, nCoordsRHS, passOnArray, equationSystemColumn,
                                         passOnValue, transferMapExp, transferMapRowsCount, remainingExp-1, transferMapOrder)
    elif remainingExp == 0:
        for ii in range(nCoordsRHS):
            sum += array[ii]
        if sum <= transferMapOrder:
            idx = -1
            for jj in range(transferMapRowsCount):
                for kk in range(nCoordsRHS):
                    if array[kk]!=transferMapExp[jj,kk]:
                        break
                    if kk == (nCoordsRHS-1):
                        idx = jj
                if idx>-1:
                    break
            equationSystem[idx,equationSystemColumn] += value

# Helper function for the backwards substitution occurring in the coordinate exchange.
# Recursive and optimized for speed, but not very readable.                   
cdef _substituteRecursive(double[:,:] tempMapValues, unsigned int indLHS, unsigned int nCoordsLHS, unsigned int nCoordsRHS, unsigned int[:] array,
                          double[:] values, unsigned int[:,:] transferMapExp, unsigned int transferMapRowsCount, unsigned int remainingExp, unsigned int transferMapOrder):
    cdef:
        unsigned int sum = 0       
        double sum2 = 0.
        unsigned int[:] passOnArray = numpy.zeros(nCoordsRHS, dtype=numpy.uint32)
        double[:] passOnValues = numpy.zeros(nCoordsLHS, dtype=numpy.double)
        int idx         
    if remainingExp > 0:
        for ii in range(transferMapRowsCount):
            sum = 0
            for jj in range(nCoordsRHS):
                passOnArray[jj] = array[jj] + transferMapExp[ii,jj]  
                sum += passOnArray[jj]
            if sum>transferMapOrder:
                continue
            for jj in range(nCoordsLHS):
                passOnValues[jj] = values[jj]*tempMapValues[ii,indLHS]
            sum2 = 0.
            for jj in range(nCoordsLHS):
                sum2 += passOnValues[jj]
            if sum2==0.:
                continue
            _substituteRecursive(tempMapValues, indLHS, nCoordsLHS, nCoordsRHS, passOnArray,
                                 passOnValues, transferMapExp, transferMapRowsCount, remainingExp-1, transferMapOrder)         
    elif remainingExp == 0:
        for ii in range(nCoordsRHS):
            sum += array[ii]
        if sum <= transferMapOrder:
            idx = -1
            for jj in range(transferMapRowsCount):
                for kk in range(nCoordsRHS):
                    if array[kk]!=transferMapExp[jj,kk]:
                        break
                    if kk == (nCoordsRHS-1):
                        idx = jj
                if idx>-1:
                    break
            for ii in range(nCoordsLHS):
                if ii!=indLHS:
                    tempMapValues[idx,ii] += values[ii]

# Calculate the binomial coefficient.
# Recursive and fast for reasonable numbers. Used for Pascal's triangle coefficients.
cdef unsigned int _binomCoeff(unsigned int n, unsigned int k):
    if k==0 or k==n:
        return 1
    if k==1:
        return n
    else:
        return (n*_binomCoeff(n-1,k-1))/k
    
# Helper function to calculate the number of Taylor map coefficients to a given order.
cdef unsigned int _nExponentsCombinations(unsigned int transferMapOrder, unsigned int nCoordsRHS):
    cdef:
        unsigned int sum = 0, add, ii
    add = nCoordsRHS
    sum += add
    for ii in range(1,transferMapOrder):
        add *= (ii+nCoordsRHS)
        add /= (ii+1)
        sum += add
    return sum

# Helper function to calculate the different combinations of exponents in the Taylor map.
cdef _expCombinations(unsigned int[:,:] transferMapExp, unsigned int transferMapOrder, unsigned int nCoordsRHS):
    cdef:
        unsigned int ii
        unsigned int[:] array = numpy.zeros(transferMapOrder, dtype=numpy.uint32)      
        unsigned int[:] fullArrayIdx = numpy.zeros(1, dtype=numpy.uint32)
    for ii in range(transferMapOrder):
        _expCombinationsRecursive(array, ii+1, nCoordsRHS, ii+1, transferMapExp, fullArrayIdx)

cdef _expCombinationsRecursive(unsigned int[:] array, unsigned int arrayLength, unsigned int nCoordsRHS, 
                               unsigned int currentOrder, unsigned int[:,:] transferMapExp, unsigned int[:] fullArrayIdx):
    cdef unsigned int ii
    if currentOrder>0:
        for ii in range(nCoordsRHS):
            array[currentOrder-1] = ii         
            _expCombinationsRecursive(array, arrayLength, array[currentOrder-1]+1, 
                                      currentOrder-1, transferMapExp, fullArrayIdx)
    elif currentOrder==0:
        for ii in range(arrayLength):
            transferMapExp[fullArrayIdx[0],array[ii]] += 1
        fullArrayIdx[0] = fullArrayIdx[0]+1
