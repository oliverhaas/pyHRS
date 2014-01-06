
filename = 'properties6.dat'

inFile = open(filename, 'r')
outFile = open('nameFormula.dat', 'w')
lines = inFile.readlines()
ii = 0
while ii < len(lines):
    currentLine = lines[ii]
    skipLines = int(currentLine[73]) + int(currentLine[65:68]) + 4
    outFile.write(lines[ii+1][2:])
    ii += skipLines
inFile.close()
outFile.close()    

inFile = open(filename, 'r')
outFile = open('iEx.dat', 'w')
lines = inFile.readlines()
ii = 0
while ii < len(lines):
    currentLine = lines[ii]
    skipLines = int(currentLine[73]) + int(currentLine[65:68]) + 4
    currentLine = lines[ii+2]
    outFile.write(str(float(currentLine[:10]))+'\n')
    ii += skipLines
inFile.close()
outFile.close()    

inFile = open(filename, 'r')
outFile = open('rho.dat', 'w')
lines = inFile.readlines()
ii = 0
while ii < len(lines):
    currentLine = lines[ii]
    skipLines = int(currentLine[73]) + int(currentLine[65:68]) + 4
    outFile.write(str(float(currentLine[53:64]))+'\n')
    ii += skipLines
inFile.close()
outFile.close()  

inFile = open(filename, 'r')
outFile = open('rhoSternheimer.dat', 'w')
lines = inFile.readlines()
ii = 0
while ii < len(lines):
    currentLine = lines[ii]
    skipLines = int(currentLine[73]) + int(currentLine[65:68]) + 4
    outFile.write(str(float(currentLine[42:53]))+'\n')
    ii += skipLines
inFile.close()
outFile.close()  

inFile = open(filename, 'r')
outFile = open('cbarSternheimer.dat', 'w')
lines = inFile.readlines()
ii = 0
while ii < len(lines):
    currentLine = lines[ii]
    skipLines = int(currentLine[73]) + int(currentLine[65:68]) + 4
    currentLine = lines[ii+2]
    outFile.write(str(float(currentLine[12:21]))+'\n')
    ii += skipLines
inFile.close()
outFile.close()   

inFile = open(filename, 'r')
outFile = open('x0Sternheimer.dat', 'w')
lines = inFile.readlines()
ii = 0
while ii < len(lines):
    currentLine = lines[ii]
    skipLines = int(currentLine[73]) + int(currentLine[65:68]) + 4
    currentLine = lines[ii+2]
    outFile.write(str(float(currentLine[21:30]))+'\n')
    ii += skipLines
inFile.close()
outFile.close()   

inFile = open(filename, 'r')
outFile = open('x1Sternheimer.dat', 'w')
lines = inFile.readlines()
ii = 0
while ii < len(lines):
    currentLine = lines[ii]
    skipLines = int(currentLine[73]) + int(currentLine[65:68]) + 4
    currentLine = lines[ii+2]
    outFile.write(str(float(currentLine[30:38]))+'\n')
    ii += skipLines
inFile.close()
outFile.close()   

inFile = open(filename, 'r')
outFile = open('aSternheimer.dat', 'w')
lines = inFile.readlines()
ii = 0
while ii < len(lines):
    currentLine = lines[ii]
    skipLines = int(currentLine[73]) + int(currentLine[65:68]) + 4
    currentLine = lines[ii+2]
    outFile.write(str(float(currentLine[38:46]))+'\n')
    ii += skipLines
inFile.close()
outFile.close()  

inFile = open(filename, 'r')
outFile = open('kSternheimer.dat', 'w')
lines = inFile.readlines()
ii = 0
while ii < len(lines):
    currentLine = lines[ii]
    skipLines = int(currentLine[73]) + int(currentLine[65:68]) + 4
    currentLine = lines[ii+2]
    outFile.write(str(float(currentLine[46:55]))+'\n')
    ii += skipLines
inFile.close()
outFile.close()  

inFile = open(filename, 'r')
outFile = open('delta0Sternheimer.dat', 'w')
lines = inFile.readlines()
ii = 0
while ii < len(lines):
    currentLine = lines[ii]
    skipLines = int(currentLine[73]) + int(currentLine[65:68]) + 4
    currentLine = lines[ii+2]
    outFile.write(str(float(currentLine[55:]))+'\n')
    ii += skipLines
inFile.close()
outFile.close() 

inFile = open(filename, 'r')
outFile = open('zOverA.dat', 'w')
lines = inFile.readlines()
ii = 0
while ii < len(lines):
    currentLine = lines[ii]
    skipLines = int(currentLine[73]) + int(currentLine[65:68]) + 4
    outFile.write(str(float(currentLine[33:41]))+'\n')
    ii += skipLines
inFile.close()
outFile.close()  

inFile = open(filename, 'r')
outFile = open('z.dat', 'w')
lines = inFile.readlines()
ii = 0
while ii < len(lines):
    currentLine = lines[ii]
    nElements = int(currentLine[65:68])
    skipLines = int(currentLine[73]) + int(currentLine[65:68]) + 4
    currentLine = lines[ii+3]
    if nElements == 1:
        outFile.write(str(float(currentLine[:10]))+'\n')
    else:
        temp = 0.
        for jj in range(nElements):
            currentLine = lines[ii+3+jj]
            temp+= float(currentLine[:10])*int(float(currentLine[10:20])+0.5)
        outFile.write(str(float(temp))+'\n')
    ii += skipLines
inFile.close()
outFile.close()  
