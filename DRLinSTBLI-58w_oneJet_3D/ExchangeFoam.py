#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:33:43 2024

@author: yizhou
"""

import numpy as np
import re
import subprocess
from time import time
from scipy.interpolate import griddata
    

def recordStateLocation(rootPath):   # probes position
    
    
    fileName = '/'.join([rootPath,'system', 'probes'])
    fopen = open(fileName, 'w+')
    fopen.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
    fopen.write('  =========                 |\n')
    fopen.write('  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n')
    fopen.write('   \\\\    /   O peration     | Website:  https://openfoam.org\n')
    fopen.write('    \\\\  /    A nd           | Version:  7\n')
    fopen.write('     \\\\/     M anipulation  |\n')
    fopen.write('-------------------------------------------------------------------------------\n')
    fopen.write('Description\n')
    fopen.write('    Writes out values of fields from cells nearest to specified locations.\n')
    fopen.write(' \n')
    fopen.write('\\*---------------------------------------------------------------------------*/\n')
    fopen.write(' \n')
    fopen.write('#includeEtc "caseDicts/postProcessing/probes/probes.cfg"\n')
    fopen.write('writeControl    runTime;\n')
    fopen.write('writeInterval    2e-4;\n')
    fopen.write(' \n')
    fopen.write('fields (p U);\n')
    fopen.write('probeLocations\n')
    fopen.write('( \n')
    for x in np.linspace(-0.03,0,16):
        # for y in np.linspace(0.0005,0.004,9):
        for z in np.linspace(0,0.014,8):
            coord = '(' +' '.join(map(str,(x,0,z)))+ ')'
            fopen.write(f'    {coord}\n')
    for x in np.linspace(0.002,0.01,5):
        # for y in np.linspace(0.0005,0.004,9):
        for z in np.linspace(0,0.014,8):
            coord = '(' + ' '.join(map(str,(x,x*np.tan(np.deg2rad(15))+0.0001,z))) + ')'
            fopen.write(f'    {coord}\n')
    fopen.write(');\n')
    fopen.write(' \n')
    fopen.write('// ************************************************************************* //\n')
    fopen.close()
    
def readStatefromFoam(episodeTime, rootPath,referenceP):   # read probes pressure
    
    
    
    
    episodeTime = round(episodeTime, 8)
    # print(f'obsearvation time {episodeTime}')
    
    subprocess.run( f'source $HOME/OpenFOAM/OpenFOAM-7/etc/bashrc && cd {rootPath} && '+
                   f'postProcess -func probes -time {episodeTime}', 
                   executable='/bin/bash',shell=True,check=True,stdout=subprocess.DEVNULL)
    
    filep = '/'.join([rootPath, 'postProcessing','probes', f'{episodeTime}', 'p'])
    fopenp = open(filep,'r')
    pProbes = fopenp.readlines()
    fopenp.close()
    for line in pProbes:
        if line.split()[0] == str(episodeTime):
            pObs = np.float32(line.split()[1:])
    
    
    return pObs/referenceP*(-1)

def act2Foam(action, episodeTime, rootPath, deltaTime, processor, referenceP):  # adjust the microjet pressure
    episodeTime = round(episodeTime, 8)
    fopen = open('/'.join([rootPath, str(episodeTime), 'p']))
    UFile = fopen.readlines()
    fopen.close()
    with open('/'.join([rootPath, str(episodeTime), 'p']), 'w+') as fopen:
        for i in range(len(UFile)):
            if 'JET' in UFile[i].split():
                for j in range(action.shape[0]):
                    UFile[i+j+3] =  f'        value    uniform {action[j]};\n'
            fopen.writelines(UFile[i])
    
    
    
    # print('FoamRun', episodeTime) 
    startTime = time()    
    
    subprocess.run( f'source $HOME/OpenFOAM/OpenFOAM-7/etc/bashrc && cd {rootPath} && '+
                   'decomposePar -force && '+
                   f'mpirun -np {processor} rhoCentralFoam -parallel && ' +
                   'reconstructPar -newTimes ',
                   executable='/bin/bash',shell=True,check=True,stdout=subprocess.DEVNULL)
   
    endTime = time()
    # print(f'foamRun time {endTime-startTime}')
    
    return  readStatefromFoam(episodeTime+deltaTime, rootPath, referenceP)
        
    
def coeffsfromFoam(episodeTime, rootPath, referenceP,action):  # read normalized separation area
    episodeTime = round(episodeTime, 8)
    # print(f'get wallShearStress {episodeTime}')
    
    
    fopen = open('/'.join([rootPath, str(episodeTime), 'wallShearStress']))
    wallShearStress = fopen.readlines()
    fopen.close()
    wallShearStressList = []
    for line in wallShearStress:
        if re.search('\((\-?\d+\.?\d*e?\-?\d*?\s+\-?\d+\.?\d*e?\-?\d*?\s+\-?\d+\.?\d*e?\-?\d*?)\)',line):
            match =re.search('\((\-?\d+\.?\d*e?\-?\d*?\s+\-?\d+\.?\d*e?\-?\d*?\s+\-?\d+\.?\d*e?\-?\d*?)\)',line)
            match = re.sub('[()]', '', match.group())
            if (np.float32(match.split()[0])!=0 or 
            np.float32(match.split()[1])!=0 or np.float32(match.split()[2])!=0):
                wallShearStressList.append(np.float32(match.split()[0]))
    wallShearStressAverage = np.array(wallShearStressList)
    wallShearStressAverage = wallShearStressAverage/referenceP
    
    
    
    
    with open('/'.join([rootPath, str(0), 'C'])) as fopen:
        meshC = fopen.readlines()
        
    for index, line in enumerate((meshC)):
        if 'BOTTOM' in line:
            bottomIndex = index
       
    
    for i in range(bottomIndex, len(meshC)):
        if '{' in meshC[i]:
            start = i
        elif '}' in meshC[i]:
            end = i
            break
        
    meshList = []
    for i in range(start,end):
        if re.search('\((\-?\d+\.?\d*e?\-?\d*?\s+\-?\d+\.?\d*e?\-?\d*?\s+\-?\d+\.?\d*e?\-?\d*?)\)', meshC[i]):
            match = re.search('\((\-?\d+\.?\d*e?\-?\d*?\s+\-?\d+\.?\d*e?\-?\d*?\s+\-?\d+\.?\d*e?\-?\d*?)\)', meshC[i])
            match = re.sub('[()]', '', match.group())
            meshList.append(np.float32(match.split()[0]))
            meshList.append(np.float32(match.split()[2]))
            
    meshList = np.array(meshList)
    meshList = meshList.reshape(-1,2)
    
    
    
    gridX, gridY = np.mgrid[-0.05:0.05:500j,0:0.014:500j]
    Cf = griddata(meshList,wallShearStressAverage,(gridX,gridY),method='linear',fill_value=1)   
    negativeIndex = Cf <0
    counts = np.sum(negativeIndex)
    reward= counts/42725*(-1)   # 42725 is the cell number in the separation zone without control
    # print(f'separation area = {reward}')
                           
    return  reward
    

if __name__ == '__main__':
    
    rootPath = '/Data/totalPBc'
    referenceP = (0.5*0.005275*1380.22*1380.22)*(-1)
    
    recordStateLocation(rootPath)    
    Obs = readStatefromFoam(0, rootPath,referenceP)
    
