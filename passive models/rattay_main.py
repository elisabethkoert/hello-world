import sys
import neuron1 as n1
import math
import functions as f
import numpy as np
import globalvariables as gv
from matplotlib import pyplot as plt


def runmain():
    
    #.buildanddraw()
    
    #rattay 1986 unmyleinated axon:
    #replicate fig. 4
    gv.rho_e=3 #Ohm*m  from 0.3kOhmcm everything is kept in SI and only changed during plotting
    z=1e-3 #m
    x=np.arange(-5e-3,5e-3,0.1e-3)#m
    I_el=[-290e-6,1450e-6]#A
    V_e=[]
    AF=[]
    Ve1=f.calculateVe(x, z, I_el[0])
    V_e.append(Ve1)
    Ve2=f.calculateVe(x, z, I_el[1])
    V_e.append(Ve2)
    #f.plotMultiple(V_e, x, legende=['-290 µA','1450 µA'],xlabel='position [mm]',ylabel='extracellular potential [mV]')
    
    AF1=f.calculateAF(x, z, I_el[0])
    AF.append(AF1)
    AF2=f.calculateAF(x, z, I_el[1])
    AF.append(AF2)
    #f.plotMultiple(AF, x, legende=['stimulation with -290 µA','1450 µA'], ylabel='dVe2/d2x [V/cm2]',yfromSI=1e-4)
    
    #try to make joint 
    makeFig4a(x,V_e,AF)




    
    
    
    





    
if __name__ == '__main__':
    print(sys.argv)    
    runmain()