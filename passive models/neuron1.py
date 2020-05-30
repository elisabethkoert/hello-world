import sys
from matplotlib import pyplot as plt
import numpy as np


def runmain():
    
    draw2D()
    
    #draw3D()
    
    
    


def draw2D():
    """
    This function draws the shape of neuron 1 in 2D
    Returns
    -------
    None.

    """
    daxon=0.2#um
    lnode=0.1
    linternode=5#270
    dsoma=1
    linitialseg=0.3#falsch
    lunmyelinated=0.5#falsch
    ldend=0.5#falsch
    ddend=0.4
    
    
 
    plt.axes()

    soma = plt.Circle((0, 0), radius=dsoma/2, fc='y')
    plt.gca().add_patch(soma)


    initialseg=plt.Rectangle((dsoma/2, -daxon/2), linternode, daxon, fc='r')
    plt.gcf().gca().add_patch(initialseg)
    
    nodes=[]
    for i in range(10):
        node= plt.Rectangle((dsoma/2+linitialseg+(i+1)*linternode+i*lnode, -daxon/2), lnode, daxon, fc='black') #lower left corner width, length
        nodes.append(node)
        plt.gcf().gca().add_patch(node)
        
    internodes=[]
    for i in range(11):
        internode= plt.Rectangle((dsoma/2+linitialseg+i*linternode+(i)*lnode, -daxon/2), linternode, daxon, fc='blue') 
        internodes.append(internode)
        plt.gcf().gca().add_patch(internode)
        
    unmyelinaxon=[]
    for i in range(6):
        unmyelinated= plt.Rectangle((dsoma/2+linitialseg+11*linternode+10*lnode+i*lunmyelinated, -daxon/2), lunmyelinated, daxon, fc='r') 
        unmyelinaxon.append(unmyelinated)
        plt.gcf().gca().add_patch(unmyelinated)
        
    dendrites=[]
    for i in range(10):
        ddend=ddend*(10-i)/10 #if tapered
        dend= plt.Rectangle((-dsoma/2-(i+1)*ldend, -ddend/2), ldend, ddend , fc='green') 
        dendrites.append(dend)
        plt.gcf().gca().add_patch(dend)
  
    
    #plt.axis('scaled')
    plt.axis([-2,7,-2,2])
    plt.xlabel('um')
    plt.ylabel('um')
    plt.show()
        
    
def draw3D():
    """
    This function draws the shape of neuron 1 in 3D
    Returns
    -------
    None.

    """
    daxon=0.2#um
    lnode=0.02#1e-7
    linternode=10#270
    dsoma=1
    linitialseg=0.1#falsch
    lunmyelinated=0.5#falsch
    ldend=1#falsch
    ddend=0.3#falsch
    
    
 
    plt.axes()

    soma = plt.Circle((0, 0), radius=dsoma/2, fc='y')
    plt.gca().add_patch(soma)


    initialseg=plt.Rectangle((dsoma/2, -daxon/2), linternode, daxon, fc='r')
    plt.gcf().gca().add_patch(initialseg)
    
    nodes=[]
    for i in range(10):
        node= plt.Rectangle((dsoma/2+linitialseg+(i+1)*linternode+i*lnode, -daxon/2), lnode, daxon, fc='r') #lower left corner width, length
        nodes.append(node)
        plt.gcf().gca().add_patch(node)
        
    internodes=[]
    for i in range(11):
        internode= plt.Rectangle((dsoma/2+linitialseg+i*linternode+(i-1)*lnode, -daxon/2), linternode, daxon, fc='blue') 
        internodes.append(internode)
        plt.gcf().gca().add_patch(internode)
        
    unmyelinaxon=[]
    for i in range(6):
        unmyelinated= plt.Rectangle((dsoma/2+linitialseg+11*linternode+10*lnode+i*lunmyelinated, -daxon/2), lunmyelinated, daxon, fc='r') 
        unmyelinaxon.append(unmyelinated)
        plt.gcf().gca().add_patch(unmyelinated)
        
    dendrites=[]
    for i in range(10):
        dend= plt.Rectangle((-dsoma/2-(i+1)*ldend, -ddend/2), ldend, ddend, fc='green') 
        dendrites.append(dend)
        plt.gcf().gca().add_patch(dend)
    
    
        
        
    
  
    
    plt.axis('scaled')
    #plt.axis([-2,2,-2,2])
    plt.xlabel('um')
    plt.ylabel('um')
    plt.show()    
    
    
    
if __name__ == '__main__':
	print(sys.argv)	
	runmain()
