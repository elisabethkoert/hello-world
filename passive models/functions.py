#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:15:40 2020

@author: elisabeth

All the functions are in here to be called from the main to execute an experiment

"""
import numpy as np
import globalvariables as gv
from matplotlib import pyplot as plt
import math



def calculateAF(x,z,I_el):
    AF=(gv.rho_e*I_el*(2*x**2-z**2))/(4*math.pi*(z**2+x**2)**(5/2))
    return AF
    
def calculateVe(x=[-1e-3,0,1e-3],z=1e-3,I_el=-290e-6):
    """
    Calculate the extracellular Potential along an unmyelinated axon generated by a monopolar current source 
    
    Parameters
    ----------
    x : Array of floats, optional
        Positions of interest along the axon. The default is [-1e-3,0,1e-3].
    z : float, optional
        distance electrode to axon. The default is 1e-3.
    I_el : Float, optional
        Stimulus current. The default is -290e-6.

    Returns
    -------
    Ve : array of floats
        extracellular potential at any point along the unmyelinated fiber.
    """
    Ve=(gv.rho_e*I_el)/(4*math.pi*np.sqrt(z**2+x**2))#Volt
    return Ve

def calculateVeR(r=1e-3,I_el=-290e-6):
    """
    Calculate the extracellular potential when the radius btw the compartment and electrode is known

    Parameters
    ----------
    r : array of floats, optional
        contains the distances etween each compartment and the current source. The default is 1e-3.
    I_el : Float, optional
        Stimulus current. The default is -290e-6.

    Returns
    -------
    Ve : array of floats
        extracellular potential at any point along the unmyelinated fiber.

    """
    Ve=(gv.rho_e*I_el)/(4*math.pi*r)#Volt
    return Ve





def plotSingle(y,x,legende=None,title=None,xlabel='position [mm]',ylabel='extracellular potential [mV]',xfromSI=1e3,yfromSI=1e3):
    """
    Plot a single plot with the possibility to change the units awy from SI
    """
    plt.figure(figsize=(8,4))
    plt.plot(x*xfromSI,y*yfromSI)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
           
    if legende != None:
        plt.legend(legende,loc='best')
    if title != None:
        plt.title(title)
    
    if gv.save==True:
        name=createFileName('figure',gv.figid,gv.folder)
        plt.savefig(name,format='pdf')
    plt.show()


def plotMultiple(y,x,legende=None,title=None,xlabel='position [mm]',ylabel='extracellular potential [mV]',xfromSI=1e3,yfromSI=1e3):
    """
    Plot multiple plots with the possibility to change them from SI, standard is going into mV & mm
    """
    plt.figure(figsize=(8,4))
    for i in range(len(y)):
        plt.plot(x*xfromSI,y[i]*yfromSI)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
           
    if legende != None:
        plt.legend(legende,loc='best')
    if title != None:
        plt.title(title)
    
    if gv.save==True:
        name=createFileName('figure',gv.figid,gv.folder)
        plt.savefig(name,format='pdf')
    plt.show()


def makeFig4a(x,V_e,AF):
    """
    reproduce the plot from rattay 1986
    """
    fig, ax = plt.subplots(constrained_layout=True)
    plt.plot(x*1e3,V_e[0]*1e3, label='extracellular potential for stimulation with -290 µA')

    
    plt.plot(x*1e3,AF[0]*1e-4, label='AF for stimulation with -290 µA')
    plt.plot(x*1e3,AF[1]*1e-4, label='AF for stimulation with 1450 µA')
    
    ax.set_ylabel('extracellular Potential [mV]')
    secaxy = ax.secondary_yaxis('right')
    secaxy.set_ylabel('AF [V/cm2]' )
    plt.legend(loc=[0,0.2])
    plt.show()


def createFileName(typ,x,folder=None):
    """
    creates a filename in reference to the type of safed data and an iterating identation number
    it is possible to save in a folder in the same dictionary
    """
    if typ=='figure':
        name= typ + str(x) + '.pdf'
        gv.figid+=1
    else:
        name= typ + str(x) + '.npy'
    if folder!=None:
        name=folder + '/' +name
    
    return name

