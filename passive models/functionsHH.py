#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:29:22 2020

@author: elisabeth
"""
import globalvariables as gv
import numpy as np
import math
import pandas as pd

import matplotlib.pyplot as plt


from scipy.integrate import odeint
#from scipy.integrate import solve_ivp #more modern function, but does not work yet


def makeStrengthDurationCurve(gK=36.0,gNa=120.0,gL=0.3,Cm=1.0,VK=-12.0,VNa=115.0,Vl=10.613,tmin=0.0,tmax=50.0):
  
    #tdur=np.linspace(0.01,20,10)
    tdur=np.logspace(-2,1,50)
    I=np.logspace(-1.0,4,50)#log numbers between 1 and 1000
    thresholds=[-2] * len(tdur)
    
    for i in range(len(tdur)):
        print('tdur='+str(tdur[i]))
        APdetected=False
        gv.I_tend=tdur[i]
        which_i=0
        
        while not APdetected:
            gv.I_amplitude=I[which_i]
            #print('round' +str(which_i)+' with current '+str(I[which_i]))
            Vm,T=calculateHH()
            APdetected=tktSimpleDetectAP_TF(Vm)
            if APdetected:    
                thresholds[i]=I[which_i]
            which_i+=1
            if which_i==len(I):
                thresholds[i]=-1
                APdetected=True
            
        print('done with this tdur') 
        
    print(tdur)
    print(len(tdur))
    print(thresholds)
    print(len(thresholds))
    plt.figure()
    plt.plot(tdur,thresholds)
    plt.xlabel('pulse length in ms')
    plt.ylabel('Threshold current in uA/cm2')
    plt.show()
    
    
    plt.figure()
    plt.plot(tdur[10:30],thresholds[10:30])
    plt.xlabel('pulse length in ms')
    plt.ylabel('Threshold current in uA/cm2')
    plt.show()

    return thresholds


def calculateHH(gK=36.0,gNa=120.0,gL=0.3,Cm=1.0,VK=-12.0,VNa=115.0,Vl=10.613,tmin=0.0,tmax=50.0):
    """
    calculates the change in membrane potential at a specific point at a membrane in response to a introduced current
    
    Parameters
    ----------
    gK : float, optional
        Average potassium channel conductance per unit area (mS/cm^2). The default is 36.0.
    gNa : float, optional
       Average sodium channel conductance per unit area (mS/cm^2). The default is 120.0.
    gL : float optional
        Average leak channel conductance per unit area (mS/cm^2). The default is 0.3.
    Cm : float, optional
        Membrane capacitance per unit area (uF/cm^2) The default is 1.0.
    VK : float, optional
        Potassium potential (mV) #nernst. The default is -12.0.
    VNa : float, optional
        Sodium potential (mV) #nernst. The default is 115.0.
    Vl : float, optional
        Leak potential (mV) #nernst. The default is 10.613.
    tmin : float, optional
        start time in ms. The default is 0.0.
    tmax : float, optional
        ens time in ms The default is 50.0.

    Returns
    -------
    None.

    """
     
    # Set random seed (for reproducibility)
    np.random.seed(1000)
    
    # Time values
    T = np.linspace(tmin, tmax, 10000)
    gv.dt=(tmax-tmin)/10000
    
    #use y as an array that contains all time dependent parameters, V,n,m,h
    # State (Vm, n, m, h) #set initial values
    Y = np.array([0.0, n_inf(), m_inf(), h_inf()])
    #formulate derivatives
    #dy=formulate_derivatives(Y, T[0],gK,gNa,gL,Cm,VK,VNa,Vl)
    
    # Solve ODE system
    # Vy = (Vm[t0:tmax], n[t0:tmax], m[t0:tmax], h[t0:tmax])
    Vy = odeint(compute_derivatives, Y, T,args=(gK,gNa,gL,Cm,VK,VNa,Vl))
    #Vy=solve_ivp(compute_derivatives, [tmin,tmax], Y,t_eval=T,args=(gK,gNa,gL,Cm,VK,VNa,Vl))
    #Vy.t gives back T
    #Vy.y gives back the solution
    
    
    if gv.display==True:
        '''make plots'''
        # Input stimulus
        Idv = [Id(t) for t in T]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(T, Idv)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel(r'Current density (uA/$cm^2$)')
        ax.set_title('Stimulus (Current density)')
        plt.grid()
        
        # Neuron potential
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(T, Vy[:, 0])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Vm (mV)')
        ax.set_title('Neuron potential with two spikes')
        plt.grid()
        
        
        #check if APS are detected
       # spikes=tktSimpleDetectAP(Vy[:, 0])
        #print(spikes)
        #detected=tktSimpleDetectAP_TF(Vy[:, 0])
        #print(detected)
        
        #n m and h
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(T, Vy[:, 1],label='n')
        ax.plot(T, Vy[:, 2],label='m')
        ax.plot(T, Vy[:, 3],label='h')
        plt.legend()
        ax.set_xlabel('Time (ms)')
        ax.set_title('changes in n, m and h with two spikes')
        plt.grid()
        
    
    return Vy[:, 0],T


"""
All the used functions are down below
"""
def compute_derivatives(Y,t0,gK,gNa,gL,Cm,VK,VNa,Vl):
        dy = np.zeros((4,))
        
        Vm = Y[0]
        n = Y[1]
        m = Y[2]
        h = Y[3]
        
        # dVm/dt
        GK = (gK / Cm) * np.power(n, 4.0)
        GNa = (gNa / Cm) * np.power(m, 3.0) * h
        GL = gL / Cm
        
        dy[0] = (Id(t0) / Cm) - (GK * (Vm - VK)) - (GNa * (Vm - VNa)) - (GL * (Vm - Vl))
        
        # dn/dt
        dy[1] = (alpha_n(Vm) * (1.0 - n)) - (beta_n(Vm) * n)
        
        # dm/dt
        dy[2] = (alpha_m(Vm) * (1.0 - m)) - (beta_m(Vm) * m)
        
        # dh/dt
        dy[3] = (alpha_h(Vm) * (1.0 - h)) - (beta_h(Vm) * h)
        
        return dy

# Potassium ion-channel rate functions

def alpha_n(Vm):
    return (0.01 * (10.0 - Vm)) / (np.exp(1.0 - (0.1 * Vm)) - 1.0)

def beta_n(Vm):
    return 0.125 * np.exp(-Vm / 80.0)

# Sodium ion-channel rate functions

def alpha_m(Vm):
    return (0.1 * (25.0 - Vm)) / (np.exp(2.5 - (0.1 * Vm)) - 1.0)

def beta_m(Vm):
    return 4.0 * np.exp(-Vm / 18.0)

def alpha_h(Vm):
    return 0.07 * np.exp(-Vm / 20.0)

def beta_h(Vm):
    return 1.0 / (np.exp(3.0 - (0.1 * Vm)) + 1.0)
  
# n, m, and h steady-state values

def n_inf(Vm=0.0):
    return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))

def m_inf(Vm=0.0):
    return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))

def h_inf(Vm=0.0):
    return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))



# Input stimulus
def Id2times(t):
    'this is an example stimulation where a 150 mA pulse is given at t[0,1] and 50 ma btw t[10,11]'
    if 0.0 < t < 1.0:
        return 150.0
    elif 20.0 < t < 21.0:
        return 50.0
    return 0.0

def Id(t,tstart=gv.I_tstart,tend=gv.I_tend,amplitude=gv.I_amplitude):
    tend=gv.I_tend
    amplitude=gv.I_amplitude
    if tstart < t < tend:
        return amplitude
    return 0.0




#more needed stuff
def tktSimpleDetectAP(V,thr=-100,dt=gv.dt,LM=-20,RM=10):
    """
    Detect spikes in simulated Vm without knowing the Spt or with many inputs.
    Using a dV/dt threshold of -100mV/ms usually is robust.
    from Thomas Kuenzel
    """
    T = np.linspace(0,(len(V)*dt)-dt,len(V))
    dV = np.diff(V)/dt
    Ilow=np.where(dV<thr)[0]
    Ilow = np.concatenate(([0],Ilow))
    dIlow=np.diff(Ilow)
    firstIlow=np.where(dIlow>1.1)[0]
    DetectI=Ilow[firstIlow+1]
    DetectT = T[DetectI]
    PeakI = []
    PeakT = []
    PeakV = []
    for nEv,IEv in enumerate(DetectI):
        if IEv+LM < 0:
            localI=V[0:IEv+RM].argmax()-IEv
            PeakV.append(V[0:IEv+RM].max())
        elif IEv+RM > len(V):
            localI=V[IEv+LM:len(V)].argmax()+LM
            PeakV.append(V[(IEv+LM):len(V)].max())
        else:
            localI=V[(IEv+LM):(IEv+RM)].argmax()+LM
            PeakV.append(V[(IEv+LM):(IEv+RM)].max())
        PeakI.append(IEv+localI)
        PeakT.append(T[PeakI[-1]])
    
            
    Res = {}
    Res['PeakI']=PeakI
    Res['PeakT']=PeakT
    Res['PeakV']=PeakV
    Res['DetectI']=DetectI
    Res['DetectT']=DetectT
    Res2=pd.DataFrame.from_dict(Res)

    return(Res2)

def tktSimpleDetectAP_TF(V,thr=-100,dt=gv.dt,LM=-20,RM=10):
    """
    Detect spikes in simulated Vm without knowing the Spt or with many inputs.
    Using a dV/dt threshold of -100mV/ms usually is robust.
    from Thomas Kuenzel
    """
    res=False
    T = np.linspace(0,(len(V)*dt)-dt,len(V))
    dV = np.diff(V)/dt
    Ilow=np.where(dV<thr)[0]
    Ilow = np.concatenate(([0],Ilow))
    dIlow=np.diff(Ilow)
    firstIlow=np.where(dIlow>1.1)[0]
    DetectI=Ilow[firstIlow+1]
    DetectT = T[DetectI]
    PeakI = []
    PeakT = []
    PeakV = []
    for nEv,IEv in enumerate(DetectI):
        if IEv+LM < 0:
            localI=V[0:IEv+RM].argmax()-IEv
            PeakV.append(V[0:IEv+RM].max())
        elif IEv+RM > len(V):
            localI=V[IEv+LM:len(V)].argmax()+LM
            PeakV.append(V[(IEv+LM):len(V)].max())
        else:
            localI=V[(IEv+LM):(IEv+RM)].argmax()+LM
            PeakV.append(V[(IEv+LM):(IEv+RM)].max())
        PeakI.append(IEv+localI)
        PeakT.append(T[PeakI[-1]])
    
    if len(PeakT)>=1:
        res=True
   

    return(res)

