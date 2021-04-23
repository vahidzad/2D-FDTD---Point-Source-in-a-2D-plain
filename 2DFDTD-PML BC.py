# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:51:38 2019

@author: Ehsan
"""

import math 
import numpy as np
import matplotlib.pyplot as plt

'''FDTD Engine'''
'''Units'''
meters=1;
centimeters=meters*1e-2;
milimeters=meters*1e-3;
nanometers= meters*1e-9;
feet=0.3048*meters;
seconds=1;
hertz = 1/seconds;
megahertz=1e6*hertz;
gigahertz=1e9*hertz;

'''constants'''
c0=299792458 * meters/seconds;
e0=8.8541878176e-12*1/meters;
u0=1.2566370614*1/meters;

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'Compute Time Steps'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

steps=2000;
dt = 9.775993866961225e-12;
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'Compute Optimize Grid'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
NPML=[10,10,10,10];
Nx=200;
Ny=200;
dx=0.005861538461538;
dy=0.005861538461538;

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'Compute The Source'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Source Parameters
tau = 1.0e-10;
t0 = 5.0e-10;
#Source Calculations
t=np.arange(0,steps-1)*dt;
s=dx/(2*c0)+dt/2;
x_src =Nx/2;
y_src= Ny/2;
Esrc =np.exp(-(np.power((t-t0)/tau,2)));
A = -1;
Hsrc =A*np.exp(-(np.power((t-t0)/tau,2)));

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'Initilize PML Parameters'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Nx2 = 2*Nx;
Ny2 = 2*Ny;
sigx = np.zeros(shape=(Nx2,Ny2));
for nx in range(1,2*NPML[0]+1):
    nx1 = 2*NPML[0] - nx + 1;
    sigx[nx1-1,:] = (0.5*e0/dt)*(nx/2/NPML[0])**3;

for nx in range(1,2*NPML[1]+1):
    nx1 = Nx2 - 2*NPML[1] + nx;
    sigx[nx1-1,:] = (0.5*e0/dt)*(nx/2/NPML[1])**3;
    
sigy = np.zeros(shape=(Nx2,Ny2));
for ny in range(1,2*NPML[2]+1):
    ny1 = 2*NPML[2] - ny + 1;
    sigy[:,ny1-1] = (0.5*e0/dt)*(ny/2/NPML[2])**3;

for ny in range(1,2*NPML[3]+1):
    ny1 = Ny2 - 2*NPML[3] + ny;
    sigy[:,ny1-1] = (0.5*e0/dt)*(ny/2/NPML[3])**3;

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'Initialize FDTD Parameters'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Update Coefficients
URxx =1;
URyy=1;
URzz=1;

sigHx = sigx[1::2,::2];
sigHy = sigy[1::2,::2];
mHx0  = (1/dt) + sigHy/(2*e0);
mHx1 = np.divide(((1/dt) - sigHy/(2*e0)),mHx0);
mHx2 = -c0/URxx/mHx0;
mHx3 = -(c0*dt/e0)*np.divide(sigHx/URxx,mHx0);

sigHx = sigx[::2,1:Ny2:2];
sigHy = sigy[::2,1:Ny2:2];
mHy0  = (1/dt) + sigHx/(2*e0);
mHy1 = np.divide(((1/dt) - sigHx/(2*e0)),mHy0);
mHy2 = - c0/URyy/mHy0;
mHy3 = - (c0*dt/e0) * np.divide(sigHy/URyy,mHy0);

sigDx = sigx[0:Nx2-1:2,0:Ny2-1:2];
sigDy = sigy[0:Nx2-1:2,0:Ny2-1:2];
mDz0  = (1/dt) + (sigDx + sigDy)/(2*e0)+ np.multiply(sigDx,sigDy)*(dt/4/e0**2);
mDz1 = (1/dt) - (sigDx + sigDy)/(2*e0) - np.multiply(sigDx,sigDy)*(dt/4/e0**2);
mDz1  = mDz1 / mDz0;
mDz2  = c0/mDz0;
mDz4 = - (dt/e0**2)*np.multiply(sigDx,sigDy)/mDz0;


#Field Initialization
CEx = np.zeros(shape=(Nx,Ny));
ICEx = np.zeros(shape=(Nx,Ny));
CEy = np.zeros(shape=(Nx,Ny));
ICEy = np.zeros(shape=(Nx,Ny));
Hx = np.zeros(shape=(Nx,Ny));
Hy = np.zeros(shape=(Nx,Ny));
CHz = np.zeros(shape=(Nx,Ny));
Dz = np.zeros(shape=(Nx,Ny));
IDz = np.zeros(shape=(Nx,Ny));
Ez = np.zeros(shape=(Nx,Ny));

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'Perform FDTD'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
for T in range(steps+1):
    #Calculate CEx
    CEx[:,0:Ny-1] =((Ez[:,1:Ny]-Ez[:,0:Ny-1])/dy);
    CEx[:,Ny-1] =((0-Ez[:,Ny-1])/dy);
        
    #Calculate CEy
    CEy[0:Nx-1,:] =-((Ez[1:Nx,:]-Ez[0:Nx-1,:])/dx);
    CEy[Nx-1,:] =-(0-Ez[Nx-1,:])/dx;
    
    #Update H Integrations
    ICEx = ICEx + CEx;
    ICEy = ICEy + CEy;
    
    #Update H
    Hx = np.multiply(mHx1,Hx) + np.multiply(mHx2,CEx) + np.multiply(mHx3,ICEx);
    Hy = np.multiply(mHy1,Hy) + np.multiply(mHy2,CEy)+ np.multiply(mHy3,ICEy);
            
    #Calculate CHz
    CHz[0,0]=(Hy[0,0]-0)/dx-(Hx[0,0]-0)/dy;   
    CHz[1:Nx,1]=(Hy[1:Nx,1]-Hy[0:Nx-1,1])/dx-(Hx[1:Nx,1]-0)/dy;
    CHz[1,1:Ny]=(Hy[1,1:Ny]-0)/dx-(Hx[1,1:Ny]-Hx[1,0:Ny-1])/dy;
    CHz[1:Nx-1,1:Ny]=(Hy[1:Nx-1,1:Ny]-Hy[0:Nx-2,1:Ny])/dx-((Hx[1:Nx-1,1:Ny]-Hx[1:Nx-1,0:Ny-1])/dy);
            
    #Update D Integrations
    IDz = IDz + Dz;
    
    #Update D
    Dz = np.multiply(mDz1,Dz) + np.multiply(mDz2,CHz) + np.multiply(mDz4,IDz);
            
    #Inject Source
    Dz[100,100]= Dz[100,100]+ Esrc[T];
    
    #Update E
    Ez=Dz;
            
    #Show Results
    if np.mod(T,2)==0:
        fig=plt.imshow(Ez,cmap='jet');
        plt.title('Ez (PML) at the time '+ str(T) +' of '+ str(steps),fontsize=15);
        plt.colorbar()
        plt.clim([0.001,0.03])
        plt.gcf();
        plt.pause(0.01)
        plt.savefig("%d.jpg"%(T),dpi=750)
        plt.clf()
        
    
    
    
    
    
    
            
    

