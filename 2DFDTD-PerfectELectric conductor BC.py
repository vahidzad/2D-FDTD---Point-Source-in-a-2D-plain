# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:20:31 2019

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
'Initialize FDTD Parameters'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Mxx =1;
Myy=1;
Ezz=1;

#Field Initialization
CEx = np.zeros(shape=(Nx,Ny));
CEy = np.zeros(shape=(Nx,Ny));
Hx = np.zeros(shape=(Nx,Ny));
Hy = np.zeros(shape=(Nx,Ny));
CHz = np.zeros(shape=(Nx,Ny));
Dz = np.zeros(shape=(Nx,Ny));
Ez = np.zeros(shape=(Nx,Ny));

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'Enter The X and Y value (coordinates for the source)'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Source_x = input("Enter a value between 10-190 for X : ") 
Source_y = input("Enter a value between 10-190 for Y : ") 
Source_x = int(Source_x)
Source_y = int(Source_y)
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
    
    #Update H
    Hx=Hx+((-c0*dt/Mxx)*CEx);
    Hy=Hy+((-c0*dt/Myy)*CEy);
            
    #Calculate CHz
    CHz[0,0]=(Hy[0,0]-0)/dx-(Hx[0,0]-0)/dy;   
    CHz[1:Nx,1]=(Hy[1:Nx,1]-Hy[0:Nx-1,1])/dx-(Hx[1:Nx,1]-0)/dy;
    CHz[1,1:Ny]=(Hy[1,1:Ny]-0)/dx-(Hx[1,1:Ny]-Hx[1,0:Ny-1])/dy;
    CHz[1:Nx-1,1:Ny]=(Hy[1:Nx-1,1:Ny]-Hy[0:Nx-2,1:Ny])/dx-((Hx[1:Nx-1,1:Ny]-Hx[1:Nx-1,0:Ny-1])/dy);
            
    #Update D
    Dz=Dz+((c0*dt)*(CHz));
            
    #Inject Source
    Dz[Source_x,Source_y]= Dz[Source_x,Source_y]+ Esrc[T];
    
    #Update E
    Ez=(1/Ezz)*Dz;
            
    #Show Results
    if np.mod(T,10)==0:
        fig=plt.imshow(Ez,cmap='jet');
        plt.title('Ez (PEC) at the time '+ str(T) +' of '+ str(steps),fontsize=15);
        plt.colorbar()
        plt.clim([0.001,0.03])
        plt.gcf();
        plt.pause(0.01)
        #plt.savefig("%d.jpg"%(T),dpi=750)
        plt.clf()
    
        
    
    
    
    
    
    
            
    

