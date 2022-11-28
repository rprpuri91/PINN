
import torch
from torch import nn
from math import cos, exp, sin, sqrt
from math import pi, atan, isclose
import numpy as np
import time
import pickle
import h5py
import os
import matplotlib.pyplot as plt
import scipy as sc
from pycse import bvp

class Blasius_Preprocessing():
    def __init__(self,  U0, L, rho0,mu, X):

        self.U0 = U0
        self.L = L
        self.rho0 = rho0
        self.mu = mu
        self.X = X
        self.f0 = 0
        self.g0 = 0
        self.h0 = np.array([0.1,0.2,0.3])

    def reynolds_fn_x(self,x):
        Rex = self.rho0*self.U0*x/self.mu
        return Rex

    def boundary_thickness(self,x):
        dx = x/sqrt(self.reynolds_fn_x(x))
        return dx

    def independent_eta(self):
        eta_values = []
        for i in range(len(self.X)):
            print(i)
            if self.X[i][0]==0 and self.X[i][1]==0:
                eta_values.append(0)
            else:
                eta = self.X[i][1]/self.boundary_thickness(self.X[i][0])
                eta_values.append(eta)
        return np.array(eta_values)

    def blasius_eq_rk(self):
        #f''' + 1/2(ff') = 0
        #f(0) = 0
        #f'(0) = 0
        #f'(inf) = 1
        #Shooting method
        #f = f1
        #f' = g = f2
        #g' = h = f3
        #h' = f''' = -1/2(fh)
        #f(0) = f'(0) = g(0)
        '''implent Runge-kutta K4 solution'''
    def rungekutta(self):
        eta = self.independent_eta()
        N = len(eta)



    def blasius_eq(self):
        eta = self.independent_eta()
        print(eta)
        f10 = eta
        f20 = np.exp(-eta)
        f30 = np.exp(-eta)
        Finitial = np.column_stack([f10,f20,f30])

        sol = bvp(odefun=self.ode,bcfun=self.bc,X= eta,yinit = Finitial)
        f1,f2,f3 = sol.T

        plt.plot(eta, f1)
        plt.show()

    def ode(self,F,x):
        f1,f2,f3 = F.T
        return np.column_stack([f2,f3,-0.5*f1*f3])

    def bc(self,Y):
        fa, fb = Y[0,:], Y[-1,:]
        return [fa[0],fa[1],1.0 - fb[1]] #f1(0) = 0, f2(0)=0, f2(inf) = 1





s = 0.05
L = 10
x_values = np.arange(0,L,s).tolist()
y_values = np.arange(0,3,s).tolist()
x_values.append(L)
y_values.append(3)
U0 = 10
rho0 = 1
mu = 1.81e-5


x,y = np.meshgrid(x_values,y_values)

X_in = np.hstack([x.flatten()[:, None], y.flatten()[:, None]])

preprocessing = Blasius_Preprocessing(U0,L, rho0,mu,X_in)
preprocessing.blasius_eq()
