
import os
from typing import ForwardRef
import torch
from torch import nn
from torch._C import Size, dtype
import torch.autograd as autograd
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
from torch import optim
from torch import tensor
import numpy as np
import scipy.io 
from pyDOE import lhs
import time
import functools
import operator
from math import cos, exp, sin,sqrt
import math
from pyDOE import lhs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import pickle
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal as sp

torch.manual_seed(1234)

np.random.seed(1234)

###########################################################################################################################################################
class Flow_pinn_f(nn.Module):
    def __init__(self,layers,ub,lb,device):
        super().__init__()
        self.device =device
        self.lb = lb
        self.ub = ub
        #self.X_in_train = X_in_train
        #self.f_train = f_train
        self.layers = layers 

        #Activation
        self.activation = nn.Tanh()
<<<<<<< HEAD
=======
        self.activation2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
        
        #Loss
        self.loss_function = nn.MSELoss(reduction ='mean')

        #layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])  

        self.iter = 0
        #xavier_normalization
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain =5/3)

            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self,x):

            if torch.is_tensor(x) != True:         
                x = torch.from_numpy(x)          
            x= x.to(device)
            u_b = torch.from_numpy(self.ub).float().to(self.device)
            l_b = torch.from_numpy(self.lb).float().to(self.device)
                      
                #preprocessing input 
            x = (x - l_b)/(u_b - l_b) #feature scaling
        
                #convert to float
            a = x.float()
                        
            '''     
            Alternatively:
        
            a = self.activation(self.fc1(a))
            a = self.activation(self.fc2(a))
            a = self.activation(self.fc3(a))
            a = self.fc4(a)
        
            '''
<<<<<<< HEAD
            for i in range(len(self.layers)-2):
=======
            for i in range(len(self.layers)-3):
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
        
            
                 z = self.linears[i](a)
                        
<<<<<<< HEAD
                 a = self.activation(z)
=======
                 a = self.activation2(z)

            a = self.activation(a)
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
            
            a = self.linears[-1](a)
        
            return a   


    


###################################################################################################################################################


class Flow_PINN(nn.Module):
    def __init__(self,layers,X_in,X_initial,X_bc,ub_X,lb_X,device,k):
        super().__init__() 

        self.device =device
        #self.lb = lb
        #self.ub = ub
        self.X_in = X_in
        self.layers = layers     
        self.X_initial = X_initial
        self.X_bc =X_bc
        self.modeleq = Flow_pinn_f(layers,ub_X,lb_X,device)
        self.modelneq = Flow_pinn_f(layers,ub_X,lb_X,device)
        self.k=k
        
        #initial data
        X_initial_train,f_eqin, f_neqin, X_test_tensorin, uin,u_trainin,p_trainin, fin, feq_trainin, fneq_trainin  = training_data(self.X_initial,self.device)
        self.X_initial_train = X_initial_train
        self.u_train_in = u_trainin
        self.feq_trainin = feq_trainin
        self.fneq_trainin = fneq_trainin

        #BC data
        X_bc_train,f_eqbc, f_neqbc, X_test_tensorbc, ubc,u_trainbc,p_trainbc, fbc, feq_trainbc, fneq_trainbc  = training_data(self.X_bc,self.device)
        self.X_bc_train = X_bc_train
        self.u_trainbc = u_trainbc
        self.p_trainbc = p_trainbc
        self.feq_trainbc = feq_trainbc
        self.fneq_trainbc = fneq_trainbc

        #loss function
        self.loss_function = nn.MSELoss(reduction ='mean')

        #Error and loss data
        self.error_f=[]
        self.trainingloss=[]      
        self.residual_data =[]

        self.iter = 0

        self.iter_divider =5
<<<<<<< HEAD
=======

        self.loss_res=[]
        self.loss_ic=[]
        self.loss_bc=[]
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
      

    def f_i_sum(self,f):
        
        size = f.size()[0]
        fneq_i_sum = torch.zeros(size,device=device)
        for i in range(size):
            fn_i = sum(f[i])
            fn_i = fn_i.item()
            fneq_i_sum[i] = fn_i

        return fneq_i_sum

    def pred_variables(self,f_pred):

        xi = xi_values()
        xi1 = np.array(xi)[:,0]
        xi2 = np.array(xi)[:,1]
        
        xi1 = torch.from_numpy(xi1).float().to(self.device)
        xi2 = torch.from_numpy(xi2).float().to(self.device)

        size = f_pred.size()[0]
        row = torch.zeros(size,device=self.device)
        xi_f = torch.zeros(size,2,device=self.device)
        u = torch.zeros(size,2,device=self.device)
        #print('empty',u)
       
        for i in range(size):
            row_i = sum(f_pred[i])
            #row_i = row_i.item()
            row[i]=row_i
            ui0 = sum(torch.mul(xi1,f_pred[i]))
            #ui0 = ui0.item()
            vi0 = sum(torch.mul(xi2,f_pred[i]))
            #vi0 = vi0.item()
            if row_i == 0.0:
                ui=0.0
                vi=0.0
            else:
                ui = torch.div(ui0,row_i)            
                vi = torch.div(vi0,row_i)
            
            u_temp = torch.tensor([ui,vi],device=self.device)            
            u[i] = u_temp   
            xi_f_temp = torch.tensor([ui0,vi0],device=self.device)
            xi_f[i] = xi_f_temp
            del u_temp
            del xi_f_temp
        
        return row,u,xi_f

    def f_pred(self,x,y):
        g= torch.cat((x,y),dim=1)
        fneq_out = self.modeleq.forward(g)
        feq_out = self.modelneq.forward(g)
        return feq_out+fneq_out/self.k

<<<<<<< HEAD
=======
    def f_normalization(self,f):
        mean_f = torch.mean(f, (0),True).to(self.device)
        std_f = torch.std(f, (0), True).to(self.device)
        f_norm =  ((f-mean_f)/std_f).to(self.device)

        return mean_f,std_f,f_norm
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c

    def residual(self, g, feq_out):     
        
        #print('resi')
        xi = xi_values()
        xi1 = np.array(xi)[:,0]
        xi2 = np.array(xi)[:,1]

        tau = 1.58e-4
        
        X = torch.split(g,1,dim=1)
       
        x = X[0]
        y = X[1]

        xi1 = torch.from_numpy(xi1).float().to(self.device)
        xi2 = torch.from_numpy(xi2).float().to(self.device)
        xi1 = torch.reshape(xi1,(9,1))
        xi2 = torch.reshape(xi2,(9,1))
        summed_loss=0
        residual_full=[]
        for i in range(9):
            v = torch.zeros_like(feq_out, device=self.device)
            v[:, i] = 1 
            # f = corresponds to the value (return of compute_f)
            # df should be a tuple of the form df = (d/dx, d/dy, d/dt)
            f, df = torch.autograd.functional.vjp(self.f_pred, (x,y), v, create_graph=True)
            dfdx=df[0].to(self.device)
            dfdy= df[1].to(self.device)
            #print(torch.cuda.memory_stats(device=self.device))
            # res = df[:, 2] + xi1[i]*df[:, 0] + xi2[i]*df[:, 1] + (f_pred[:, i] - feq_out[:, i])/tau 
            # no time input
            
            f_i= f[:, i]
            f_eq_i = feq_out[:, i]
            size = f_i.size()[0]
            f_i = torch.reshape(f_i, (size,1))
            f_eq_i = torch.reshape(f_eq_i, (size,1))
            res = xi1[i]*dfdx + xi2[i]*dfdy + (f_i - f_eq_i)/tau
          
            with torch.no_grad():
                residual_full.append(res)
<<<<<<< HEAD
            target = torch.zeros_like(res,device=self.device)
            loss = self.loss_function(res,target)
            summed_loss += loss
        
        self.residual_data = residual_full           

        return summed_loss,residual_full
=======
            #target = torch.zeros_like(res,device=self.device)
            #loss = self.loss_function(res,target)
            #summed_loss += loss
        
        self.residual_data = residual_full           

        return residual_full
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c

    

    def loss_residual(self):
        file = open('total_data.pkl', 'rb')
        data = pickle.load(file)
        X_in_train = data[0]
<<<<<<< HEAD
       
        g= X_in_train.clone()
        fneq_out = self.modelneq.forward(g)
        feq_out = self.modeleq.forward(g)
        summed_loss,_ = self.residual(g, feq_out)
=======
        feq_trainin = data[7].to(device)

        g= X_in_train.clone()

        mean_feq, std_feq, feq_norm = self.f_normalization(feq_trainin)

        #fneq_out = self.modelneq.forward(g)
        feq_out = self.modeleq.forward(g)

        feq_out_denorm = std_feq*feq_out + mean_feq

        residual_full = self.residual(g, feq_out_denorm)

        residual = torch.cat(residual_full, dim=1)       

        target = torch.zeros_like(residual,device=self.device)

        loss = self.loss_function(residual,target)
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
        #summed_loss=0
        ## for each output (i.e. for each distribution)
        #for i in range(9):
        #    resi = self.residual(g, feq_out, fneq_out, i)
        #    target = torch.zeros_like(resi,device=self.device)
        #    loss = self.loss_function(resi,target)
        #    summed_loss += loss
        
<<<<<<< HEAD
        print('residual',summed_loss)


        return summed_loss
        

    def lossIC(self):
        
        fneq_out = self.modelneq.forward(self.X_initial_train)
        feq_out = self.modeleq.forward(self.X_initial_train)
        

        loss_f_neq=0
        for i in range(feq_out.shape[1]):
            loss_f_neq+=self.loss_function(fneq_out[:,i],self.k*self.fneq_trainin[:,i])

        loss_f_eq=0
        for i in range(feq_out.shape[1]):
            loss_f_eq+=self.loss_function(feq_out[:,i],self.feq_trainin[:,i])

        lossfIC  = loss_f_eq+loss_f_neq
        f_pred = feq_out + fneq_out/self.k
        
        _,u_pred,_ = self.pred_variables(f_pred)
        
        lossmIC =  self.loss_function(u_pred,self.u_train_in)

        loss = lossfIC +lossmIC
        
        print('ic',loss)
=======
        #print('residual',summed_loss)
        self.loss_res.append(loss)

        return loss
        

    def lossIC(self):

        file = open('total_data.pkl', 'rb')
        data = pickle.load(file)
        X_in_train = data[0]
        feq_trainin = data[7].to(device)
        fneq_trainin = data[8].to(device)
        u_trainin = data[5].to(device)

        g = X_in_train.clone()
              
        fneq_out = self.modelneq.forward(g)
        feq_out = self.modeleq.forward(g)
        

        mean_feq, std_feq, feq_norm = self.f_normalization(feq_trainin)
        mean_fneq, std_fneq, fneq_norm = self.f_normalization(fneq_trainin)

        loss_f_neq=0
        for i in range(feq_out.shape[1]):
            loss_f_neq+=self.loss_function(fneq_out[:,i],self.k*fneq_norm[:,i])

        loss_f_eq=0
        for i in range(feq_out.shape[1]):
            loss_f_eq+=self.loss_function(feq_out[:,i],feq_norm[:,i])

        lossfIC  = loss_f_eq+loss_f_neq

        feq_out_denorm = std_feq*feq_out + mean_feq
        fneq_out_denorm = std_fneq*fneq_out + mean_fneq

        f_pred = feq_out_denorm + fneq_out_denorm/self.k
        
        _,u_pred,_ = self.pred_variables(f_pred)
        
        lossmIC =  self.loss_function(u_pred,u_trainin)

        loss = lossfIC +lossmIC
        
        self.loss_ic.append(loss)
        #print('ic',loss)
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
        return loss

    def lossBC(self):
        
        fneq_out = self.modelneq.forward(self.X_bc_train)
        feq_out = self.modeleq.forward(self.X_bc_train)
<<<<<<< HEAD
=======

        mean_feq, std_feq, feq_norm = self.f_normalization(self.feq_trainbc)
        mean_fneq, std_fneq, fneq_norm = self.f_normalization(self.fneq_trainbc)
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
        
        
        loss_f_neq=0
        for i in range(feq_out.shape[1]):
<<<<<<< HEAD
            loss_f_neq+=self.loss_function(fneq_out[:,i],self.k*self.fneq_trainbc[:,i])
=======
            loss_f_neq+=self.loss_function(fneq_out[:,i],self.k*fneq_norm[:,i])
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
        

        loss_f_eq=0
        for i in range(feq_out.shape[1]):
<<<<<<< HEAD
            loss_f_eq+=self.loss_function(feq_out[:,i],self.feq_trainbc[:,i])

        f_pred = feq_out+fneq_out/self.k
=======
            loss_f_eq+=self.loss_function(feq_out[:,i],feq_norm[:,i])

        feq_out_denorm = std_feq*feq_out + mean_feq
        fneq_out_denorm = std_fneq*fneq_out + mean_fneq

        f_pred = feq_out_denorm + fneq_out_denorm/self.k
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
        #print('f_loss',lossfBCeq+lossfBCneq)
        row_pred,u_pred,xi_f = self.pred_variables(f_pred)
     
        RT = 100

        row = torch.div(self.p_trainbc, RT)
        size = self.p_trainbc.size()[0]
        #size.append(1)
        row1 = torch.reshape(row,(size,1))        
        
        row_u_train = torch.mul(row1,self.u_trainbc)
        #print(row_u_train.shape)
        lossfrow = self.loss_function(row_pred,row)
        #print('row',lossfrow)
        lossfxi = self.loss_function(xi_f,row_u_train)
        #print('xi',lossfxi)
        lossfBC  = loss_f_eq+loss_f_neq + lossfrow + lossfxi

        lossmBC = self.loss_function(u_pred,self.u_trainbc)
        #print('m',lossmBC)
        loss = lossfBC + lossmBC
<<<<<<< HEAD
        print('bc',loss)
=======

        #print('bc',loss)

        self.loss_bc.append(loss)

>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
        return loss
    

    def loss(self):
        lossEq = self.loss_residual()
        lossIC = self.lossIC()
        lossBC = self.lossBC()
        
        return lossEq + lossIC +lossBC

<<<<<<< HEAD
    def closure(self, optimizer,model, X_in_test,f ):
=======
    def closure(self, optimizer,model, X_in_test,f, f_eq, f_neq):
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
        
        optimizer.zero_grad()
        loss = self.loss()

        

        loss.backward()
        self.iter += 1

        if self.iter % self.iter_divider == 0:
            self.trainingloss.append(loss)
            with torch.no_grad():
<<<<<<< HEAD
                error_vec, _ ,_= model.test(X_in_test,f)
            self.error_f.append(error_vec)
            print(loss,error_vec,self.residual_data[1])
        
        return loss   

    def test(self,X_test_tensor,f):       
=======
                error_vec, _ ,_= model.test(X_in_test,f,f_eq,f_neq)
            self.error_f.append(error_vec)
            print(loss,error_vec)
            print(self.residual_data[1])
            #print('residual',self.loss_res)
            #print('ic',self.loss_ic)
            #print('bc',self.loss_bc)
        return loss   

    def test(self,X_test_tensor,f, f_eq, f_neq):       
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c

       
        fneq_out = self.modelneq.forward(X_test_tensor)
        feq_out = self.modeleq.forward(X_test_tensor)
<<<<<<< HEAD
        f_pred = feq_out+fneq_out     
=======

        mean_feq, std_feq, feq_norm = self.f_normalization(f_eq)
        mean_fneq, std_fneq, fneq_norm = self.f_normalization(f_neq)

        feq_out_denorm = std_feq*feq_out + mean_feq
        fneq_out_denorm = std_fneq*fneq_out + mean_fneq

        f_pred = feq_out_denorm+fneq_out_denorm/self.k     
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
        
        error_vec = torch.linalg.norm((f - f_pred),2)/torch.linalg.norm(f,2)  # Relative L2 Norm of the error (vector)

        f_pred = f_pred.cpu().detach().numpy()
        #feq_out = feq_out.cpu().detach().numpy()
        
        return error_vec, f_pred, feq_out

   
######################################################################################################################################################################################
def training_data(X_in,device):
    #print('creating traning data')
    N_u = int(0.8*len(X_in))
    
    var = variables(X_in)
    var = np.array(var)
    u_star = var[:,0:2]
    p_star = var[:,2]
    f_exact = feq_data(var)
    f_eq = np.array(f_exact)   
    f_neq = fneq_exact(X_in,device)
    f_neq1 = fneq_exact(X_in,device).cpu().detach().numpy()
    f_star = f_eq+f_neq1
    #print(X_in.size)
    idx0 = np.random.choice(X_in.shape[0], N_u, replace = False)
    X_in_train = X_in[idx0,:]
    #u_train = np.array([var[i] for i in idx0])
  
    feq_train = np.array([f_eq[i] for i in idx0])   
    fneq_train = np.array([f_neq1[i] for i in idx0])
    u_train = np.array([u_star[i] for i in idx0]) 
    p_train = np.array([p_star[i] for i in idx0]) 

    f_eq  = torch.from_numpy(f_eq).float().to(device)
    X_in_train = torch.from_numpy(X_in_train).float().to(device)    
    X_test_tensor = torch.from_numpy(X_in).float().to(device)
    u = torch.from_numpy(u_star).float().to(device)
    u_train = torch.from_numpy(u_train).float().to(device)
    p_train = torch.from_numpy(p_train).float().to(device)
    f = torch.from_numpy(f_star).float().to(device)
    feq_train = torch.from_numpy(feq_train).float().to(device)
    fneq_train = torch.from_numpy(fneq_train).float().to(device)

    return X_in_train,f_eq, f_neq, X_test_tensor, u,u_train,p_train, f, feq_train, fneq_train

  
def variables(X_in):
    var=[]       
    #X_in = X_in.cpu().detach().numpy()
    Re = 10
    u0 = 0.1581
    p0 = 0.05
    RT = 100
    C = RT
    L = 1
    pi = math.pi
    l = Re/2 - sqrt(((Re^2)/4)+4*(pi*pi))
    for cor in X_in:
        uj = u0*(1-exp(l*cor[0])*cos(2*pi*cor[1]))
        vj = u0*(l/(2*pi))*exp(l*cor[0])*sin(2*pi*cor[1])
        pj = p0*(1-(0.5*exp(2*l*cor[0]))) + C
        v = [uj,vj,pj]
        #print('u',uj,'v',vj,'p',pj)
            
        var.append(v)        
    return var

def velocity(x,y):
    Re = 10
    u0 = 0.1581
    p0 = 0.05
    RT = 100
    C = RT
    L = 1
    pi = math.pi
    l = Re/2 - sqrt(((Re^2)/4)+4*(pi*pi))
        
        
    u = u0*(1- torch.exp(l*x))*torch.cos(2*pi*y)
    v = u0*(l/(2*pi))*torch.exp(l*x)*torch.sin(2*pi*y)
    p_ = 0.5*torch.exp(2*l*x)
    c = torch.tensor(100,device=device)
    c = c.repeat(p_.shape)
    p = p0*(torch.ones(p_.shape,device=device)-p_) + c
    vel = torch.concat((u,v), dim=1)
    del u,v,p_,c
    torch.cuda.empty_cache()
    return vel,p


def xi_values():
    RT = 100
    c = sqrt(3*RT)
    e0 = [0,0]
    e1 = [1,0]
    e2 = [0,1]
    e3 = [-1,0]
    e4 = [0,-1]
    e5 = [1,1]
    e6 = [-1,1]
    e7 = [-1,-1]
    e8 = [1,-1]

    e = [e0,e1,e2,e3,e4,e5,e6,e7,e8]

    xi = []

    for ele in e:
        xi.append([x * c for x in ele])

    return xi

def weight_factor():
    w = {}
    xi = xi_values()
    for i in range(len(xi)):
        if i == 0:
            w[i] = 4/9
        elif i in [1,2,3,4]:
            w[i] = 1/9
        else:
            w[i] = 1/36

    return w
    
def dot_product_lists(l1,l2):
    a =  sum([x*y for x,y in zip(l1,l2)])
    return a

def feq_value(v):
    u = [v[0],v[1]]
    p = v[2]
    xi = xi_values()   
    w = weight_factor()
    RT=100
    rho=p/RT
    #print(len(xi))
    feq = []
    for i in range(len(xi)):
        feq1 = dot_product_lists(xi[i],u)/RT
        feq2 = ((dot_product_lists(xi[i],u))/(2*RT))**2
        feq3 = dot_product_lists(u,u)/(2*RT)
        feqi = w[i]*rho*(1+feq1+feq2+feq3)
        feq.append(feqi)
        
    return feq

def feq_tensor_grad(x,y):
    
    xi = xi_values()   
    xi = np.array(xi)
    xi = torch.from_numpy(xi).float().to(device)
    w = weight_factor()
    w = np.array(list(w.values()))
    w = torch.from_numpy(w).float().to(device)
    w = torch.reshape(w, (9,1))
    u,p = velocity(x,y)
    RT=100
    rho=p/RT
   
    feq1 = (torch.matmul(xi,u.T))/RT
    feq2 = (torch.matmul(xi,u.T)/(2*RT))**2
    feq3 = torch.square(u.T)/(2*RT)
        
    feq3 = torch.sum(feq3,dim=0)
    feq3 = feq3.repeat(9,1)
        
    feqp = torch.ones(feq3.shape,device=device)+feq1+feq2+feq3   
    feq = w*feqp
    feq = rho*feq.T
    torch.cuda.empty_cache()
    return feq


def feq_data(var):    
    
    f =[]
    for v in var:
        fi = feq_value(v)
        f.append(fi)

    return f
    

def fneq_exact(X_in,device):
    xi = xi_values()
    xi = np.array(xi)
       
    xi0 = torch.from_numpy(xi[:,0]).float().to(device)
    xi1 = torch.from_numpy(xi[:,1]).float().to(device)
    xi0 = torch.reshape(xi0,(9,1))
    xi1 = torch.reshape(xi0,(9,1))
        
    tau = 1.58e-4
            
    X_in = torch.from_numpy(X_in).float().to(device)
 
    X =torch.split(X_in,1,dim=1)
    x= X[0]
    y= X[1]   

    g=(x,y)
     
    f_x_y = torch.autograd.functional.jacobian(feq_tensor_grad,g)
   
    shape = list(f_x_y[0].size())
    shape.pop()
    
    f_x = torch.reshape(f_x_y[0],tuple(shape))
    f_y = torch.reshape(f_x_y[1],tuple(shape))
    f_x = torch.sum(f_x,2)
    f_y = torch.sum(f_y,2)
   
    Fneq_x= torch.mul(xi0,f_x.T)
    
    Fneq_y= torch.mul(xi1,f_y.T)

    fneq_exact = -tau*(Fneq_x+Fneq_y)
    torch.cuda.empty_cache()   

    return fneq_exact.T  

def total_data_extraction(X_in,device):
    X_in_train,f_eq, f_neq, X_test_tensor, u,u_train,p_train, f, feq_train, fneq_train = training_data(X_in,device)

<<<<<<< HEAD
    data = [X_in_train,f_eq, f_neq, X_test_tensor, u, f, feq_train, fneq_train]
=======
    data = [X_in_train,f_eq, f_neq, X_test_tensor, u, u_train, f, feq_train, fneq_train]
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
    f = open('total_data.pkl', 'wb')
    pickle.dump(data, f)
    f.close()
    print(X_in_train)

<<<<<<< HEAD
=======
######################################################################################################################################################################################

>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
def main_loop(layers,X_in,X_initial,X_bc,epochs,device,x,y,user,neq_scale):
     
    lb_X = X_in.min(0)
    ub_X = X_in.max(0)  
    
    #total_data_extraction(X_in,device)
    #for total points
    file = open('total_data.pkl', 'rb')
    data = pickle.load(file)
    #print(type(data))
    X_in_train = data[0].to(device)
<<<<<<< HEAD
    X_test_tensor = data[3]
    u = data[4].to(device)
    f = data[5].to(device)
    f_train = (data[6]+data[7]).to(device)
=======
    f_eq = data[1]
    f_neq = data[2]
    X_test_tensor = data[3]
    u = data[4].to(device)
    f = data[6].to(device)
    f_train = (data[7]+data[8]).to(device)
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
    #f_eq = data[1]
    #f_neq = data[2]

    k=neq_scale
    model = Flow_PINN(layers,X_in,X_initial,X_bc,ub_X,lb_X,device,k)   
    print('models created')   
       
  
    model.to(device)
    

    params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=0.005)

    optimizer_b = torch.optim.LBFGS(model.parameters(), lr=0.1,
                              max_iter = epochs,
                              max_eval = None,
                              tolerance_grad = 1e-05,
                              tolerance_change=1e-09,
                              history_size=100,
                              line_search_fn='strong_wolfe')

    start_time = time.time()

    for epoch in range(0,epochs):

        print('epoch',epoch)
<<<<<<< HEAD
        optimizer.step(lambda: model.closure(optimizer,model,X_test_tensor,f))
=======
        optimizer.step(lambda: model.closure(optimizer,model,X_test_tensor,f,f_eq,f_neq))
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c

    #optimizer_b.step(lambda: model.closure(optimizer,model,X_test_tensor,f))

    elapsed = time.time() - start_time

    print('Training time: %.2f' % (elapsed))
    
    error,f_pred,f_eq_out = model.test(X_test_tensor,f)
    
    print(f_pred[0])
    print(f.cpu().detach().numpy()[0])
    print('Error: %.5f' % (error))

    f_pred = torch.from_numpy(f_pred).float().to(device)
    _,u_pred,_ = model.pred_variables(f_pred)
<<<<<<< HEAD
    _,residual_data = model.residual(X_test_tensor,f_eq_out)
     #plotting
    
    

    result = [f_pred,f_eq_out,f,u_pred,residual_data,u,model.trainingloss,model.error_f]
    f = open('result.pkl', 'wb')
    pickle.dump(result, f)
    f.close()
  
=======
    residual_data = model.residual(X_test_tensor,f_eq_out)
       

    result = [f_pred,f_eq_out,f,u_pred,residual_data,u,model.trainingloss,model.error_f, model.loss_res, model.loss_ic, model.loss_bc]
    f = open('result.pkl', 'wb')
    pickle.dump(result, f)
    f.close()

def final_PINN(layers,X_in,X_initial,X_bc,epochs,device,x,y,user,neq_scale):
    #total_data_extraction(X_in,device)
    main_loop(layers,X_in,X_initial,X_bc,epochs,device,x,y,user,neq_scale)
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c
   
    
    

#############################################################################################################################################################

x_values = np.arange(-0.5,2.0,0.0147).tolist()
y_values = np.arange(-0.5,1.5,0.0200).tolist()
t_values = np.arange(0.0,2.0,0.0118).tolist()
x_values.append(2.000)
y_values.append(1.500)
#x1,t = np.meshgrid(x_values,t_values)
y,x= np.meshgrid(y_values,x_values)
#t = np.resize(t,x.shape)
#coordinates = np.column_stack((x.ravel(),y.ravel()))  

X_in = np.hstack([x.flatten()[:,None],y.flatten()[:,None]])


#print(y)
X_initial = np.hstack((x[0,:][:,None],y[0,:][:,None]))
#print(X_initial)
X_bc_lower = np.hstack((x[:,0][:,None],y[:,0][:,None]))

X_bc_upper = np.hstack((x[:,0][:,None],y[:,-1][:,None]))

X_bc = np.vstack([X_initial,X_bc_lower,X_bc_upper])

layers = np.array([2,80,80,80,80,80,80,80,80,9])


#device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')  

user='rprpu'

<<<<<<< HEAD
epochs=20000

neq_scale=1
main_loop(layers,X_in,X_initial,X_bc,epochs,device,x,y,user,neq_scale)
=======
epochs=15

neq_scale=1
final_PINN(layers,X_in,X_initial,X_bc,epochs,device,x,y,user,neq_scale)
>>>>>>> 9c150e5ac03cf7481d2b3b3dd99f469ccc28c54c




