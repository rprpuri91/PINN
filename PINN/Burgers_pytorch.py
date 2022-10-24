
import os
from typing import ForwardRef
import torch
from torch import nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
from torch import optim
from torch import tensor
import numpy as np
import scipy.io 
from pyDOE import lhs
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker

torch.set_default_dtype(torch.float)

torch.manual_seed(1234)

np.random.seed(1234)


class PiNN(nn.Module):

    def __init__(self,layers, lb, ub):
        super().__init__() 

        self.lb = lb
        self.ub = ub

        self.layers = layers

        

        #session

        #variables
        #self.lambda_1 = autograd.Variable([0.0])
        #self.lambda_2 = autograd.Variable([-6.0])

        #self.f_pred = self.net_f()


        #Activation
        self.activation = nn.Tanh()

        #Loss
        self.loss_function = nn.MSELoss(reduction ='mean')

        #layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        

        #self.fc1 = nn.Linear(2,40)
        #self.fc2 = nn.Linear(40,40)
        #self.fc3 = nn.Linear(40,40)
        #self.fc3 = nn.Linear(40,40)
        #self.fc3 = nn.Linear(40,40)
        #self.fc3 = nn.Linear(40,40)
        #self.fc3 = nn.Linear(40,40)
        #self.fc4 = nn.Linear(40,1)


        self.iter = 0
        #xavier_normalization
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain =5/3)

            nn.init.zeros_(self.linears[i].bias.data)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

    def forward(self,x):

        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        
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
        for i in range(len(layers)-2):
        
            
             z = self.linears[i](a)
                        
             a = self.activation(z)
            
        a = self.linears[-1](a)
        
        return a      
         
             

    def net_f(self, x_to_train_f):
        
        nu = 0.01/np.pi

        g = x_to_train_f.clone()

        g.requires_grad = True

        u = self.forward(g)

        u_x_t = autograd.grad(u,g,torch.ones([x_to_train_f.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
                                
        u_xx_tt = autograd.grad(u_x_t,g,torch.ones(x_to_train_f.shape).to(self.device), create_graph=True)[0]

        u_x = u_x_t[:,[0]]

        u_t = u_x_t[:,[1]]

        u_xx = u_xx_tt[:,[0]]

        f = u_t + u*u_x - nu*u_xx

        return f

    def loss_BC(self,x,y):

        loss_u = self.loss_function(self.forward(x),y)

        return loss_u

    def loss_PDE(self, f, f_hat):                   

        loss_f = self.loss_function(f,f_hat)

        return loss_f

    def loss(self, x, y, x_to_train_f, f_hat):

        loss_u = self.loss_BC(x,y)
        f = self.net_f(x_to_train_f)
        loss_f = self.loss_PDE(f, f_hat)

        loss_val = loss_u + loss_f

        return loss_val

    

    def closure(self,model,X_u_train, u_train, X_f_train, optimizer, f_hat, X_u_test_tensor, u):

        optimizer.zero_grad()

        loss = self.loss(X_u_train, u_train, X_f_train, f_hat)

        loss.backward()
        self.iter += 1

        if self.iter % 100 == 0:

            error_vec, _ = model.test(X_u_test_tensor,u)
        
            print(loss,error_vec)
        
        return loss    

    
    def test(self, X_u_test_tensor, u):
        u_pred = self.forward(X_u_test_tensor)
        
        error_vec = torch.linalg.norm((u-u_pred),2)/torch.linalg.norm(u,2)  # Relative L2 Norm of the error (vector)

        u_pred = u_pred.cpu().detach().numpy()
        
        u_pred = np.reshape(u_pred, (256,100), order = 'F')

        return error_vec, u_pred


    def main_loop(N_u, N_f, noise, layers, user):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #device = 'cpu'
        print(f'Using {device} device')

        user_name = user
        file_name = 'burgers_shock.mat'
        path = 'C:/Users/'+user_name+'/Desktop/'+file_name
        print(path)
        data = scipy.io.loadmat(path)
        x=data['x'].flatten()[:,None]
        t=data['t'].flatten()[:,None]

        
        usol = data['usol']
        Exact = np.real(usol).T                        

        X,T =np.meshgrid(x,t)

        #print(X)
        #print(T)
        #print(Exact)

        #Test Data
          
        leftedge_x = np.hstack((X[0,:][:,None], T[0,:][:,None])) #L1
        leftedge_u = usol[:,0][:,None]

        
        bottomedge_x = np.hstack((X[:,0][:,None], T[:,0][:,None])) #L2
        bottomedge_u = usol[-1,:][:,None]

        
        topedge_x = np.hstack((X[:,-1][:,None], T[:,0][:,None])) #L3
        topedge_u = usol[0,:][:,None]

        all_X_u_train = np.vstack([leftedge_x, bottomedge_x, topedge_x]) 
        all_u_train = np.vstack([leftedge_u, bottomedge_u, topedge_u])   

        X_star =np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        print(np.shape(X_star))
        u_star =Exact.flatten()[:,None]

        
        #domain_bounds
        lb = X_star.min(0)
        ub = X_star.max(0)

        idx = np.random.choice(all_X_u_train.shape[0], N_u, replace = False)
        X_u_train1 = all_X_u_train[idx,:]
        u_train = all_u_train[idx,:]       

        
        X_f_train1 = lb + (ub-lb)*lhs(2,N_f)
        X_f_train1 = np.vstack((X_f_train1, X_u_train1))

        u_train2 = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])

        X_f_train = torch.from_numpy(X_f_train1).float().to(device)
        X_u_train = torch.from_numpy(X_u_train1).float().to(device)
        u_train = torch.from_numpy(u_train2).float().to(device)
        X_u_test_tensor = torch.from_numpy(X_star).float().to(device)
        u = torch.from_numpy(u_star).float().to(device)
        f_hat = torch.zeros(X_f_train.shape[0],1).to(device)

        

        model = PiNN(layers, lb, ub)

        

        model.to(device)

        print(model)

        params = list(model.parameters())

        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1,
                              max_iter = 15000,
                              max_eval = None,
                              tolerance_grad = 1e-05,
                              tolerance_change=1e-09,
                              history_size=100,
                              line_search_fn='strong_wolfe')
        

        start_time = time.time()

        model.train()          

        optimizer.step(lambda: model.closure(model,X_u_train, u_train, X_f_train, optimizer, f_hat, X_u_test_tensor, u))

        elapsed = time.time() - start_time
        
        print('Training time: %.2f' % (elapsed))

        f_pred = model.net_f(X_f_train)

        error, u_pred = model.test(X_u_test_tensor,u)

        print('Error: %.5f' % (error))

        #plotting

        X_u_train = X_u_train.cpu()
    
        fig, ax = plt.subplots()
        ax.axis('off')

        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
        ax = plt.subplot(gs0[:, :])

        h = ax.imshow(u_pred, interpolation='nearest', cmap='rainbow', 
                    extent=[T.min(), T.max(), X.min(), X.max()], 
                    origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

        line = np.linspace(x.min(), x.max(), 2)[:,None]
        ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
        ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
        ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.legend(frameon=False, loc = 'best')
        ax.set_title('$u(t,x)$', fontsize = 10)
    
        ''' 
        Slices of the solution at points t = 0.25, t = 0.50 and t = 0.75
        '''
    
        ####### Row 1: u(t,x) slices ##################
        gs1 = gridspec.GridSpec(1, 3)
        gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

        ax = plt.subplot(gs1[0, 0])
        ax.plot(x,usol.T[25,:], 'b-', linewidth = 2, label = 'Exact')       
        ax.plot(x,u_pred.T[25,:], 'r--', linewidth = 2, label = 'Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_title('$t = 0.25s$', fontsize = 10)
        ax.axis('square')
        ax.set_xlim([-1.1,1.1])
        ax.set_ylim([-1.1,1.1])

        ax = plt.subplot(gs1[0, 1])
        ax.plot(x,usol.T[50,:], 'b-', linewidth = 2, label = 'Exact')       
        ax.plot(x,u_pred.T[50,:], 'r--', linewidth = 2, label = 'Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.axis('square')
        ax.set_xlim([-1.1,1.1])
        ax.set_ylim([-1.1,1.1])
        ax.set_title('$t = 0.50s$', fontsize = 10)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

        ax = plt.subplot(gs1[0, 2])
        ax.plot(x,usol.T[75,:], 'b-', linewidth = 2, label = 'Exact')       
        ax.plot(x,u_pred.T[75,:], 'r--', linewidth = 2, label = 'Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.axis('square')
        ax.set_xlim([-1.1,1.1])
        ax.set_ylim([-1.1,1.1])    
        ax.set_title('$t = 0.75s$', fontsize = 10)
    
        plt.savefig('C:/Users/'+user_name+'/Desktop/ThesisBurgers.png',dpi = 500)
        plt.show()
        

#layers = np.array([2,20,20,1])
layers = np.array([2,40,40,40,40,40,40,40,40,1])
N_u = 100
N_f = 10000
noise = 0.1

user = 'rprpu'

PiNN.main_loop(N_u, N_f, noise, layers, user)


