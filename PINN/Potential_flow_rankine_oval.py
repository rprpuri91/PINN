
import torch
from torch import nn
from math import cos, exp, sin, sqrt
import math
import numpy as np
import time
import pickle
import os
import matplotlib.pyplot as plt
import scipy as sc

#from torch_geometric.data import data

torch.manual_seed(1234)

np.random.seed(1234)

cwd = os.getcwd()

user = 'rishabh.puri'

#os.chdir()

# Print the current working directory
print("Current working directory: {0}".format(cwd))
class Potential_flow_preprocessing():
    def __init__(self, R, U0,m,h,device,a):

        self.V_min = 0
        self.V_max = 0
        self.R =R
        self.U0 = U0
        self.m = m
        self.h = h
        self.a =a
        self.device = device
   

 
    def stream_fn_rankine_oval(self,x,y):


        t = (2*self.a*y)/torch.sub((torch.square(x) + torch.square(y)), (self.a*self.a))
        m0 = self.m/(2*math.pi)

        psi = - self.U0*y + m0*torch.atan(t)

        return psi


    def potential_fn_rankine_oval(self,x,y):

        x1 = torch.square(x + self.a) + torch.square(y)

        x2 = torch.square(x - self.a) + torch.square(y)

        x3 = torch.divide(x1,x2)

        m0 = self.m/(4*math.pi)

        phi = self.U0*x + m0*torch.log(x3)

        return phi

    def velocity_rankine_oval(self,X):
      
        x = X[:,0]
        y = X[:,1]

        m0 = self.m/(2*math.pi)
        a = self.a

        V = []

        for i in range(len(X)):

            #print(x[i],y[i])

            x1 = (x[i]+a)**2 + (y[i])**2
            x2 = (x[i]-a)**2 + (y[i])**2

            #u = self.U0 + m0*(((x[i]+a)/x1) + ((x[i]-a)/x2))

            #v = m0*y[i]*((1/x1) - (1/x2))

            u = U0 +(self.m/(2*np.pi))*(((x[i]+self.a)/((x[i]+self.a)*(x[i]+self.a)+y[i]*y[i]))-((x[i]-self.a)/((x[i]-self.a)*(x[i]-self.a)+(y[i]*y[i]))))
            v = (self.m/(2*np.pi))*y[i]*((1/((x[i]+self.a)*(x[i]+self.a)+y[i]*y[i]))-(1/((x[i]-self.a)*(x[i]-self.a)+(y[i]*y[i]))))

            Vi = [u,v]

            #print(Vi)
            V.append(Vi)


        V_xy = np.array(V)

        return V_xy
    

    def velocity_dataset(self):

      
        X_domain, indices_domain, X_boundary = mesh_rankine_oval(self.m)
        
        
        V_domain = self.velocity_rankine_oval(X_domain)

        V_inlet = self.velocity_rankine_oval(X_initial)

        V_outlet = self.velocity_rankine_oval(X_outlet)

        V_wall = self.velocity_rankine_oval(X_wall)

        N_f = int(0.3* len(V_domain))

        idx = np.random.choice(V_domain.shape[0], N_f, replace=False)
        V_domain2 = V_domain[idx, :]
        V_boundary = self.velocity_rankine_oval(X_boundary)

        V_domain = torch.from_numpy(V_domain).float().to(self.device)
        V_inlet = torch.from_numpy(V_inlet).float().to(self.device)
        V_outlet = torch.from_numpy(V_outlet).float().to(self.device)
        V_wall = torch.from_numpy(V_wall).float().to(self.device)
        V_domain2 = torch.from_numpy(V_domain2).float().to(self.device)
        V_boundary = torch.from_numpy(V_boundary).float().to(self.device)
        V_max = V_domain.max()
        V_min = V_domain.min()
            
        
        velocity = [V_domain, V_inlet, V_outlet, V_wall, V_domain2,V_boundary,V_max, V_min]

        #velocity = [V_domain,V_max,V_min]

        return velocity


    
    def velocity_cartesian_vjp(self, X):
        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X).to(self.device)
            # print(X.size())

        X = torch.split(X, 1, dim=1)

        x = X[0]
        y = X[1]

        # print(x.size())

        g = (x, y)

        v = torch.ones_like(x, device=self.device)

        phi, phi_x_y = torch.autograd.functional.vjp(self.stream_fn_rankine_oval, g, v, create_graph=False)

        V_x = phi_x_y[1]
        V_y = -phi_x_y[0]

        # print(V_y.size())

        V = torch.concat((V_x, V_y), dim=1)

        return V


    def normalize_velocity(self, V,V_max,V_min):
        V_norm = (V - V_min) / (V_max - V_min)

        return V_norm

class Rankine_oval_PINN(nn.Module):
    def __init__(self, layers,nu,X,V, norm,device):
        super().__init__()

        # Activation
        self.activation = nn.Tanh()
        self.activation2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        # loss function
        self.loss_function = nn.MSELoss(reduction='mean')

        self.layers = layers

        # layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        self.nu = nu

        self.norm = norm

        self.device = device

        self.X_domain = X[0]

        self.X_boundary = X[1]

        #print("X",type(X))

        N_f = int(0.3* len(self.X_domain))

        idx = np.random.choice(self.X_domain.shape[0], N_f, replace=False)
        self.X_domain2 = X_domain[idx, :]
        
        self.V_domain2 = V[4]
        self.V_domain  = V[0]

        self.V_boundary = V[5]
        
        self.V_inlet = V[1]

        self.V_outlet = V[2]

        self.V_wall = V[3]

        self.V_min = V[6]
        self.V_max = V[7]

        self.iter = 0

        self.divider = 5

        self.training_loss = []
        self.error = []
    
    def forward(self, X):

        if torch.is_tensor(X) != True:
            x = torch.from_numpy(X)
        X = X.to(self.device)
        # u_b = torch.from_numpy(self.ub).float().to(self.device)
        # l_b = torch.from_numpy(self.lb).float().to(self.device)

        # preprocessing input
        # x = (x - l_b)/(u_b - l_b) #feature scaling

        x = self.scaling(X)
        # convert to float
        a = x.float()

        '''     
            Alternatively:
        
            a = self.activation(self.fc1(a))
            a = self.activation(self.fc2(a))
            a = self.activation(self.fc3(a))
            a = self.fc4(a)
        
            '''
        for i in range(len(self.layers) - 3):
            z = self.linears[i](a)

            a = self.activation2(z)

        a = self.activation(a)

        a = self.linears[-1](a)

        return a  

    def scaling(self, X):

        mean, std, var = torch.mean(X), torch.std(X), torch.var(X)
        # preprocessing input
        x = (X - mean) / (std)  # feature scaling

        return x

    def normalize_velocity(self, V):
        V_norm = (V - self.V_min) / (self.V_max - self.V_min)

        return V_norm

    def denormalize_velocity(self,V_norm):
        V = V_norm*(self.V_max - self.V_min) + self.V_min

        return V

    def train_test_data(self,X, V):
        
        N_u = int(self.nu * len(X))

        idx = np.random.choice(X.shape[0], N_u, replace=False)

        
        X_star = X[idx, :]
        V_train = V[idx, :].float()

        X_test = np.delete(X, idx, axis = 0)
        idxtest = []

        for i in range(0, X.shape[0]):
            if i in idx:
                continue
            else:
                idxtest.append(i)
        
        V_test = V[idxtest, :].float()

        X_train = torch.from_numpy(X_star).float().to(self.device)
        X_test = torch.from_numpy(X_test).float().to(self.device)

        if self.norm is True:
            V_train = self.normalize_velocity(V_train)
            V_test = self.normalize_velocity(V_test)

        return V_train, X_train, V_test, X_test

    def velocity_pred(self, x, y):

        g = torch.cat((x, y), dim=1)

        velocity = self.forward(g)

        return velocity

    def laplace_potential(self, X):

        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X).to(self.device)

        v1 = torch.zeros_like(X, device=self.device)
        v2 = torch.zeros_like(X, device=self.device)

        v1[:, 0] = 1
        v2[:, 1] = 1

        X = torch.split(X, 1, dim=1)

        x = X[0]
        y = X[1]

        g = (x, y)

        V, U_x_y = torch.autograd.functional.vjp(self.velocity_pred, (x, y), v1, create_graph=True)

        V1, V_x_y = torch.autograd.functional.vjp(self.velocity_pred, (x, y), v2, create_graph=True)

        u_x = U_x_y[0]

        u_y = U_x_y[1]

        v_y = V_x_y[1]

        v_x = V_x_y[0]

        # print("Velocity1",V)
        # print("Velocity2",V1)

        laplace_phi = u_x + v_y

        vorticity = v_x - u_y

        # print(laplace_phi)

        return laplace_phi, vorticity
        

    def loss(self,X,V):

        V_train, X_train, V_test, X_test = self.train_test_data(X,V)

        V_pred = self.forward(X_train)

        laplace_phi, vorticity = self.laplace_potential(X_train)
        
        target = torch.zeros_like(laplace_phi, device = self.device)

        laplace_loss = self.loss_function(laplace_phi,target)

        vorticity_loss = self.loss_function(vorticity,target)

        velocity_loss = self.loss_function(V_pred,V_train)

        loss = laplace_loss + velocity_loss + vorticity_loss

        return loss

    def total_loss(self):

        loss_domain = self.loss(self.X_domain2,self.V_domain2)

        loss_inlet = self.loss(X_initial, self.V_inlet)

        loss_outlet = self.loss(X_outlet, self.V_outlet)

        loss_wall = self.loss(X_wall, self.V_wall)

        loss_boundary = self.loss(self.X_boundary, self.V_boundary)

        total_loss =  loss_domain + loss_inlet + loss_outlet + loss_wall + loss_boundary

        return total_loss

    def closure(self, optimizer, model):

        optimizer.zero_grad()

        loss = self.total_loss()
        
        loss.backward()

        print("Epoch: ", self.iter)

        self.iter+=1

        V_train, X_train, V_test, X_test = self.train_test_data(self.X_domain,self.V_domain)

        if self.iter % self.divider == 0:
            self.training_loss.append(loss)
            with torch.no_grad():
                error_vec, _ = model.test(model, X_test, V_test)
            self.error.append(error_vec)
            print(loss, error_vec)

        return loss

    def test(self, model,X_test, V_test):

        V_pred = model.forward(X_test)

        error_vec = torch.linalg.norm((V_test - V_pred), 2) / torch.linalg.norm(V_test,2)  # Relative L2 Norm of the error (vector)

        #V_pred = V_pred.cpu().detach().numpy()

        return error_vec, V_pred



h =0.05

x_values = np.arange(-8.0, 8.0, h).tolist()
y_values = np.arange(-4.0, 4.0, h).tolist()
x_values.append(8.0)
y_values.append(4.0)

x, y = np.meshgrid(x_values, y_values)

U0 = 10.0

R = 2.0

X_in = np.hstack([x.flatten()[:, None], y.flatten()[:, None]])

X_initial = np.hstack((x[0, :][:, None], y[0, :][:, None]))

X_bc_lower = np.hstack((x[:, 0][:, None], y[:, 0][:, None]))

X_bc_upper = np.hstack((x[:, 0][:, None], y[:, -1][:, None]))

X_wall = np.vstack([X_bc_upper, X_bc_lower])

X_outlet = np.hstack((x[-1, :][:, None], y[0, :][:, None]))

m  = 40

a = 3.0

b = 0.5

def mesh_rankine_oval(m):

   
    indices_domain = []
    indices_boundary = []
    
    x = X_in[:,0]
    y = X_in[:,1]

    for i in range(len(X_in)):
        t = (2 * a * y[i]) / ((x[i]) * (x[i]) + (y[i]) * (y[i]) - (a * a))
        m0 = m / (2 * math.pi)

        psi = - U0 * y[i] + m0 * math.atan(t)

        if(psi < 0.05) and (psi > -0.05):
            indices_boundary.append(i)

    for i in range(len(X_in)):
        
        if (x[i]<(a+b) and x[i]>(a-b)) and (y[i]<0.5 and y[i]>-0.5):
            print(X_in[i])
            indices_domain.append(i)
        elif (x[i]<-(a-b) and x[i]>-(a+b)) and (y[i]<0.5 and y[i]>-0.5):
            print(X_in[i])
            indices_domain.append(i)


    X_boundary = np.take(X_in, indices_boundary, axis=0)
   
    X_domain = np.delete(X_in, indices_domain, axis=0)
    
    print("domain",X_domain)

    return  X_domain, indices_domain, X_boundary

    

#def dataset_rankine_oval():
    
    
#    mesh_dataset = {}

    
#    for a in A:
#        #print(a)
#        X_domain, X_b, indices = mesh_rankine_oval(m,a)

#        dataset = [X_domain, X_b, indices,a]
        
#        mesh_dataset["rankine"+str(a)] = dataset


#    print(mesh_dataset.items())

#    return mesh_dataset





device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


preprocessing = Potential_flow_preprocessing(R, U0,m,h,device,a)

V = preprocessing.velocity_dataset()
V_domain = V[0]



layers = np.array([2, 60, 60, 60,60,60, 2])

nu = 0.8

epochs = 3000

X_domain, indices_domain, X_boundary = mesh_rankine_oval(m)

X = [X_domain, X_boundary, indices_domain,a]

norm = True

model = Rankine_oval_PINN(layers,nu,X,V, norm,device)

model.to(device)

start_time = time.time()

optimizerA = torch.optim.Adam(model.parameters(), lr=0.001)
optimizerB = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for e in range(0,epochs):
    optimizerA.step(lambda: model.closure(optimizerA, model))

elapsed = time.time() - start_time

print('Training time: %.2f' % (elapsed))


V_domain_norm = model.normalize_velocity(V_domain)
X_domain =  torch.from_numpy(X_domain).float().to(device)

error, V_pred_norm = model.test(model,X_domain, V_domain_norm) 
print(error)

V_pred = model.denormalize_velocity(V_pred_norm)

result = [V_pred,V_domain, V_pred_norm, V_domain_norm,indices_domain, model.error, model.training_loss]
f = open('result_rankine_oval_potential_flow.pkl', 'wb')
pickle.dump(result, f)
f.close()



