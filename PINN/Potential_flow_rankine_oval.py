
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
from sympy import *

#from torch_geometric.data import data

torch.manual_seed(1234)

np.random.seed(1234)

cwd = os.getcwd()

user = 'rishabh.puri'

#os.chdir()

# Print the current working directory
print("Current working directory: {0}".format(cwd))
class Potential_flow_preprocessing():
    def __init__(self, U0,m,h,device,a):

        self.V_min = 0
        self.V_max = 0
        #self.R =R
        self.U0 = U0
        self.m = m
        self.h = h
        self.a =a
        self.device = device
   

 
    def stream_fn_rankine_oval(self,x,y):


        t = (2*self.a*y)/torch.sub((torch.square(x) + torch.square(y)), (self.a*self.a))
        m0 = self.m/(2*pi)

        psi = - self.U0*y + m0*torch.atan(t)

        return psi


    def potential_fn_rankine_oval(self,x,y):

        x1 = torch.square(x + self.a) + torch.square(y)

        x2 = torch.square(x - self.a) + torch.square(y)

        x3 = torch.divide(x1,x2)

        m0 = float(self.m/(4*pi))

        #print(type(m0))

        phi = self.U0*x + torch.mul(torch.log(x3),m0)

        return phi

    def velocity_rankine_oval(self,X):
      
        x = X[:,0]
        y = X[:,1]

        m0 = self.m/(2*pi)
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
    

    def velocity_dataset(self,X_domain,X_domain2, indices_domain, X_boundary):

        #V_in = self.velocity_rankine_oval(X_in)

        V_domain = self.velocity_cartesian_vjp(X_domain)
        print(V_domain)
        V_inlet = self.velocity_cartesian_vjp(X_initial)

        V_outlet = self.velocity_cartesian_vjp(X_outlet)

        V_wall = self.velocity_cartesian_vjp(X_wall)

        V_domain2 = self.velocity_cartesian_vjp(X_domain2)
        V_boundary = self.velocity_cartesian_vjp(X_boundary)

        #V_in = torch.from_numpy(V_in).float().to(self.device)
        '''V_domain = torch.from_numpy(V_domain).float().to(self.device)
        V_inlet = torch.from_numpy(V_inlet).float().to(self.device)
        V_outlet = torch.from_numpy(V_outlet).float().to(self.device)
        V_wall = torch.from_numpy(V_wall).float().to(self.device)
        
        V_boundary = torch.from_numpy(V_boundary).float().to(self.device)'''
        #V_domain2 = torch.from_numpy(V_domain2).float().to(self.device)
        self.V_max = V_domain.max()
        self.V_min = V_domain.min()
            
        
        velocity = [V_domain, V_inlet, V_outlet, V_wall, V_domain2,V_boundary,self.V_max, self.V_min]

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

        phi, phi_x_y = torch.autograd.functional.vjp(self.potential_fn_rankine_oval, g, v, create_graph=False)

        V_x = phi_x_y[0]
        V_y = phi_x_y[1]

        # print(V_y.size())

        V = torch.concat((V_x, V_y), dim=1)

        return V

    def mesh_rankine_oval(self,m):

        indices_domain = []
        indices_boundary = []
        indices_inside_oval = []
        solution = []

        X = X_in[:, 0]
        Y = X_in[:, 1]

        X_list = X.tolist()
        Y_list = Y.tolist()

        for i in range(len(X_in)):
            print(i)
            if X[i] != a and Y[i] != 0:
                t = (2 * a * Y[i]) / ((X[i]) * (X[i]) + (Y[i]) * (Y[i]) - (a * a))
                m0 = m / (2 * pi)

                psi = - (U0 * Y[i]) + (m0 * atan(t))

                if (psi < 0.1) and (psi > -0.1):
                    indices_boundary.append(i)

        print('4')
        for i in range(len(X_in)):

            if (X[i] < (a + b) and X[i] > (a - b)) and (Y[i] < b and Y[i] > -b):
                # print(X_in[i])
                indices_domain.append(i)
            elif (X[i] < -(a - b) and X[i] > -(a + b)) and (Y[i] < b and Y[i] > -b):
                # print(X_in[i])
                indices_domain.append(i)

        X_boundary = np.take(X_in, indices_boundary, axis=0)

        X_domain_in = []

        print('2')
        y_i = 0
        x_i = 0

        for i in range(len(X_boundary) - 1):
            print(X_boundary[i])
            print(X_boundary[i + 1])
            print('old_y', y_i)

            if isclose(X_boundary[i][1], 0, abs_tol=0.001):
                nox1 = int(2 * x_i / h)
                # x1 = np.linspace(-x_i, x_i, nox1).tolist()

                x1 = np.arange(-x_i, x_i, h).tolist()
                x1 = [round(elem, dec) for elem in x1]
                # x2 = np.arange(x_i, 8.0, h).tolist()
                for j in x1:
                    X_domain_in.append([j, X_boundary[i][1]])
                '''for k in x2:
                    X_domain_in.append([k, X_boundary[i][1]])'''

            elif isclose(X_boundary[i][1], X_boundary[i + 1][1], abs_tol=0.01) and isclose(X_boundary[i][0],
                                                                                           -X_boundary[i + 1][0],
                                                                                           abs_tol=0.01):

                if isclose(X_boundary[i][1], (y_i + h), abs_tol=0.01) == False and y_i != 0:
                    h1 = X_boundary[i][1] - y_i
                    n = round(h1 / h)
                    print('difference_scale', n)
                    while n != 0:
                        y_i = round((y_i + h), dec)
                        if y_i < 0:
                            x_i = round((x_i + h), dec)
                        else:
                            x_i = round((x_i - h), dec)
                        x1 = np.arange(-x_i, x_i, h).tolist()
                        x1 = [round(elem, dec) for elem in x1]
                        print('new_y', y_i)
                        for j in x1:
                            X_domain_in.append([j, y_i])
                        '''for k in x2:
                            X_domain_in.append([k, y_i])'''
                        n -= 1
                else:
                    nox1 = int(2 * x_i / h)
                    # x1 = np.linspace(-x_i,x_i,nox1).tolist()
                    x_i = abs(X_boundary[i][0])
                    x1 = np.arange(-x_i, x_i, h).tolist()
                    x1 = [round(elem, dec) for elem in x1]
                    # x2 = np.arange(x_i, 8.0, h).tolist()
                    # print('x',x)
                    y_i = X_boundary[i][1]
                    print('new_y_', y_i)
                    for j in x1:
                        X_domain_in.append([j, X_boundary[i][1]])
                    '''for k in x2:
                        X_domain_in.append([k,X_boundary[i][1]])'''

        print('thelist', X_domain_in)

        X_domain_in_np = np.array(X_domain_in)
        X_in_list = X_in.tolist()

        # print(X_domain_in.round(2))

        for i in range(len(X_in)):
            print(i)
            if X_in[i].tolist() in X_domain_in:
                print('found')
                print(X_in[i])
                indices_inside_oval.append(i)

        print("inside", indices_inside_oval)

        X_domain = np.delete(X_in, indices_inside_oval, axis=0)

        fig, ax = plt.subplots(1,2)
        ax[0].scatter(X_domain_in_np[:,0],X_domain_in_np[:,1])
        ax[1].scatter(X_domain[:,0], X_domain[:,1])

        plt.show()

        return X_domain, indices_inside_oval, X_boundary


    def normalize_velocity(self,V):
        V_norm = (V - self.V_min) / (self.V_max - self.V_min)

        return V_norm

    def denormalize_velocity(self,V_norm):
        V = V_norm * (self.V_max - self.V_min) + self.V_min

        return V

    def train_test_data(self, X, V):

        N_u = int(nu * len(X))

        idx = np.random.choice(X.shape[0], N_u, replace=False)

        X_star = X[idx, :]
        V_train = V[idx, :].float()

        X_test = np.delete(X, idx, axis=0)
        idxtest = []

        for i in range(0, X.shape[0]):
            if i in idx:
                continue
            else:
                idxtest.append(i)

        V_test = V[idxtest, :].float()

        X_train = torch.from_numpy(X_star).float().to(self.device)
        X_test = torch.from_numpy(X_test).float().to(self.device)

        if norm is True:
            V_train = self.normalize_velocity(V_train)
            V_test = self.normalize_velocity(V_test)

        return V_train, X_train, V_test, X_test

    def data_generation(self):

        X_domain, indices_domain, X_boundary = self.mesh_rankine_oval(m)
        N_f = int(0.3 * len(X_domain))

        idx = np.random.choice(X_domain.shape[0], N_f, replace=False)
        X_domain2 = X_domain[idx, :]
        V = self.velocity_dataset(X_domain, X_domain2, indices_domain, X_boundary)

        # [V_domain, V_inlet, V_outlet, V_wall, V_domain2, V_boundary, V_max, V_min]

        h5 = h5py.File('data_rankine_oval_potential_flow.h5', 'w')
        g1 = h5.create_group('domain')
        g1.create_dataset('data1', data=X_domain)
        g1.create_dataset('data2', data=V[0])

        V_train_d, X_train_d, V_test_d, X_test_d= self.train_test_data(X_domain2, V[4])
        g2 = h5.create_group('domain2')
        g2.create_dataset('data1', data=V_train_d)
        g2.create_dataset('data2', data=V_test_d)
        g2.create_dataset('data3', data=X_train_d)
        g2.create_dataset('data4', data=X_test_d)

        V_train_o, X_train_o, V_test_o, X_test_o = self.train_test_data(X_outlet, V[2])
        V_train_w, X_train_w, V_test_w, X_test_w = self.train_test_data(X_wall, V[3])
        V_train_i, X_train_i, V_test_i, X_test_i = self.train_test_data(X_initial, V[4])
        V_train_b, X_train_b, V_test_b, X_test_b = self.train_test_data(X_boundary, V[5])

        train_data_vel = torch.cat((V_train_i,V_train_b,V_train_w,V_train_o),dim=0)
        test_data_vel = torch.cat((V_test_i,V_test_b,V_test_w,V_test_o),dim=0)
        train_data_X = torch.cat((X_train_i,X_train_b,X_train_w,X_train_o), dim=0)
        test_data_X = torch.cat((X_test_i,X_test_b,X_test_w,X_test_o), dim=0)
        g3 = h5.create_group('test_train')
        g3.create_dataset('data1', data=train_data_vel)
        g3.create_dataset('data2', data=test_data_vel)
        g3.create_dataset('data3', data=train_data_X)
        g3.create_dataset('data4', data=test_data_X)


        g4 = h5.create_group('extras')
        g4.create_dataset('data1', data=np.array(indices_domain))
        g4.create_dataset('data2', data=np.array([V[6], V[7]]))
        g4.create_dataset('data3', data=X_boundary)

        h5.close()


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
        self.X_train_d = X[1]
        self.X_test_d = X[2]
        self.X_train = X[3]
        self.X_test = X[4]

        self.V_domain  = V[0]
        self.V_train_d = V[1]
        self.V_test_d = V[2]
        
        self.V_train = V[3]

        self.V_test = V[4]

        self.V_min = V[5][1]
        self.V_max = V[5][0]

        self.iter = 0

        self.divider = 50

        self.training_loss = []
        self.error = []
    
    def forward(self, X):

        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X)
        X = X.to(self.device)

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


        V_pred = self.forward(X)

        laplace_phi, vorticity = self.laplace_potential(X)
        
        target = torch.zeros_like(laplace_phi, device = self.device)

        laplace_loss = self.loss_function(laplace_phi,target)

        vorticity_loss = self.loss_function(vorticity,target)

        velocity_loss = self.loss_function(V_pred,V)

        loss = laplace_loss + velocity_loss + vorticity_loss

        return loss

    def total_loss(self):

        loss_domain = self.loss(self.X_train_d,self.V_train_d)

        loss_bc = self.loss(self.X_train, self.V_train)

        total_loss =  loss_domain + loss_bc

        return total_loss

    def closure(self, optimizer, model):

        optimizer.zero_grad()

        loss = self.total_loss()
        
        loss.backward()

        print("Epoch: ", self.iter)

        self.iter+=1


        if self.iter % self.divider == 0:
            self.training_loss.append(loss)
            with torch.no_grad():
                error_vec, _ = model.test(model, self.X_test_d, self.V_test_d)
            self.error.append(error_vec)
            print(loss, error_vec)

        return loss

    def test(self, model,X_test, V_test):

        V_pred = model.forward(X_test)

        error_vec = torch.linalg.norm((V_test - V_pred), 2) / torch.linalg.norm(V_test,2)  # Relative L2 Norm of the error (vector)

        #V_pred = V_pred.cpu().detach().numpy()

        return error_vec, V_pred


########################################################################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

h =0.02

dec = 2

x_values = np.arange(-8.0, 8.0, h).tolist()
y_values = np.arange(-5.0, 5.0, h).tolist()
x_values.append(8.0)
y_values.append(5.0)


x_values = [ round(elem, dec) for elem in x_values ]
y_values = [ round(elem, dec) for elem in y_values ]
print(y_values)

x, y = np.meshgrid(x_values, y_values)

U0 = 10.0

#R = 2.0

m  = 120.0

a = 2.0

b = a/6.0

X_in = np.hstack([x.flatten()[:, None], y.flatten()[:, None]])

X_initial = np.hstack((x[ :,0][:, None], y[:,0][:, None]))

X_bc_lower = np.hstack((x[0,:][:, None], y[0,:][:, None]))

X_bc_upper = np.hstack((x[0,:][:, None], y[-1,:][:, None]))

X_wall = np.vstack([X_bc_upper, X_bc_lower])

X_outlet = np.hstack((x[:,-1][:, None], y[:,0][:, None]))

print('1')
nu = 0.8
norm = True

#preprocessing = Potential_flow_preprocessing(U0, m, h, device, a)

#preprocessing.data_generation()

######################################################################################################################
def sort_points(xy: np.ndarray) -> np.ndarray:
    # normalize data  [-1, 1]
    xy_sort = np.empty_like(xy)
    xy_sort[:, 0] = 2 * (xy[:, 0] - np.min(xy[:, 0])) / (np.max(xy[:, 0] - np.min(xy[:, 0]))) - 1
    xy_sort[:, 1] = 2 * (xy[:, 1] - np.min(xy[:, 1])) / (np.max(xy[:, 1] - np.min(xy[:, 1]))) - 1

    # get sort result
    sort_array = np.arctan2(xy_sort[:, 0], xy_sort[:, 1])
    sort_result = np.argsort(sort_array)

    # apply sort result
    return xy[sort_result]

def main(X_in):
    layers = np.array([2, 60, 60, 60,60,60, 2])



    epochs = 1

    h5 = h5py.File('data_rankine_oval_potential_flow.h5','r')

    domain = h5.get('domain')
    domain2 = h5.get('domain2')
    test_train = h5.get('test_train')
    extras = h5.get('extras')

    domain.items()
    domain2.items()
    test_train.items()
    extras.items()

    X_domain = np.array(domain.get('data1'))
    X_train_d = np.array(domain2.get('data3'))
    X_test_d = np.array(domain2.get('data4'))
    X_train = np.array(test_train.get('data3'))
    X_test = np.array(test_train.get('data4'))
    indices_domain = np.array(extras.get('data1'))
    X_boundary = np.array(extras.get('data3'))
    X_boundary_sort = sort_points(X_boundary)


    X = [X_domain,X_train_d,X_test_d,X_train,X_test]

    V_domain = np.array(domain.get('data2'))
    V_train_d = np.array(domain2.get('data1'))
    V_test_d = np.array(domain2.get('data2'))
    V_train = np.array(test_train.get('data1'))
    V_test = np.array(test_train.get('data2'))

    V_limit = np.array(extras.get('data2'))
    V_domain = torch.from_numpy(V_domain).float().to(device)
    V_train_d = torch.from_numpy(V_train_d).float().to(device)
    V_test_d = torch.from_numpy(V_test_d).float().to(device)
    V_train = torch.from_numpy(V_train).float().to(device)
    V_test = torch.from_numpy(V_test).float().to(device)

    V_limit = torch.from_numpy(V_limit).float().to(device)

    V = [V_domain,V_train_d,V_test_d,V_train,V_test,V_limit]

    model = Rankine_oval_PINN(layers,nu,X,V, norm,device)

    model.to(device)

    start_time = time.time()

    optimizerA = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizerB = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start_epoch =0
    PATH = "model.pt"
    '''if os.path.isfile(PATH):
        checkpoint = torch.load(cwd + '/' + PATH)
        start_epoch = checkpoint['epoch']
        #best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizerA.load_state_dict(checkpoint['optimizer_state_dict'])'''

    for e in range(start_epoch,epochs):
        optimizerA.step(lambda: model.closure(optimizerA, model))
        if e % 50 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizerA.state_dict(),
                'loss': model.closure(optimizerA, model),
                }, PATH)

    elapsed = time.time() - start_time

    print('Training time: %.2f' % (elapsed))

    preprocessing = Potential_flow_preprocessing(U0, m, h, device, a)

    V_domain_norm = (V_domain - V_domain.min()) / (V_domain.max() - V_domain.min())

    X_domain =  torch.from_numpy(X_domain).float().to(device)
    X_in = torch.from_numpy(X_in).float().to(device)

    V_in = preprocessing.velocity_cartesian_vjp(X_in)

    error, V_pred_norm = model.test(model,X_domain, V_domain_norm)
    print(error)

    V_pred = V_pred_norm * (V_domain.max() - V_domain.min()) + V_domain.min()

    result = [V_pred,V_domain, V_pred_norm, V_domain_norm,indices_domain, model.error, model.training_loss,X_boundary_sort, V_in]
    f = open('result_rankine_oval_potential_flow.pkl', 'wb')
    pickle.dump(result, f)
    f.close()


main(X_in)
