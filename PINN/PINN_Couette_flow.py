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
from matplotlib.markers import MarkerStyle
from scipy.signal import lfilter

class Preprocessing_poiseuille():
    def __init__(self, h, P, U, mu):

        self.h = h
        self.P = P
        self.U = U
        self.mu = mu

    def pressure_drop(self):
        dp_dx = -self.P*2*self.mu*self.U/(self.h**2)
        return dp_dx

    def velocity(self,X):
        #u = U(y/h) - (1/2mu)Gc(y^2 - hy)
        y = X[:,1]

        G_c = -self.pressure_drop()
        print('G_c',G_c)
        u = []
        for i in range(len(y)):
            u_i = self.U*(y[i]/self.h) - (G_c/(2*self.mu))*(y[i]**2 - self.h*y[i])
            u.append(u_i)

        vel = np.array(u)

        vel = torch.from_numpy(vel).float().to(device)
        return torch.reshape(vel, (vel.shape[0],1))

    def pressure(self,X):
        x = X[:,0]
        p = []
        dpdx = self.pressure_drop()
        for i in range(len(x)):
            p_i = dpdx*x[i]
            p.append(p_i)

        p = np.array(p)
        p_tensor = torch.from_numpy(p).float().to(device)
        return torch.reshape(p_tensor, (p_tensor.shape[0],1))




class PINN_Poisuelle(nn.Module):
    def __init__(self, layers,nu,X_domain,X_wall,U_domain,U_wall,X_initial,U_initial,Gc,mu,U, device):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.X_domain = X_domain
        self.X_wall = X_wall
        self.Gc = Gc
        self.U_domain = U_domain
        self.device = device
        self.mu = mu
        self.U0 = U
        self.U_wall= U_wall
        self.X_initial = X_initial
        self.U_initial = U_initial
        # Activation
        self.activation = nn.Tanh()
        self.activation2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.activation3 = nn.PReLU(num_parameters=80)

        # loss function
        self.loss_function = nn.MSELoss(reduction='mean')
        self.loss_function2 = nn.L1Loss()

        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)])

        self.iter = 0

        self.divider = 5

        self.training_loss = []
        self.error = []

        self.V_max = self.U_initial.max()
        self.V_min = self.U_initial.min()

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

    def normalize_velocity(self, V):
        V_norm = (V - self.V_min) / (self.V_max - self.V_min)

        return V_norm

    def denormalize_velocity(self, V_norm):
        V = V_norm * (self.V_max - self.V_min) + self.V_min

        return V

    def normalize(self,X):
        x_min = X.min()
        x_max = X.max()

        X_norm = (X - x_min)/(x_max-x_min)
        return X_norm

    def train_test_data(self, X, V):

        N_u = int(self.nu * len(X))

        idx = np.random.choice(X.shape[0], N_u, replace=False)

        X_star = X[idx,]
        V_train = V[idx,].float()


        X_test = np.delete(X, idx, axis=0)
        idxtest = []

        for i in range(0, X.shape[0]):
            if i in idx:
                continue
            else:
                idxtest.append(i)

        V_test = V[idxtest,].float()


        X_train = torch.from_numpy(X_star).float().to(self.device)
        X_test = torch.from_numpy(X_test).float().to(self.device)

        V_train = self.normalize_velocity(V_train)
        V_test = self.normalize_velocity(V_test)

        #V_train = torch.reshape(V_train, (V_train.shape[0], 1))
        #V_test = torch.reshape(V_test, (V_test.shape[0], 1))


        return V_train, X_train, V_test, X_test
    def variable_pred(self, x, r):
        g = torch.cat((x, r), dim=1)

        U = self.forward(g)

        return U

    def variable_pred_hessian(self, x, r):
        g = torch.cat((x, r), dim=1)

        U = self.forward(g).sum()
        return U

    def governing_equation(self, X):

        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X).to(self.device)

        v3 = torch.zeros(X.shape[0], 1).to(self.device)
        v4 = torch.ones(X.shape[0], 1).to(self.device)
        g = torch.clone(X)
        g.requires_grad = True

        v1 = torch.zeros_like(X, device=self.device)
        v2 = torch.zeros_like(X, device=self.device)

        v1[:, 0] = 1
        v2[:, 1] = 1
        U = self.forward(g)
        '''X = torch.split(X, 1, dim=1)

        x = X[0]
        y = X[1]'''

        #U, U_x_y = torch.autograd.functional.vjp(self.variable_pred, (x, y), v1, create_graph=True)
        U_x_y = torch.autograd.grad(U, g, torch.ones([X.shape[0], 1]).to(self.device), retain_graph=True,
                                    create_graph=True)[0]
        u_x = U_x_y[:,0]
        u_y = U_x_y[:,1]

        U_xx_yy = torch.autograd.grad(U_x_y, g, torch.ones(X.shape).to(self.device), retain_graph=True, create_graph=True)[0]
        U_yy = U_xx_yy[:,1]

        U_xxx_yyy = torch.autograd.grad(U_x_y, g, torch.ones(X.shape).to(self.device), create_graph=True)[0]
        U_yyy = U_xxx_yyy[:,1]
        '''U1, P_x_y = torch.autograd.functional.vjp(self.variable_pred, (x, y), v2, create_graph=True)
        p_x = P_x_y[0]
        p_r = P_x_y[1]'''

        '''C, H = torch.autograd.functional.vhp(self.variable_pred_hessian, (x, y), (v3, v4), create_graph=True)
        u_yy = H[1]'''
        #G = torch.full(U_yy.shape,self.Gc)
        #res = u_yy - self.mu*p_x
        #res = U_yy - G/(self.mu*self.U0)
        res = U_yyy
        #print('res', res)
        return U, res, u_x

    def loss(self,x,v):

        V_train, X_train, V_test, X_test= self.train_test_data(x,v)

        U, res, u_x = self.governing_equation(X_train)

        #print(res)
        #G = torch.full((res.shape), -self.Gc/(self.mu*self.U0))

        target = torch.zeros_like(res, device = self.device)
        target1 = torch.zeros_like(u_x, device = self.device)
        #target2 = torch.zeros_like(p_r, device = self.device)

        loss_residual = self.loss_function(res, target)
        #print('residual', loss_residual)

        #loss_diff_pressure = self.loss_function(p_r, target2)
        #print('diff_pressure',loss_diff_pressure)

        loss_ux = self.loss_function(u_x, target1)
        #print('loss ux',loss_ux)

        loss_variable = self.loss_function(U, V_train)
        #print('variable', loss_variable)



        loss =  loss_variable + loss_residual + loss_ux

        return loss

    def total_loss(self):

        loss1 = self.loss(self.X_domain,self.U_domain)

        loss2 = self.loss(self.X_wall, self.U_wall)

        loss3 = self.loss(self.X_initial,self.U_initial)

        loss = loss1 + loss2 + loss3

        return loss

    def closure(self, optimizer, model):

        optimizer.zero_grad()

        loss = self.total_loss()

        loss.backward()

        print("Epoch: ", self.iter)

        self.iter+=1
        #print('test')
        V_train, X_train, V_test, X_test = self.train_test_data(self.X_domain,self.U_domain)

        if self.iter % self.divider == 0:
            self.training_loss.append(loss)
            with torch.no_grad():
                error_vec, _ = model.test(model, X_test, V_test)
            self.error.append(error_vec)
            print(loss, error_vec)
        #print('end')
        return loss

    def test(self, model, X_test, V_test):

        U = model.forward(X_test)

        error_vec = torch.linalg.norm((V_test - U), 2) / torch.linalg.norm(V_test,
                                                                                2)  # Relative L2 Norm of the error (vector)
        # V_pred = V_pred.cpu().detach().numpy()

        return error_vec, U

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

h = 1.0
#Gc = 500
U = 10.0
mu = 1.0016e-3  # 20°C water
L = 20
P = 1

preprocessing = Preprocessing_poiseuille(h, P, U, mu)
Gc = -preprocessing.pressure_drop()

x_values = np.arange(0.0, L, 0.05).tolist()

y_values = np.arange(0.0, h, 0.05).tolist()

x_values.append(L)

y_values.append(h)


x,y = np.meshgrid(x_values,y_values)

X_mid = np.hstack((x[ :,200][:, None], y[ :,0][:, None]))
X_initial = np.hstack((x[ :,0][:, None], y[ :,0][:, None]))
X_in = np.hstack([x.flatten()[:, None], y.flatten()[:, None]])

X_wall_lower = np.hstack((x[0,:][:, None], y[0,:][:, None]))

X_wall_upper = np.hstack((x[0,:][:, None], y[-1,:][:, None]))

X_wall = np.vstack([X_wall_upper, X_wall_lower])

vel_initial = preprocessing.velocity(X_initial)
p_initial = preprocessing.pressure(X_initial)
U_initial = torch.cat((vel_initial,p_initial), dim=1)


vel_mid = preprocessing.velocity(X_mid)
p_mid = preprocessing.velocity(X_mid)
U_mid = torch.cat((vel_mid,p_mid), dim=1)

vel_wall = preprocessing.velocity(X_wall)

p_wall = preprocessing.pressure(X_wall)

U_wall = torch.cat((vel_wall,p_wall),dim=1)

N_x = int(0.7 * len(X_in))
idx = np.random.choice(X_in.shape[0], N_x, replace=False)

X_domain = X_in[idx, :]

vel_domain = preprocessing.velocity(X_domain)
p_domain = preprocessing.pressure(X_domain)

U_domain = torch.cat((vel_domain,p_domain),dim=1)

layers = np.array([2, 80, 80, 80,80,80, 1])

nu = 0.8

epochs = 5000

def main():
    model = PINN_Poisuelle(layers,nu,X_domain,X_wall,vel_domain,vel_wall,X_initial,vel_initial,Gc,mu,U, device)

    model.to(device)

    start_time = time.time()

    optimizerA = torch.optim.Adam(model.parameters(), lr=0.005)
    optimizerB = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    for e in range(0,epochs):
        optimizerB.step(lambda: model.closure(optimizerB, model))

    elapsed = time.time() - start_time

    print('Training time: %.2f' % (elapsed))

    U_norm = model.normalize_velocity(U_initial)
    error, U_pred_norm = model.test(model, X_initial, U_norm)

    '''v_pred_norm = U_pred_norm[:,0]

    v_pred = v_pred_norm * (vel_initial.max() - vel_initial.min()) + vel_initial.min()
    v_avg = torch.div(vel_initial,U)
    v_avg_pred = torch.div(v_pred,U)'''

    result = [U_norm, U_pred_norm, model.training_loss, model.error, error, elapsed]
    f = open('result_NN_sgd_8000_couette_flow.pkl', 'wb')
    pickle.dump(result, f)
    f.close()


def plotting():
    plt.rcParams.update({

        "font.family": "times new roman",
        "mathtext.default": "regular"
    })
    file0 = open('result_PINN_sgd_couette_flow.pkl', 'rb')
    data0 = pickle.load(file0)
    file1 = open('result_NN_sgd_8000_couette_flow.pkl', 'rb')
    data1 = pickle.load(file1)

    U_norm = data0[0]
    U_avg_pred = data0[1]
    v_avg = U_norm[:,0]
    v_avg_pred = U_avg_pred[:,0]
    error_plot = data0[3]
    error = data1[4]
    print(error)


    U_avg_pred1 = data1[1]
    v_avg_pred1 = U_avg_pred1[:, 0]
    error_plot1 = data1[3]
    n = 30
    b = [1.0 / n] * n
    a = 1

    e1 = lfilter(b, a, error_plot)

    e2 = lfilter(b, a, error_plot1)
    error_loss_plot(e1, e2)
    fig, ax = plt.subplots(1,1)
    ax.stem(X_initial[:,1], v_avg.cpu().detach().numpy(), orientation='horizontal',markerfmt= '>', )
    ax.plot(v_avg.cpu().detach().numpy(),X_initial[:,1], label='U_exact')
    ax.axes.yaxis.set_visible(False)
    #ax.axes.xaxis.set_visible(False)
    ax.set_xlim(0, 1.5)
    ax.set_xticks((0, 0.5,1.0,1.5), fontsize=20)
    #ax.set_title('$u/U (PINN)$', y=-0.1)
    #ax.title.set_fontsize(20)
    ax.hlines(y=h, xmin = -U/2, xmax = 2*U, color='Black')
    ax.hlines(y=0, xmin = -U/2, xmax = 2*U, color='Black')
    ax.scatter(v_avg_pred.cpu().detach().numpy(),X_initial[:,1],color='Orange', marker ='s' , label='PINN')
    ax.scatter(v_avg_pred1.cpu().detach().numpy(), X_initial[:, 1], color='Red', marker='x', label='DNN')
    #fig.patch.set_visible(False)
    ax.legend(borderaxespad=1.5)
    ax.axis('off')

    plt.show()

def error_loss_plot(error1,error2):
    e= int(epochs/5)
    xmax = int(e)
    x1 = [*range(1, xmax + 1)]

    fig,ax = plt.subplots(1,1)
    #ax.plot(x1,error1, label='PINN with SGD')
    ax.plot(x1, error2, label='DNN with SGD')
    #ax.plot(x1, e3, label='PINN with ADAM')
    #ax[0][0].set_yscale('log')
    ax.set_ylim(0,0.5)
    ax.set_yticks((0,0.1,0.2,0.3,0.4,0.5), fontsize=20)
    ax.set_xticks((0,e/4,2*e/4,3*e/4,e), fontsize=20)
    ax.set_xticklabels((0,epochs/4,2*epochs/4,3*epochs/4,epochs), fontsize=10)
    #ax.set_ylabel('L2 Error')
    #ax.set_xlabel('Epochs')
    #ax.xaxis.label.set_fontsize(15)
    #ax.yaxis.label.set_fontsize(15)
    ax.grid()
    ax.legend()

#main()
plotting()

