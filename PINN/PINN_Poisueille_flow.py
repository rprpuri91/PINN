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


class Preprocessing_poiseuille():
    def __init__(self, R, L, Q, mu):
        self.R = R
        self.L = L
        self.Q = Q
        self.mu = mu

    def pressure_drop(self):
        delta_p = (8 * self.mu * self.L * self.Q) / (np.pi * (self.R ** 4))
        return delta_p

    def velocity(self, X):
        r = X[:, 1]
        G = self.pressure_drop() / self.L
        u = []
        for i in range(len(r)):
            u_i = G * (self.R ** 2 - r[i] ** 2) / (4 * self.mu)
            u.append(u_i)

        vel = np.array(u)
        vel = torch.from_numpy(vel).float().to(device)
        return vel


class PINN_Poisuelle(nn.Module):
    def __init__(self, layers,nu,X_initial,vel_initial,G,mu, device):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.x = X_initial
        self.mu = mu
        self.v = vel_initial
        self.device = device
        self.G = G

        # Activation
        self.activation = nn.Tanh()
        self.activation2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        # loss function
        self.loss_function = nn.MSELoss(reduction ='mean')
        self.loss_function2 = nn.L1Loss()

        self.linears = nn.ModuleList(
            [nn.Linear(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)])

        self.iter = 0

        self.divider = 5

        self.training_loss = []
        self.error = []

        self.V_max = self.v.max()
        self.V_min = self.v.min()
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain =5/3)

            nn.init.zeros_(self.linears[i].bias.data)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        V_train = torch.reshape(V_train, (V_train.shape[0], 1))
        V_test = torch.reshape(V_test, (V_test.shape[0], 1))

        return V_train, X_train, V_test, X_test

    def velocity_pred(self, x, r):
        g = torch.cat((x, r), dim=1)

        velocity = self.forward(g)

        return velocity

    def velocity_pred2(self, x, r):
        g = torch.cat((x, r), dim=1)

        velocity = self.forward(g).sum()

        return velocity

    def continuity_equation(self, X):

        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X).to(self.device)


        g = torch.clone(X)
        g.requires_grad = True

        u_norm = self.forward(g)

        u_x_r = torch.autograd.grad(u_norm, g, torch.ones([X.shape[0], 1]).to(self.device), retain_graph=True,
                                    create_graph=True)[0]
        u_r = u_x_r[:,1]
        u_xx_rr = torch.autograd.grad(u_x_r, g, torch.ones(X.shape).to(self.device), create_graph=True)[0]
        u_rr = u_xx_rr[:, 1]
        #print(u_rr)
        #print(u_r)
        '''print(X[:,1])
        print(torch.div(u_r,X[:,1]))'''

        res = self.mu * (u_rr + torch.div(u_r, g[:,1]))

        return u_norm, res

    def loss(self):

        V_train, X_train, V_test, X_test = self.train_test_data(self.x, self.v)

        U_norm, res = self.continuity_equation(X_train)

        G = torch.full((res.shape), -self.G)
        G_mean = torch.div(G,U_norm)
        #print('g',G_norm)
        #target = torch.zeros_like(res, device=self.device)
        #print(target)
        loss_continuity = self.loss_function2(res, G_mean)
        print('continuity', loss_continuity)
        loss_velocity = self.loss_function2(U_norm, V_train)
        print('velocity', loss_velocity)
        loss =  loss_velocity +loss_continuity

        return loss

    def closure(self, optimizer, model):

        optimizer.zero_grad()

        loss = self.loss()

        loss.backward()

        print("Epoch: ", self.iter)

        self.iter += 1

        V_train, X_train, V_test, X_test = self.train_test_data(self.x, self.v)

        if self.iter % self.divider == 0:
            self.training_loss.append(loss)
            with torch.no_grad():
                error_vec, _ = model.test(model, X_test, V_test)
            self.error.append(error_vec)
            print(loss, error_vec)

        return loss

    def test(self, model, X_test, V_test):

        V_pred = model.forward(X_test)

        error_vec = torch.linalg.norm((V_test - V_pred), 2) / torch.linalg.norm(V_test,
                                                                                2)  # Relative L2 Norm of the error (vector)

        # V_pred = V_pred.cpu().detach().numpy()

        return error_vec, V_pred


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

R = 1
L = 20
Q = 0.00314
mu = 0.0010016  # 20°C water
rho = 1000
Re = 2000 # rho*Q*D/(mu*A) = 2*rho*Q*R/(mu*pi*R*R) = 2000*Q/(mu*pi*R) => Q = mu*pi*R => Q = 0.00314*R

# U = Q/(np.pi*(R^2))

preprocessing = Preprocessing_poiseuille(R, L, Q, mu)
G = preprocessing.pressure_drop()/L
x_values = np.arange(0.0, L, 0.05).tolist()
r_values = np.arange(-R, R, 0.05).tolist()
x_values.append(L)
r_values.append(R)

x, r = np.meshgrid(x_values, r_values)

X_in = np.hstack([x.flatten()[:, None], r.flatten()[:, None]])

X_initial = np.hstack((x[:, 0][:, None], r[:, 0][:, None]))

vel_initial = preprocessing.velocity(X_initial)

layers = np.array([2, 60, 60, 60,60,60, 1])

nu = 0.8

epochs = 2000
def main():

    model = PINN_Poisuelle(layers,nu,X_initial,vel_initial,G,mu, device)

    model.to(device)

    start_time = time.time()

    optimizerA = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizerB = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for e in range(0,epochs):
        optimizerA.step(lambda: model.closure(optimizerA, model))

    elapsed = time.time() - start_time

    print('Training time: %.2f' % (elapsed))

    V_norm = model.normalize_velocity(vel_initial)
    error, V_pred_norm = model.test(model, X_initial, V_norm)

    v_pred = model.denormalize_velocity(V_pred_norm)
    U = Q/(np.pi*(R**2))

    v_avg = torch.div(vel_initial,U)
    v_avg_pred = torch.div(v_pred,U)

    result = [v_avg,v_avg_pred, model.training_loss, model.error, error]
    f = open('result_poisueille_flow.pkl', 'wb')
    pickle.dump(result, f)
    f.close()



def plotting():
    file0 = open('result_poisueille_flow.pkl', 'rb')
    data0 = pickle.load(file0)

    v_avg = data0[0]
    v_avg_pred = data0[1]
    error_values = data0[3]
    plt.plot(error_values)
    fig, ax = plt.subplots(1, 1)
    ax.plot(v_avg.cpu().detach().numpy(), r, label='U_exact')
    ax.hlines(y=-R, xmin=-1, xmax=5, color='black')
    ax.hlines(y=R, xmin=-1, xmax=5, color='black')
    ax.set_xlabel('$u_x$/U', rotation=0)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    ax.annotate('------------- $R_p$ -------------->', xy=(0.0, 0.0), xycoords='data', fontsize=9.7, rotation=90)
    ax.annotate('Pipe wall, $u_x = 0$', xy=(0.0, R+0.0001), xycoords='data', fontsize=9.7)
    ax.annotate('Pipe wall, $u_x = 0$', xy=(0.0, -R-0.0004 ), xycoords='data', fontsize=9.7)
    ax.annotate('--->', xy=(-1.0, 0.0), xycoords='data', fontsize=15)
    ax.set_xlim(-1, 2.5)
    ax.set_xticks((0,0.5,1,1.5,2))
    ax.set_xticklabels((0,0.5,1,1.5,2))
    ax.scatter(v_avg_pred.cpu().detach().numpy(), X_initial[:,1], label='U_pred', marker='x', color='Orange')
    ax.set_title('$u_x/U$', y=-0.1)
    #ax.legend()
    #ax[1].hlines(y=-R, xmin=-1, xmax=5, color='black')
    #ax[1].hlines(y=R, xmin=-1, xmax=5, color='black')
    #ax[1].set_xlabel('$u_x$/U', rotation=0)
    #ax[1].axes.yaxis.set_visible(False)
    #ax[1].set_xlim(-1, 5)
    plt.show()


plotting()
#main()