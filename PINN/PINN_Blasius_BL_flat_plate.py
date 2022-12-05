import pickle
import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sc
import torch
from torch import nn


class Blasius_Preprocessing():
    def __init__(self, U0, L, rho0, mu, x):

        self.U0 = U0
        self.L = L
        self.rho0 = rho0
        self.mu = mu
        self.X = x
        self.f0 = 0
        self.g0 = 0
        self.h0 = np.array([0.1, 0.2, 0.3])
        self.nu = self.mu / self.rho0

    def reynolds_fn_x(self, x):
        Rex = self.rho0 * self.U0 * x / self.mu
        return Rex

    def boundary_thickness(self, x):
        dx = x / sqrt(self.reynolds_fn_x(x))
        return dx

    def independent_eta(self):
        eta_values = []
        for i in range(len(self.X)):
            print(i)
            if self.X[i][0] == 0 and self.X[i][1] == 0:
                eta_values.append(0)
            else:
                eta = self.X[i][1] / self.boundary_thickness(self.X[i][0])
                eta_values.append(eta)
        return np.array(eta_values)

    def data_from_matlab(self):
        eta = sc.loadmat('Blasius_eta.mat')
        eta_values = torch.tensor(list(eta.values())[3].T)

        F = sc.loadmat('Blasius_F.mat')
        F_values = list(F.values())[3]
        f = torch.tensor(F_values[:, 0])
        g = torch.tensor(F_values[:, 1])
        h = torch.tensor(F_values[:, 2])

        return eta_values, f, g, h

    def exact_velocity(self):
        eta, f, g, h = self.data_from_matlab()
        v = []
        u = self.U0 * g  # U*df/dn

        for i in range(len(self.X)):

            if (self.X[i] == 0):
                v.append(0.0)
            else:
                vi = 0.5 * np.sqrt(self.nu * self.U0 / self.X[i]) * (g[i] - f[i])
                v.append(vi)

        v = np.array(v)

        return u, v


class PINN_blasius(nn.Module):
    def __init__(self, layers, nu, eta, f, device):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.eta = eta
        self.f = f
        print(type(self.f))
        self.device = device
        self.mu = mu

        # Activation
        self.activation = nn.Tanh()
        self.activation2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        # loss function
        self.loss_function = nn.MSELoss(reduction='mean')

        self.linears = nn.ModuleList(
            [nn.Linear(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)])

        self.iter = 0

        self.divider = 5

        self.training_loss = []
        self.error = []

        self.f_max = self.f.max()
        self.f_min = self.f.min()

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

    def normalize_velocity(self, f):
        f_norm = (f - self.f_min) / (self.f_max - self.f_min)

        return f_norm

    def denormalize_velocity(self, f_norm):
        f = f_norm * (self.f_max - self.f_min) + self.f_min

        return f

    def train_test_data(self, eta, f):

        N_u = int(self.nu * len(eta))

        idx = np.random.choice(eta.shape[0], N_u, replace=False)

        eta_train = eta[idx,]
        f_train = f[idx,].float()

        eta_test = np.delete(eta, idx, axis=0)
        idxtest = []

        for i in range(0, eta.shape[0]):
            if i in idx:
                continue
            else:
                idxtest.append(i)

        f_test = f[idxtest,].float()

        # eta_train = torch.from_numpy(eta_star).float().to(self.device)
        # eta_test = torch.from_numpy(eta_test).float().to(self.device)

        f_train = self.normalize_velocity(f_train)
        f_test = self.normalize_velocity(f_test)
        f_train = torch.reshape(f_train, (f_train.shape[0], 1))
        f_test = torch.reshape(f_test, (f_test.shape[0], 1))

        return f_train, eta_train, f_test, eta_test

    def continuity_equation(self, X):
        # u_x'' = -G_c/mu

        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X).to(self.device)

        g = torch.clone(X)
        g.requires_grad = True
        u = self.forward(g)

        '''X = torch.split(X, 1, dim=1)
        x = X[0]
        y = X[1]'''

        u_x_y = torch.autograd.grad(u, g, torch.ones([X.shape[0], 1]).to(self.device), retain_graph=True,
                                    create_graph=True)[0]

        u_xx_yy = torch.autograd.grad(u_x_y, g, torch.ones(X.shape).to(self.device), create_graph=True)[0]
        u_yy = u_xx_yy[:, 1]
        # H = torch.autograd.functional.hessian(self.velocity_pred, (x,y), create_graph=True)
        # u_yy = torch.diagonal(H[1][1], 0)

        G = torch.full((u_yy.shape), self.Gc)
        res = u_yy - G / (self.mu * self.U)
        return res

    def loss(self):

        f_train, eta_train, f_test, eta_test = self.train_test_data(self.eta, self.f)

        # res = self.continuity_equation(X_train)

        f = self.forward(eta_train)

        # target = torch.zeros_like(res, device=self.device)

        # loss_continuity = self.loss_function(res, target)
        # print('continuity', loss_continuity)
        loss_velocity = self.loss_function(f, f_train)
        # print('velocity',loss_velocity)
        loss = loss_velocity

        return loss

    def closure(self, optimizer, model):

        optimizer.zero_grad()

        loss = self.loss()

        loss.backward()

        print("Epoch: ", self.iter)

        self.iter += 1

        f_train, eta_train, f_test, eta_test = self.train_test_data(self.eta, self.f)

        if self.iter % self.divider == 0:
            self.training_loss.append(loss)
            with torch.no_grad():
                error_vec, _ = model.test(model, eta_test, f_test)
            self.error.append(error_vec)
            print(loss, error_vec)

        return loss

    def test(self, model, eta_test, f_test):

        f_pred = model.forward(eta_test)

        error_vec = torch.linalg.norm((f_test - f_pred), 2) / torch.linalg.norm(f_test,
                                                                                2)  # Relative L2 Norm of the error (vector)

        # V_pred = V_pred.cpu().detach().numpy()

        return error_vec, f_pred


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

'''
% ODE => { f '= g ; g' = h ; h'=-0.5*f*h }  , here η is independent variable
% initial condition f(0) = 0 , g(0) = 0 , h(0) = ? such that g(∞) = 1 ;
% we find h(0) by obtaining improved values of h(0) that minimises the
% error g(∞)-1 to 0
% η (similarity variable) is of the order y/δ(x) so η_max = ∞ (10)
'''

s = 0.0001
L = 1
x_values = np.arange(0, L, s).tolist()
y_values = np.arange(0, 0.01, 0.001).tolist()
x_values.append(L)
# y_values.append(0.01)
U0 = 1
rho0 = 1000
mu = 0.001
nu = 0.8

x, y = np.meshgrid(x_values, y_values)

X_in = np.hstack([x.flatten()[:, None], y.flatten()[:, None]])

print(X_in.shape)
preprocessing = Blasius_Preprocessing(U0, L, rho0, mu, x_values)
# eta = preprocessing.independent_eta()

eta, f, g, h = preprocessing.data_from_matlab()


def main():
    eta, f, g, h = preprocessing.data_from_matlab()
    vel = preprocessing.exact_velocity()
    layers = np.array([1, 60, 60, 60, 60, 60, 1])

    epochs = 100

    model = PINN_blasius(layers, nu, eta, f, device)

    model.to(device)

    start_time = time.time()

    optimizerA = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizerB = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for e in range(0, epochs):
        optimizerA.step(lambda: model.closure(optimizerA, model))

    elapsed = time.time() - start_time

    print('Training time: %.2f' % (elapsed))

    f_norm = model.normalize_velocity(f)
    error, f_pred = model.test(model, eta, f_norm)

    result = [f_norm, f_pred, model.training_loss, model.error]
    f = open('result_blasius_flow.pkl', 'wb')
    pickle.dump(result, f)
    f.close()


def plotting():
    file0 = open('result_blasius_flow.pkl', 'rb')
    data0 = pickle.load(file0)

    f_norm = data0[0]
    f_pred = data0[1]
    fig, ax = plt.subplots(1, 1)
    ax.plot(f_norm.cpu().detach().numpy(), eta, label='U_exact')
    ax.set_xlabel('f', rotation=0)
    ax.axes.yaxis.set_visible(False)
    ax.set_xlim(-1, 2)
    ax.plot(f_pred.cpu().detach().numpy(), eta, label='U_pred', marker='x')
    # ax[1].hlines(y=-R, xmin=-1, xmax=5, color='black')
    # ax[1].hlines(y=R, xmin=-1, xmax=5, color='black')
    # ax[1].set_xlabel('$u_x$/U', rotation=0)
    # ax[1].axes.yaxis.set_visible(False)
    # ax[1].set_xlim(-1, 5)
    plt.show()


main()
