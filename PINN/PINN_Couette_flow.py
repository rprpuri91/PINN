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
        return vel




class PINN_Poisuelle(nn.Module):
    def __init__(self, layers,nu,x,v,Gc,mu,U, device):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.x = x
        self.Gc = Gc
        self.v = v
        self.device = device
        self.mu = mu
        self.U = U
        # Activation
        self.activation = nn.Tanh()
        self.activation2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        # loss function
        self.loss_function = nn.MSELoss(reduction='mean')

        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)])

        self.iter = 0

        self.divider = 5

        self.training_loss = []
        self.error = []

        self.V_max = self.v.max()
        self.V_min = self.v.min()

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

        X_star = X[idx, ]
        V_train = V[idx, ].float()

        X_test = np.delete(X, idx, axis=0)
        idxtest = []

        for i in range(0, X.shape[0]):
            if i in idx:
                continue
            else:
                idxtest.append(i)

        V_test = V[idxtest, ].float()

        X_train = torch.from_numpy(X_star).float().to(self.device)
        X_test = torch.from_numpy(X_test).float().to(self.device)

        V_train = self.normalize_velocity(V_train)
        V_test = self.normalize_velocity(V_test)
        V_train = torch.reshape(V_train, (V_train.shape[0],1))
        V_test = torch.reshape(V_test, (V_test.shape[0], 1))

        return V_train, X_train, V_test, X_test

    def velocity_pred(self,x,r):
        g = torch.cat((x,r), dim = 1)

        velocity = self.forward(g).sum()

        return velocity

    def continuity_equation(self,X):
        #u_x'' = -G_c/mu

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
        u_yy = u_xx_yy[:,1]
        #H = torch.autograd.functional.hessian(self.velocity_pred, (x,y), create_graph=True)
        #u_yy = torch.diagonal(H[1][1], 0)

        G = torch.full((u_yy.shape), self.Gc)
        res = u_yy -G/(self.mu*self.U)
        return res

    def loss(self):

        V_train, X_train, V_test, X_test = self.train_test_data(self.x,self.v)

        res = self.continuity_equation(X_train)

        U = self.forward(X_train)

        target = torch.zeros_like(res, device = self.device)

        #loss_continuity = self.loss_function(res,target)
        #print('continuity',loss_continuity)
        loss_velocity = self.loss_function(U,V_train)
        print('velocity',loss_velocity)
        loss =  loss_velocity

        return loss

    def closure(self, optimizer, model):

        optimizer.zero_grad()

        loss = self.loss()

        loss.backward()

        print("Epoch: ", self.iter)

        self.iter+=1

        V_train, X_train, V_test, X_test = self.train_test_data(self.x,self.v)

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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

h = 6.0
#Gc = 500
U = 10.0
mu = 1.0016e-3  # 20°C water
L = 100
P = 1

preprocessing = Preprocessing_poiseuille(h, P, U, mu)
Gc = -preprocessing.pressure_drop()

x_values = np.arange(0.0, L, 0.05).tolist()
x1_values = np.arange(0.0, L, 0.3).tolist()
y_values = np.arange(0.0, h, 0.05).tolist()
y1_values = np.arange(0.0, h, 0.3).tolist()
x_values.append(L)
x1_values.append(L)
y_values.append(h)
y1_values.append(h)

x,y = np.meshgrid(x_values,y_values)
x1,y1 =np.meshgrid(x1_values,y1_values)

X_initial = np.hstack((x[ :,0][:, None], y[ :,0][:, None]))
X_initial1 = np.hstack((x1[ :,0][:, None], y1[ :,0][:, None]))

vel_initial = preprocessing.velocity(X_initial)
vel_initial1 = preprocessing.velocity(X_initial1)
print(vel_initial.shape)
layers = np.array([2, 60, 60, 60,60,60, 1])

nu = 0.8

epochs = 1000

def main():
    model = PINN_Poisuelle(layers,nu,X_initial,vel_initial,Gc,mu,U, device)

    model.to(device)

    start_time = time.time()

    optimizerA = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizerB = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for e in range(0,epochs):
        optimizerA.step(lambda: model.closure(optimizerA, model))

    elapsed = time.time() - start_time

    print('Training time: %.2f' % (elapsed))

    V_norm = model.normalize_velocity(vel_initial1)
    error, V_pred_norm = model.test(model, X_initial1, V_norm)



    v_pred = model.denormalize_velocity(V_pred_norm)
    v_avg = torch.div(vel_initial1,U)
    v_avg_pred = torch.div(v_pred,U)

    result = [v_avg, v_avg_pred, model.training_loss, model.error]
    f = open('result_couette_flow.pkl', 'wb')
    pickle.dump(result, f)
    f.close()


def plotting():
    file0 = open('result_couette_flow.pkl', 'rb')
    data0 = pickle.load(file0)

    v_avg = data0[0]
    v_avg_pred = data0[1]
    fig, ax = plt.subplots(1,2)
    ax[0].stem(X_initial1[:,1], v_avg.cpu().detach().numpy(), orientation='horizontal',markerfmt= '>', label='U_exact')
    ax[0].plot(v_avg.cpu().detach().numpy(),y1)
    ax[0].annotate('Moving plate, $U=10$', xy=(-0.5, 6.1), xycoords='data', fontsize=9.7)
    ax[0].annotate('Stationary plate, $U=0$', xy=(-0.5, -0.2), xycoords='data', fontsize=9.7)
    ax[0].axes.yaxis.set_visible(False)
    ax[0].axes.xaxis.set_visible(False)
    ax[0].set_xlim(-1, 2)
    ax[0].set_title('$u_{exact}/U$', y=-0.1)
    ax[0].title.set_fontsize(20)
    ax[0].hlines(y=h, xmin = -U/2, xmax = 2*U, color='Black')
    ax[0].hlines(y=0, xmin = -U/2, xmax = 2*U, color='Black')
    ax[1].stem(X_initial1[:,1],v_avg_pred.cpu().detach().numpy(),orientation='horizontal', markerfmt ='>', label='U_pred')
    ax[1].plot(v_avg_pred.cpu().detach().numpy(),y1)
    ax[1].annotate('Moving plate, $U=10$', xy=(-0.5, 6.1), xycoords='data', fontsize=9.7)
    ax[1].annotate('Stationary plate, $U=0$', xy=(-0.5, -0.2), xycoords='data', fontsize=9.7)
    ax[1].axes.yaxis.set_visible(False)
    ax[1].axes.xaxis.set_visible(False)
    ax[1].set_xlim(-1, 2)
    ax[1].set_title('$u_{pred}/U$', y=-0.1)
    ax[1].title.set_fontsize(20)
    ax[1].hlines(y=h, xmin = -U/2, xmax = 2*U, color='Black')
    ax[1].hlines(y=0, xmin = -U/2, xmax = 2*U, color='Black')
    plt.show()

#main()
plotting()

