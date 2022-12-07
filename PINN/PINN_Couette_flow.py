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

    def pressure(self,X):
        x = X[:,0]
        p = []
        dpdx = self.pressure_drop()
        for i in range(len(x)):
            p_i = dpdx*x[i]
            p.append(p_i)

        p = np.array(p)
        p_tensor = torch.from_numpy(p).float().to(device)
        return p_tensor




class PINN_Poisuelle(nn.Module):
    def __init__(self, layers,nu,X_in,vel_in,Gc,p_in,mu,U, device):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.x = X_in
        self.Gc = Gc
        self.v = vel_in
        self.device = device
        self.mu = mu
        self.U0 = U
        self.p = p_in
        # Activation
        self.activation = nn.Tanh()
        self.activation2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        # loss function
        self.loss_function = nn.MSELoss(reduction='mean')
        self.loss_function2 = nn.L1Loss()

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

    def normalize(self,X):
        x_min = X.min()
        x_max = X.max()

        X_norm = (X - x_min)/(x_max-x_min)
        return X_norm

    def train_test_data(self, X, V, P):

        N_u = int(self.nu * len(X))

        idx = np.random.choice(X.shape[0], N_u, replace=False)

        X_star = X[idx, ]
        V_train = V[idx, ].float()
        P_train = P[idx,]

        X_test = np.delete(X, idx, axis=0)
        idxtest = []

        for i in range(0, X.shape[0]):
            if i in idx:
                continue
            else:
                idxtest.append(i)

        V_test = V[idxtest, ].float()
        P_test = P[idxtest,].float()

        X_train = torch.from_numpy(X_star).float().to(self.device)
        X_test = torch.from_numpy(X_test).float().to(self.device)

        V_train = self.normalize_velocity(V_train)
        V_test = self.normalize_velocity(V_test)
        P_train = self.normalize(P_train)
        P_test = self.normalize(P_test)
        V_train = torch.reshape(V_train, (V_train.shape[0], 1))
        V_test = torch.reshape(V_test, (V_test.shape[0], 1))
        P_train = torch.reshape(P_train, (P_train.shape[0], 1))
        P_test = torch.reshape(P_test, (P_test.shape[0], 1))

        return V_train, X_train, V_test, X_test, P_train, P_test

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

        X = torch.split(X, 1, dim=1)

        x = X[0]
        y = X[1]

        U, U_x_y = torch.autograd.functional.vjp(self.variable_pred, (x, y), v1, create_graph=True)
        '''U_x_r = torch.autograd.grad(U, g, torch.ones([X.shape[0], 2]).to(self.device), retain_graph=True,
                                    create_graph=True)[0]'''
        # U_x = U_x_r[:,0]
        # U_r = U_x_r[:,1]
        u_x = U_x_y[0]
        u_y = U_x_y[1]

        U1, P_x_y = torch.autograd.functional.vjp(self.variable_pred, (x, y), v2, create_graph=True)
        p_x = P_x_y[0]
        p_r = P_x_y[1]

        # u_xx_rr = torch.autograd.grad(u_r, g,v , create_graph=True)[0]
        # u_rr = u_xx_rr[:, 1]
        C, H = torch.autograd.functional.vhp(self.variable_pred_hessian, (x, y), (v3, v4), create_graph=True)
        u_yy = H[1]

        res = u_yy - self.mu*p_x
        # print('res', res.shape)
        return U, u_x, res, p_r

    def loss(self):

        V_train, X_train, V_test, X_test, P_train, P_test = self.train_test_data(self.x,self.v,self.p)

        U, u_x, res, p_r = self.governing_equation(X_train)

        #print(res)
        #G = torch.full((res.shape), -self.Gc/(self.mu*self.U0))
        u = torch.reshape(U[:, 0], (U[:, 0].shape[0], 1))
        p = torch.reshape(U[:, 1], (U[:, 1].shape[0], 1))

        target = torch.zeros_like(res, device = self.device)
        target1 = torch.zeros_like(u_x, device = self.device)
        target2 = torch.zeros_like(p_r, device = self.device)

        loss_residual = self.loss_function(res, target)
        print('residual', loss_residual)

        loss_diff_pressure = self.loss_function(p_r, target2)
        print('diff_pressure',loss_diff_pressure)

        loss_ux = self.loss_function(u_x, target1)
        print('loss ux',loss_ux)

        loss_velocity = self.loss_function(u, V_train)
        print('velocity', loss_velocity)

        loss_presure = self.loss_function(p, P_train)
        print('loss_pressure', loss_presure)

        loss =  loss_velocity + loss_presure + loss_residual + loss_diff_pressure + loss_ux

        return loss

    def closure(self, optimizer, model):

        optimizer.zero_grad()

        loss = self.loss()

        loss.backward()

        print("Epoch: ", self.iter)

        self.iter+=1

        V_train, X_train, V_test, X_test, P_train, P_test = self.train_test_data(self.x,self.v,self.p)

        if self.iter % self.divider == 0:
            self.training_loss.append(loss)
            with torch.no_grad():
                error_vec, _ = model.test(model, X_test, V_test)
            self.error.append(error_vec)
            print(loss, error_vec)

        return loss

    def test(self, model,X_test, V_test):

        U = model.forward(X_test)
        u_pred = U[:, 0]

        error_vec = torch.linalg.norm((V_test - u_pred), 2) / torch.linalg.norm(V_test,2)  # Relative L2 Norm of the error (vector)

        #V_pred = V_pred.cpu().detach().numpy()

        return error_vec, u_pred


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

h = 6.0
#Gc = 500
U = 10.0
mu = 1.0016e-3  # 20°C water
L = 40
P = 1

preprocessing = Preprocessing_poiseuille(h, P, U, mu)
Gc = -preprocessing.pressure_drop()

x_values = np.arange(0.0, L, 0.05).tolist()

y_values = np.arange(0.0, h, 0.05).tolist()

x_values.append(L)

y_values.append(h)


x,y = np.meshgrid(x_values,y_values)


X_initial = np.hstack((x[ :,0][:, None], y[ :,0][:, None]))


X_in = np.hstack([x.flatten()[:, None], y.flatten()[:, None]])
print(X_in.shape)
vel_initial = preprocessing.velocity(X_initial)

vel_in = preprocessing.velocity(X_in)
p_in = preprocessing.pressure(X_in)


layers = np.array([2, 60, 60, 60,60,60, 2])

nu = 0.8

epochs = 1000

def main():
    model = PINN_Poisuelle(layers,nu,X_in,vel_in,Gc,p_in,mu,U, device)

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
    v_avg = torch.div(vel_initial,U)
    v_avg_pred = torch.div(v_pred,U)

    result = [v_avg, v_avg_pred, model.training_loss, model.error, error, elapsed]
    f = open('result_adam_couette_flow.pkl', 'wb')
    pickle.dump(result, f)
    f.close()


def plotting():
    file0 = open('result_NN_couette_flow.pkl', 'rb')
    data0 = pickle.load(file0)

    v_avg = data0[0]
    v_avg_pred = data0[1]
    error = data0[4]
    print(error)
    fig, ax = plt.subplots(1,1)
    ax.stem(X_initial[:,1], v_avg.cpu().detach().numpy(), orientation='horizontal',markerfmt= '>', label='U_exact')
    ax.plot(v_avg.cpu().detach().numpy(),X_initial[:,1])
    ax.annotate('Moving plate, $U=10$', xy=(-0.5, 6.1), xycoords='data', fontsize=9.7)
    ax.annotate('Stationary plate, $U=0$', xy=(-0.5, -0.2), xycoords='data', fontsize=9.7)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    ax.set_xlim(-1, 2)
    ax.set_title('$u/U (PINN)$', y=-0.1)
    ax.title.set_fontsize(20)
    ax.hlines(y=h, xmin = -U/2, xmax = 2*U, color='Black')
    ax.hlines(y=0, xmin = -U/2, xmax = 2*U, color='Black')
    #ax[1].stem(X_initial1[:,1],v_avg_pred.cpu().detach().numpy(),orientation='horizontal', markerfmt ='>', label='U_pred')
    ax.plot(v_avg_pred.cpu().detach().numpy(),X_initial[:,1],color='Orange' )
    '''ax[1].annotate('Moving plate, $U=10$', xy=(-0.5, 6.1), xycoords='data', fontsize=9.7)
    ax[1].annotate('Stationary plate, $U=0$', xy=(-0.5, -0.2), xycoords='data', fontsize=9.7)
    ax[1].axes.yaxis.set_visible(False)
    ax[1].axes.xaxis.set_visible(False)
    ax[1].set_xlim(-1, 2)
    ax[1].set_title('$u_{pred}/U$', y=-0.1)
    ax[1].title.set_fontsize(20)
    ax[1].hlines(y=h, xmin = -U/2, xmax = 2*U, color='Black')
    ax[1].hlines(y=0, xmin = -U/2, xmax = 2*U, color='Black')'''
    plt.show()

main()
#plotting()

