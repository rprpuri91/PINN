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
from scipy.signal import lfilter
import tikzplotlib as mt

torch.manual_seed(1234)

np.random.seed(1234)
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
        return torch.reshape(vel, (vel.shape[0],1))

    def pressure(self,X):
        x = X[:,0]
        p = []
        dpdx = -self.pressure_drop()
        for i in range(len(x)):
            p_i = dpdx*x[i]
            p.append(p_i)

        p = np.array(p)
        p_tensor = torch.from_numpy(p).float().to(device)
        return torch.reshape(p_tensor, (p_tensor.shape[0],1))


class PINN_Poisuelle(nn.Module):
    def __init__(self, layers,nu,X_domain,U_domain,X_wall,U_wall,G,mu, device):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.X_domain = X_domain
        self.X_wall = X_wall
        self.mu = mu
        self.U_domain = U_domain
        self.U_wall = U_wall
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

        self.V_max = self.U_domain.max()
        self.V_min = self.U_domain.min()
        '''for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain =5/3)

            nn.init.zeros_(self.linears[i].bias.data)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'''

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

        return V_train, X_train, V_test, X_test

    def variable_pred(self, x, r):
        g = torch.cat((x, r), dim=1)

        U = self.forward(g)

        return U

    def variable_pred_hessian(self, x,r):
        g = torch.cat((x, r), dim=1)

        U = self.forward(g).sum()
        return U

    def governing_equation(self, X):

        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X).to(self.device)

        v3 = torch.zeros(X.shape[0],1).to(self.device)
        v4 = torch.ones(X.shape[0],1).to(self.device)
        g = torch.clone(X)
        g.requires_grad = True

        v1 = torch.zeros_like(X, device=self.device)
        v2 = torch.zeros_like(X, device=self.device)

        v1[:, 0] = 1
        v2[:, 1] = 1

        X = torch.split(X, 1, dim=1)

        x = X[0]
        r = X[1]

        U, U_x_r = torch.autograd.functional.vjp(self.variable_pred, (x, r), v1, create_graph=True)
        '''U_x_r = torch.autograd.grad(U, g, torch.ones([X.shape[0], 2]).to(self.device), retain_graph=True,
                                    create_graph=True)[0]'''
        #U_x = U_x_r[:,0]
        #U_r = U_x_r[:,1]
        u_x = U_x_r[0]
        u_r = U_x_r[1]

        U1, P_x_r = torch.autograd.functional.vjp(self.variable_pred, (x, r), v2, create_graph=True)
        p_x = P_x_r[0]
        p_r = P_x_r[1]

        #u_xx_rr = torch.autograd.grad(u_r, g,v , create_graph=True)[0]
        #u_rr = u_xx_rr[:, 1]
        C, H = torch.autograd.functional.vhp(self.variable_pred_hessian, (x, r), (v3, v4), create_graph=True)
        u_rr = H[1]

        '''print(X[:,1])
        print(torch.div(u_r,X[:,1]))'''
        res = p_x - self.mu * (u_rr + torch.div(u_r, r))
        #print('res', res.shape)
        return U, u_x, res, p_r

    def loss(self,x,v):

        V_train, X_train, V_test, X_test= self.train_test_data(x, v)

        U, res, u_x, p_r = self.governing_equation(X_train)

        target1 = torch.zeros_like(res, device=self.device)
        target2 = torch.zeros_like(u_x, device=self.device)
        target3 = torch.zeros_like(p_r, device=self.device)


        loss_residual = self.loss_function(res, target1)
        #print('residual', loss_residual)

        loss_diff_pressure = self.loss_function(p_r,target3)
        #print('diff_pressure',loss_diff_pressure)

        loss_ux = self.loss_function(u_x, target2)
        #print('loss ux',loss_ux)

        loss_variable = self.loss_function(U, V_train)
        #print('velocity', loss_velocity)


        loss =  loss_variable  +loss_residual + loss_diff_pressure + 0.001*loss_ux

        return loss

    def total_loss(self):

        loss1 = self.loss(self.X_domain,self.U_domain)
        loss2 = self.loss(self.X_wall, self.U_wall)

        loss = loss1+ loss2

        return loss

    def closure(self, optimizer, model):

        optimizer.zero_grad()

        loss = self.total_loss()

        loss.backward()

        print("Epoch: ", self.iter)

        self.iter += 1

        V_train, X_train, V_test, X_test = self.train_test_data(self.X_domain, self.U_domain)

        if self.iter % self.divider == 0:
            self.training_loss.append(loss)
            with torch.no_grad():
                error_vec, _ = model.test(model, X_test, V_test)
            self.error.append(error_vec)
            print(loss, error_vec)

        return loss

    def test(self, model, X_test, V_test):

        U = model.forward(X_test)


        error_vec = torch.linalg.norm((V_test - U), 2) / torch.linalg.norm(V_test,2)  # Relative L2 Norm of the error (vector)
        # V_pred = V_pred.cpu().detach().numpy()

        return error_vec, U


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

R = 1
L = 40
Q = 0.00314
mu = 0.0010016  # 20??C water
rho = 1000
Re = 2000 # rho*Q*D/(mu*A) = 2*rho*Q*R/(mu*pi*R*R) = 2000*Q/(mu*pi*R) => Q = mu*pi*R => Q = 0.00314*R

# U = Q/(np.pi*(R^2))

preprocessing = Preprocessing_poiseuille(R, L, Q, mu)
G = preprocessing.pressure_drop()/L
x_values = np.arange(0.0, L, 0.01).tolist()
r_values = np.arange(-R, R, 0.05).tolist()
x_values.append(L)
r_values.append(R)

x, r = np.meshgrid(x_values, r_values)

X_in = np.hstack([x.flatten()[:, None], r.flatten()[:, None]])
print(X_in.shape)
X_wall_lower = np.hstack((x[0,:][:, None], r[0,:][:, None]))

X_wall_upper = np.hstack((x[0,:][:, None], r[-1,:][:, None]))

X_wall = np.vstack([X_wall_upper, X_wall_lower])

X_initial = np.hstack((x[:, 0][:, None], r[:, 0][:, None]))
vel_initial = preprocessing.velocity(X_initial)
p_initial = preprocessing.pressure(X_initial)
U_initial = torch.cat((vel_initial,p_initial), dim=1)

vel_wall = preprocessing.velocity(X_wall)
p_wall = preprocessing.pressure(X_wall)

U_wall = torch.cat((vel_wall,p_wall),dim=1)

N_x = int(0.2 * len(X_in))
idx = np.random.choice(X_in.shape[0], N_x, replace=False)

X_domain = X_in[idx, :]

vel_domain = preprocessing.velocity(X_domain)
p_domain = preprocessing.pressure(X_domain)

U_domain = torch.cat((vel_domain,p_domain),dim=1)

layers = np.array([2, 60, 60, 60,60,60, 2])

nu = 0.8

epochs = 5000
def main():

    model = PINN_Poisuelle(layers,nu,X_domain,U_domain,X_wall,U_wall,G,mu, device)

    model.to(device)

    start_time = time.time()

    optimizerA = torch.optim.Adam(model.parameters(), lr=0.005)
    optimizerB = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    optimizerC = torch.optim.LBFGS(model.parameters(), lr=0.001,
                                  max_iter=5000,
                                  max_eval=None,
                                  #tolerance_grad=1e-05,
                                  #tolerance_change=1e-09,
                                  #history_size=100,
                                  line_search_fn='strong_wolfe')

    model.train()

    #optimizerC.step(lambda: model.closure(optimizerC, model))

    for e in range(0,epochs):
        optimizerA.step(lambda: model.closure(optimizerA, model))

    elapsed = time.time() - start_time

    print('Training time: %.2f' % (elapsed))

    U_norm = model.normalize_velocity(U_initial)
    error, U_pred_norm = model.test(model, X_initial, U_norm)

    #v_pred = model.denormalize_velocity(U_pred_norm)
    U = Q/(np.pi*(R**2))

    v_pred_norm = U_pred_norm[:,0]
    v_pred = v_pred_norm * (vel_initial.max() - vel_initial.min()) + vel_initial.min()
    v_avg = torch.div(vel_initial,U)
    v_avg_pred = torch.div(v_pred,U)

    result = [v_avg,v_avg_pred, model.training_loss, model.error, error, elapsed]
    f = open('result_adam_poisueille_flow.pkl', 'wb')
    pickle.dump(result, f)
    f.close()



def plotting():
    file0 = open('result_sgd_poisueille_flow.pkl', 'rb')
    data0 = pickle.load(file0)
    file1 = open('result_NN_poisueille_flow.pkl', 'rb')
    data1 = pickle.load(file1)
    file2 = open('result_adam_poisueille_flow.pkl', 'rb')
    data2 = pickle.load(file2)

    v_avg = data0[0]
    v_avg_sgd_pred = data0[1]
    v_avg_NN_pred = data1[1]
    v_avg_adam_pred = data2[1]
    error0 = data0[3]
    error1 = data1[3]
    error2 = data2[3]

    errorA = data0[-1]
    errorB = data1[-1]

    print('PINN',errorA)
    print('DNN',errorB)

    print(error0)
    fig, ax = plt.subplots(1, 1)
    ax.plot(v_avg.cpu().detach().numpy(), X_initial[:,1], label='U_exact')
    ax.hlines(y=-R, xmin=-1, xmax=5, color='black')
    ax.hlines(y=R, xmin=-1, xmax=5, color='black')
    ax.set_xlabel('$u_x$/U', rotation=0)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    #ax.annotate('------------- $R_p$ -------------->', xy=(0.0, 0.0), xycoords='data', fontsize=9.7, rotation=90)
    #ax.annotate('Pipe wall, $u_x = 0$', xy=(0.0, R+0.01), xycoords='data', fontsize=9.7)
    #ax.annotate('Pipe wall, $u_x = 0$', xy=(0.0, -R-0.04 ), xycoords='data', fontsize=9.7)
    #ax.annotate('--->', xy=(0.0, 0.0), xycoords='data', fontsize=15)
    ax.set_xlim(0, 2.2)
    ax.set_xticks((0,0.5,1,1.5,2))
    ax.set_xticklabels((0,0.5,1,1.5,2))
    ax.scatter(v_avg_sgd_pred.cpu().detach().numpy(), X_initial[:,1], label='U_pred_sgd_PINN', marker='s', color='Orange')
    ax.scatter(v_avg_NN_pred.cpu().detach().numpy(), X_initial[:, 1], label='U_pred_adam_DNN', marker='.', color='Black')
    #ax.scatter(v_avg_adam_pred.cpu().detach().numpy(), X_initial[:, 1], label='U_pred_adam_PINN', marker='x', color='Red')
    #ax.set_title('$u_x/U$', y=-0.1)
    ax.legend(loc='best')
    ax.axis('off')
    #ax[1].hlines(y=-R, xmin=-1, xmax=5, color='black')
    #ax[1].hlines(y=R, xmin=-1, xmax=5, color='black')
    #ax[1].set_xlabel('$u_x$/U', rotation=0)
    #ax[1].axes.yaxis.set_visible(False)
    #ax[1].set_xlim(-1, 5)

    #######################################
    error0_cpu = torch.tensor(error0, device=device)
    error1_cpu = torch.tensor(error1, device=device)
    error2_cpu = torch.tensor(error2, device=device)

    n = 30
    b = [1.0 / n] * n
    a =1

    e1 = lfilter(b,a,error0_cpu)
    e2 = lfilter(b, a, error1_cpu)
    e3 = lfilter(b, a, error2_cpu)
    #error_loss_plot(e1,e2,e3)
    #mt.save('poiseuille_result.tex')

    plt.show()

def error_loss_plot(e1,e2,e3):
    e= int(epochs/5)
    xmax = int(e)

    x1 = [*range(1, xmax + 1)]
    fig,ax = plt.subplots(1,1)
    ax.plot(x1,e1, label='PINN')
    ax.plot(x1, e2, label='DNN')
    #ax.plot(x1, e3, label='PINN with ADAM')
    #ax[0][0].set_yscale('log')
    ax.set_ylim(1,2)
    #ax.set_yticks((0,0.01,0.02,0.03,0.04,0.05,0.1))
    ax.set_xlim(e/5,e)
    ax.set_xticks((e/5,2*e/5,3*e/5,4*e/5,e))
    ax.set_xticklabels((epochs/5,2*epochs/5,3*epochs/5,4*epochs/5,epochs), fontsize=10)
    ax.set_ylabel('$\epsilon$')
    ax.set_xlabel('Epochs')
    #ax.xaxis.label.set_fontsize(15)
    #ax.yaxis.label.set_fontsize(15)
    ax.grid()
    #ax.legend()
    mt.save('poiseuille_error.tex')



plotting()
#main()