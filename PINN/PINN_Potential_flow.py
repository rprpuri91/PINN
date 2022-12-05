import torch
from torch import nn
from math import cos, exp, sin, sqrt
import math
import numpy as np
import time
import pickle

torch.manual_seed(1234)

np.random.seed(1234)


class Potential_flow_PINN(nn.Module):
    def __init__(self, layers, device, R, U, nu):
        super().__init__()

        self.layers = layers
        self.device = device
        self.R = R
        self.U = U

        # self.lb = lb
        # self.ub = ub

        # Activation
        self.activation = nn.Tanh()
        self.activation2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        # loss function
        self.loss_function = nn.MSELoss(reduction='mean')

        # layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        self.iter_divider = 5

        self.nu = nu

        self.error = []
        self.trainingloss = []
        self.bc_loss = []
        self.inlet_loss = []
        self.outlet_loss = []
        self.cyl_bc_loss = []
        self.domain_loss = []
        self.V_max = 0
        self.V_min  = 0

        self.iter = 0

        self.norm = False

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

            a = self.activation(z)

        a = self.activation(a)

        a = self.linears[-1](a)

        return a

    def training_test_data(self, X):

        N_u = int(self.nu * len(X))

        idx = np.random.choice(X.shape[0], N_u, replace=False)
        X_star = X[idx, :]

        X_test = np.delete(X, idx, axis=0)

        V_train = self.velocity_cartesian_vjp(X_star).float()
        V_test = self.velocity_cartesian_vjp(X_test).float()
        X_in_train = torch.from_numpy(X_star).float().to(self.device)
        X_test = torch.from_numpy(X_test).float().to(self.device)

        if self.norm is True:
            V_train = self.normalize_velocity(V_train)
            V_test = self.normalize_velocity(V_test)

        return X_in_train, X_test, V_train, V_test

    def normalize_velocity(self, V):
        V_norm = (V - self.V_min) / (self.V_max - self.V_min)

        return V_norm

    def denormalize_velocity(self,V_norm):
        V = V_norm*(self.V_max - self.V_min) + self.V_min

        return V

    def scaling(self, X):

        mean, std, var = torch.mean(X), torch.std(X), torch.var(X)
        # preprocessing input
        x = (X - mean) / (std)  # feature scaling

        return x

    # def velocity_cartesian(self,X):

    #    if torch.is_tensor(X) != True:         
    #            X = torch.from_numpy(X).to(self.device)           
    #    print(X.size())

    #    X =torch.split(X,1,dim=1)

    #    x= X[0]
    #    y= X[1]   

    #    g=(x,y)

    #    phi_x_y = torch.autograd.functional.jacobian(self.potential_fn,g)

    #    shape = list(phi_x_y[0].size())
    #    shape.pop()

    #    u  = torch.reshape(phi_x_y[0],tuple(shape))
    #    v = torch.reshape(phi_x_y[1],tuple(shape))

    #    u = torch.sum(u,2)
    #    v = torch.sum(v,2)

    #    #print(u.size())
    #    #print(v.size())

    #    V = torch.concat((u,v) ,dim=1)

    #    return V        

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

        phi, phi_x_y = torch.autograd.functional.vjp(self.potential_fn, g, v, create_graph=False)

        V_x = phi_x_y[0]
        V_y = phi_x_y[1]

        # print(V_y.size())

        V = torch.concat((V_x, V_y), dim=1)

        return V

    def potential_fn(self, x, y):

        Q = (pow((self.R), 2)) * math.pi * self.U  # Constant

        x1 = torch.square(x) + torch.square(y)

        phi = U * x + Q / math.pi * (torch.divide(x, x1))  ## Ux + Q/pi(x/(x^2+y^2))

        return phi

    def velocity_pred(self, x, y):

        g = torch.cat((x, y), dim=1)

        velocity = self.forward(g)

        # velocity = self.velocity_cartesian_vjp(g)
        #velocity = self.denormalize_velocity(velocity_norm)

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

    def loss(self, X):

        X_train, _, V_train, _ = self.training_test_data(X)

        V_pred = self.forward(X_train)

        laplace_phi, vorticity = self.laplace_potential(X_train)

        target1 = torch.zeros_like(laplace_phi, device=self.device)
        target2 = torch.zeros_like(vorticity, device=self.device)

        #loss_laplace = self.loss_function(laplace_phi, target1)
        #print(loss_laplace)

        #loss_vorticity = self.loss_function(vorticity, target2)
        #print(loss_vorticity)

        # V_train_norm = torch.nn.functional.normalize(V_train, dim=1).float()
        loss_velocity = self.loss_function(V_pred, V_train)
        #print(loss_velocity)

        loss =  loss_velocity

        return loss

    def loss_total(self):

        X_initial, X_bc, X_outlet, X_domain,X_in_domain, X_cyl_bc, indices = preprocessing()

        loss_initial = self.loss(X_initial)
        self.inlet_loss.append(loss_initial)

        loss_bc = self.loss(X_bc)
        self.bc_loss.append(loss_bc)

        loss_outlet = self.loss(X_outlet)
        self.outlet_loss.append(loss_outlet)

        loss_cyl_bc = 0.1*self.loss(X_cyl_bc)
        self.cyl_bc_loss.append(loss_cyl_bc)

        loss_domain = self.loss(X_in_domain)
        self.domain_loss.append(loss_domain)

        total_loss = loss_initial + loss_bc + loss_cyl_bc + loss_outlet + loss_domain
        # total_loss = loss_cyl_bc
        # print(type(total_loss))

        return total_loss

    def closure(self, optimizer, model, X_test, V_test):

        optimizer.zero_grad()
        loss = self.loss_total()

        loss.backward()

        self.iter += 1
        print('Epoch', self.iter)

        if self.iter % self.iter_divider == 0:
            self.trainingloss.append(loss)
            with torch.no_grad():
                error_vec, _ = model.test(model, X_test, V_test)
            self.error.append(error_vec)
            print(loss, error_vec)

        return loss

    def test(self, model, X_test, V_test):

        V_pred = model.forward(X_test)

        # V_test_norm = torch.nn.functional.normalize(V_test, dim=1).float()
        error_vec = torch.linalg.norm((V_test - V_pred), 2) / torch.linalg.norm(V_test,
                                                                                2)  # Relative L2 Norm of the error (vector)

        #V_pred = V_pred.cpu().detach().numpy()

        return error_vec, V_pred

    def tensor_compare(self, tensor_1, tensor_2):

        result = torch.eq(tensor_1, tensor_2)

        print(result)

        return result

    ################################################################Preprocessing#########################################################################################################

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

def preprocessing():
    ##Grid creation

    x_values = np.arange(-8.0, 8.0, 0.05).tolist()
    y_values = np.arange(-4.0, 4.0, 0.05).tolist()
    x_values.append(8.0)
    y_values.append(4.0)

    x, y = np.meshgrid(x_values, y_values)

    X_in = np.hstack([x.flatten()[:, None], y.flatten()[:, None]])

    X_initial = np.hstack((x[:,0][:, None], y[:,0][:, None]))

    X_bc_lower = np.hstack((x[0,:][:, None], y[0,:][:, None]))

    X_bc_upper = np.hstack((x[0,:][:, None], y[-1,:][:, None]))

    X_bc = np.vstack([X_bc_upper, X_bc_lower])

    X_outlet = np.hstack((x[:,-1][:, None], y[:,0][:, None]))

    indices_R = []
    indices_inR = []

    for i in range(len(X_in)):
        radius = sqrt(pow(X_in[i][0], 2) + pow(X_in[i][1], 2))
        # print(radius)

        if radius < R:
            # print(i)
            indices_inR.append(i)


        elif radius < (R + 0.05) and radius > (R - 0.05):
            indices_R.append(i)

    indices = indices_R + indices_inR

    X_domain = np.delete(X_in, indices_inR, axis=0)

    X_cyl_bc = np.take(X_in, indices_R, axis=0)

    X_domain2 = np.delete(X_in,indices,axis=0)

    N_x = int(0.2 * len(X_domain2))
    idx = np.random.choice(X_domain2.shape[0], N_x, replace=False)

    X_domain2 = X_domain2[idx, :]

    return X_initial, X_bc, X_outlet, X_domain,X_domain2, X_cyl_bc, indices_inR


# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

R = 2.0  ##Radius cylinder
U = 10.0  ##Velocity Inlet

layers = np.array([2, 60, 60, 60,60,60, 2])

nu = 0.8

epochs = 5000

X_initial, X_bc, X_outlet, X_domain,_, X_cyl_bc, indices = preprocessing()

X_cyl = sort_points(X_cyl_bc)

X_train_test = np.concatenate((X_initial, X_bc, X_outlet, X_cyl_bc))


model = Potential_flow_PINN(layers, device, R, U, nu)
print('model created')

model.to(device)

V_domain = model.velocity_cartesian_vjp(X_domain)

model.V_max = torch.max(V_domain)
model.V_min = torch.min(V_domain)


model.norm = True
X_in_train, X_test, V_train, V_test = model.training_test_data(X_train_test)

optimizerA = torch.optim.Adam(model.parameters(), lr=0.001)
optimizerB = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()

for epoch in range(0, epochs):
    optimizerA.step(lambda: model.closure(optimizerA, model, X_test, V_test))

# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter = epochs, max_eval = None, tolerance_grad = 1e-05, tolerance_change=1e-09, history_size=100, line_search_fn='strong_wolfe')

# optimizer.step(lambda: model.closure(optimizer,model,X_test,V_test))

elapsed = time.time() - start_time

print('Training time: %.2f' % (elapsed))


# print(V_domain.size())
# V_domain_norm = torch.nn.functional.normalize(V_domain)
V_domain_norm = model.normalize_velocity(V_domain)
# print(V_domain_norm.size())
X_domain = torch.from_numpy(X_domain).float().to(device)

error_vec, V_pred_norm = model.test(model, X_domain, V_domain_norm)
V_pred = model.denormalize_velocity(V_pred_norm)
print(error_vec)

result = [V_pred_norm, V_domain_norm, model.error, model.trainingloss, indices, model.bc_loss, model.inlet_loss,
          model.outlet_loss, model.cyl_bc_loss, model.domain_loss, V_pred, V_domain, X_cyl]
f = open('result_potential_flow.pkl', 'wb')
pickle.dump(result, f)
f.close()
