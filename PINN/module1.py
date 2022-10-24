
from math import cos, exp, sin,sqrt
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import pickle
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.signal import lfilter
import torchshow as ts
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import math


#device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')  


#def funct(x,y):
#    y = 2*torch.sin(x)*torch.cos(y) + x*y 
#    return y
 
#x = 1.58e-4
#a = torch.randn(1)
#b= torch.randn(1)
#g = (a,b)
#y1 = torch.autograd.functional.jacobian(funct,g)
#y2 = 2*torch.cos(a)*torch.cos(b) +b
#y3 = -2*torch.sin(a)*torch.sin(b) +a

#x1 = np.arange(9.0)
#x2 = np.arange(9.0)
#x3 =np.multiply(x1, x2)

#c = torch.empty(9,1)
#d = torch.randn(9,1)
##d1 = torch.randn(2,20)
##e = torch.matmul(c, d)
##f = torch.square(d)
##g = torch.sum(f, dim=0)
##g = g.repeat(9,1)
##h = torch.randn(9,1)
##j = torch.randn(9,1)
#i = torch.mul(c,d)
#y = y2+y3
#print(c)
#print(d)
#print(torch.cat((c,d),1))

file = open('result.pkl', 'rb')
data = pickle.load(file)


file = open('total_data.pkl', 'rb')
in_data = pickle.load(file)

#print(data[-1])
error = data[-1]
loss = data[-2]
f_pred = data[0]
f_exact= data[2]
u_pred=data[3]
residual = data[4]
u=data[5]
fpred2= f_pred[:,1]
fexact2=f_exact[:,1]
f_eq_pred= data[1]
x_in_train = in_data[0]
#print(residual[1].shape)

x1 = x_in_train[:12500,0]
y1 = x_in_train[:12500,1]

#x = torch.reshape(x1,(125,100)).detach().numpy()
#y = torch.reshape(y1,(125,100)).detach().numpy()
res = residual[1]
print(res)
res = torch.reshape(res,(172,101)).cpu().detach().numpy()

f_pred = f_pred[:,2]

f_pred = torch.reshape(f_pred,(172,101)).cpu().detach().numpy()

print(res)





#u_pred_u = u_pred[:,0]
#u_pred_v = u_pred[:,1]

#u_exact_u = u[:,0]
#u_exact_v = u[:,1]


#print(fpred2.shape)


x_values = np.arange(-0.5,2.0,0.0147).tolist()
y_values = np.arange(-0.5,1.5,0.0200).tolist()
t_values = np.arange(0.0,2.0,0.0118).tolist()
x_values.append(2.000)
y_values.append(1.500)
#x1,t = np.meshgrid(x_values,t_values)
y0,x0= np.meshgrid(y_values,x_values)

print(y0)


epochs = 5000

xmax = int(epochs/5)

x1 = [*range(1,xmax+1)]

n = 15  # the larger n is, the smoother curve will be
n2 = 30
b2 = [1.0 / n2] * n2
b = [1.0 / n] * n
a = 1
filt_error = lfilter(b,a,error)
filt_error2 = lfilter(b2,a,error)
filt_loss = lfilter(b,a,loss)

#filt_u_pred = lfilter(b,a,u_pred_u.cpu())
#filt_v_pred = lfilter(b,a,u_pred_v.cpu())  
#filt_u_exact = lfilter(b,a,u_exact_u.cpu())
#filt_v_exact = lfilter(b,a,u_exact_v.cpu())
##plt.plot(filt_u_pred)
##plt.plot(filt_u_exact)
##print(u_pred.shape)



#m = x.shape[0]
#n = y.shape[1]

##print(f_pred.shape)

#fpred2 = torch.reshape(fpred2,(m,n))
#fexact2 = torch.reshape(fexact2,(m,n))

def xi_values():
    RT = 100
    c = sqrt(3*RT)
    e0 = [0,0]
    e1 = [1,0]
    e2 = [0,1]
    e3 = [-1,0]
    e4 = [0,-1]
    e5 = [1,1]
    e6 = [-1,1]
    e7 = [-1,-1]
    e8 = [1,-1]

    e = [e0,e1,e2,e3,e4,e5,e6,e7,e8]

    xi = []

    for ele in e:
        xi.append([x * c for x in ele])

    return xi

def pred_variables(f_pred):

    xi = xi_values()
    xi1 = np.array(xi)[:,0]
    xi2 = np.array(xi)[:,1]
        
    xi1 = torch.from_numpy(xi1).float().to(device)
    xi2 = torch.from_numpy(xi2).float().to(device)

    size = f_pred.size()[0]
    #row = torch.zeros(size,device=device)
    #xi_f = torch.zeros(size,2,device=device)
    u = torch.zeros(size,2,device=device)
    #print('empty',u)
       
    for i in range(size):
        row_i = sum(f_pred[i])
        #row_i = row_i.item()
        #row[i]=row_i
        ui0 = sum(torch.mul(xi1,f_pred[i].to(device)))
        #ui0 = ui0.item()
        vi0 = sum(torch.mul(xi2,f_pred[i].to(device)))
        #vi0 = vi0.item()
        if row_i == 0.0:
            ui=0.0
            vi=0.0
        else:
            ui = torch.div(ui0,row_i)            
            vi = torch.div(vi0,row_i)
            
        u_temp = torch.tensor([ui,vi],device=device)            
        u[i] = u_temp   
        #xi_f_temp = torch.tensor([ui0,vi0],device=device)
        #xi_f[i] = xi_f_temp
        #print(u)
    return u


u_eq = pred_variables(f_eq_pred)
#fexact2= fexact2[:-1,:-1]
print('vel',u_eq)

u_eq_u = u_eq[:,0]
u_eq_v = u_eq[:,1]


#error_u = torch.linalg.norm((u_exact_v - u_eq_v),2)/torch.linalg.norm(u_exact_v,2)

#print(error_u)

fe2_min,fe2_max = abs(fexact2).min(), abs(fexact2).max()
fp2_min,fp2_max = abs(fpred2).min(), abs(fpred2).max()

levels = MaxNLocator(nbins=15).tick_values(fe2_min.cpu(),fe2_max.cpu())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
#norm=colors.LogNorm(vmin=fe2_min, vmax=fe2_max)
fig,ax = plt.subplots(1,2)
ax[0].plot(x1,error, label='f error')
#ax[0][0].set_yscale('log')
ax[0].set_yticks((0,5,10))
ax[0].set_xticks((0,500,1000,1500,2000))
ax[0].set_xticklabels((0,2500,5000,7500,10000), fontsize=20)
ax[0].set_ylabel('Relative L2 Error')
ax[0].set_xlabel('Iterations')
ax[0].xaxis.label.set_fontsize(25)
ax[0].yaxis.label.set_fontsize(25)
ax[0].grid()
ax[0].legend()


    
ax[1].plot(x1,filt_error2, label='f error')
ax[1].grid()
ax[1].legend()
ax[1].set_ylim(0,0.5)
ax[1].set_yticks((0,0.02,0.1,0.2,0.3,0.4,0.5))
ax[1].set_xticks((0,500,1000,1500,2000))
ax[1].set_xticklabels((0,2500,5000,7500,10000), fontsize=20)
ax[1].set_ylabel('Relative L2 Error')
ax[1].set_xlabel('Iterations')
ax[1].xaxis.label.set_fontsize(25)
ax[1].yaxis.label.set_fontsize(25)

plt.setp(ax[0].get_yticklabels(), Fontsize=20)
plt.setp(ax[1].get_yticklabels(), Fontsize=20)




f_max,f_min= f_pred.max(),f_pred.min()
res_max, res_min = res.max(), res.min()

norm = colors.Normalize(vmin=res_min, vmax= res_max)

print(f_max,f_min)

filt_res = lfilter(b2,a,res)


fig,ax = plt.subplots(1,2)
c1=ax[0].pcolormesh(x0,y0, res, shading = 'gouraud', label='residual', vmin=res_min, vmax= res_max, cmap=plt.get_cmap('rainbow'))
fig.colorbar(c1, ax=ax[0])

c2=ax[1].pcolormesh(x0,y0, f_pred, shading = 'gouraud', label='f_pred', vmin=f_min, vmax= f_max)
fig.colorbar(c2, ax=ax[1])
plt.show()
