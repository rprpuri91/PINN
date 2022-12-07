
from math import cos, exp, sin,sqrt
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import pickle
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#from scipy.signal import lfilter

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import math



#class BGK_boltzmann():
#    def __init__(self):
    

#        #print(torch.version.cuda)

#        self.device = 'cpu'
#        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
#        print(f'Using {device} device')  


#        #def funct(x,y):
#        #    y = 2*torch.sin(x)*torch.cos(y) + x*y 
#        #    return y
 
#        #x = 1.58e-4
#        #a = torch.randn(1)
#        #b= torch.randn(1)
#        #g = (a,b)
#        #y1 = torch.autograd.functional.jacobian(funct,g)
#        #y2 = 2*torch.cos(a)*torch.cos(b) +b
#        #y3 = -2*torch.sin(a)*torch.sin(b) +a

#        #x1 = np.arange(9.0)
#        #x2 = np.arange(9.0)
#        #x3 =np.multiply(x1, x2)

#        #c = torch.empty(9,1)
#        #d = torch.randn(9,1)
#        ##d1 = torch.randn(2,20)
#        ##e = torch.matmul(c, d)
#        ##f = torch.square(d)
#        ##g = torch.sum(f, dim=0)
#        ##g = g.repeat(9,1)
#        ##h = torch.randn(9,1)
#        ##j = torch.randn(9,1)
#        #i = torch.mul(c,d)
#        #y = y2+y3
#        #print(c)
#        #print(d)
#        #print(torch.cat((c,d),1))

#        file = open('result.pkl', 'rb')
#        data = pickle.load(file)


#        file = open('total_data.pkl', 'rb')
#        in_data = pickle.load(file)

#        #print(data[-1])
#        error = data[7]
#        residual_error = data[8]
#        ic_error = data[9]
#        bc_error = data[10]
#        res_err_cpu = torch.tensor(residual_error,device='cpu')
#        ic_error_cpu = torch.tensor(ic_error,device='cpu')
#        bc_error_cpu = torch.tensor(bc_error,device='cpu')
#        error_cpu = torch.tensor(error,device='cpu')
#        loss = data[6]
#        f_pred = data[0]
#        f_exact= data[2]
#        u_pred=data[3]
#        residual = data[4]
#        u=data[5]
#        fpred2= f_pred[:,1]
#        fexact2=f_exact[:,1]
#        f_eq_pred= data[1]
#        f_eq_pred2 = f_eq_pred[:,1]
#        x_in_train = in_data[0]
#        print('residual',residual)

#        x1 = x_in_train[:12500,0]
#        y1 = x_in_train[:12500,1]

#        #x = torch.reshape(x1,(125,100)).detach().numpy()
#        #y = torch.reshape(y1,(125,100)).detach().numpy()
#        res2 = residual[2]
#        print(res2)
#        res2 = torch.reshape(res2,(172,101)).cpu().detach().numpy()

#        u_x0 = u_pred[:,0]
#        u_x = torch.reshape(u_x0,(172,101)).cpu().detach().numpy()

#        u_y0 = u_pred[:,1]
#        u_y = torch.reshape(u_y0,(172,101)).cpu().detach().numpy()

#        u_X0 = u[:,0]
#        u_X = torch.reshape(u_X0,(172,101)).cpu().detach().numpy()

#        u_Y0 = u[:,1]
#        u_Y = torch.reshape(u_Y0,(172,101)).cpu().detach().numpy()


#        res5 = residual[5]
#        print(res5)
#        res5 = torch.reshape(res5,(172,101)).cpu().detach().numpy()

#        f_pred2 = torch.reshape(fpred2,(172,101)).cpu().detach().numpy()
#        f_exact2 = torch.reshape(fexact2,(172,101)).cpu().detach().numpy()
#        f_eq_pred2 = torch.reshape(f_eq_pred2,(172,101)).cpu().detach().numpy()

#        print("u shape",u.shape)
#        print("F shape",f_pred.shape)





#        #u_pred_u = u_pred[:,0]
#        #u_pred_v = u_pred[:,1]

#        #u_exact_u = u[:,0]
#        #u_exact_v = u[:,1]


#        #print(fpred2.shape)


#        x_values = np.arange(-0.5,2.0,0.0147).tolist()
#        y_values = np.arange(-0.5,1.5,0.0200).tolist()
#        t_values = np.arange(0.0,2.0,0.0118).tolist()
#        x_values.append(2.000)
#        y_values.append(1.500)
#        #x1,t = np.meshgrid(x_values,t_values)
#        y0,x0= np.meshgrid(y_values,x_values)

#        print(y0)


#        epochs = 7000

#        xmax = int(epochs/5)

#        x1 = [*range(1,xmax+1)]

#        n = 15  # the larger n is, the smoother curve will be
#        n2 = 30
#        b2 = [1.0 / n2] * n2
#        b = [1.0 / n] * n
#        a = 1
#        filt_error = lfilter(b,a,error_cpu)
#        filt_error2 = lfilter(b2,a,error_cpu)
#        #filt_loss = lfilter(b,a,loss)

#        #filt_u_pred = lfilter(b,a,u_pred_u.cpu())
#        #filt_v_pred = lfilter(b,a,u_pred_v.cpu())  
#        #filt_u_exact = lfilter(b,a,u_exact_u.cpu())
#        #filt_v_exact = lfilter(b,a,u_exact_v.cpu())
#        ##plt.plot(filt_u_pred)
#        ##plt.plot(filt_u_exact)
#        ##print(u_pred.shape)



#        #m = x.shape[0]
#        #n = y.shape[1]

#        ##print(f_pred.shape)

#        #fpred2 = torch.reshape(fpred2,(m,n))
#        #fexact2 = torch.reshape(fexact2,(m,n))

#    def xi_values():
#        RT = 100
#        c = sqrt(3*RT)
#        e0 = [0,0]
#        e1 = [1,0]
#        e2 = [0,1]
#        e3 = [-1,0]
#        e4 = [0,-1]
#        e5 = [1,1]
#        e6 = [-1,1]
#        e7 = [-1,-1]
#        e8 = [1,-1]

#        e = [e0,e1,e2,e3,e4,e5,e6,e7,e8]

#        xi = []

#        for ele in e:
#            xi.append([x * c for x in ele])

#        return xi

#    def pred_variables(f_pred):

#        xi = xi_values()
#        xi1 = np.array(xi)[:,0]
#        xi2 = np.array(xi)[:,1]
        
#        xi1 = torch.from_numpy(xi1).float().to(device)
#        xi2 = torch.from_numpy(xi2).float().to(device)

#        size = f_pred.size()[0]
#        row = torch.zeros(size,device=device)
#        #xi_f = torch.zeros(size,2,device=device)
#        u = torch.zeros(size,2,device=device)
#        #print('empty',u)
       
#        for i in range(size):
#            row_i = sum(f_pred[i])
#            row_i = row_i.item()
#            row[i]=row_i
#            ui0 = sum(torch.mul(xi1,f_pred[i].to(device)))
#            #ui0 = ui0.item()
#            vi0 = sum(torch.mul(xi2,f_pred[i].to(device)))
#            #vi0 = vi0.item()
#            if row_i == 0.0:
#                ui=0.0
#                vi=0.0
#            else:
#                ui = torch.div(ui0,row_i)            
#                vi = torch.div(vi0,row_i)
            
#            u_temp = torch.tensor([ui,vi],device=device)            
#            u[i] = u_temp   
#            #xi_f_temp = torch.tensor([ui0,vi0],device=device)
#            #xi_f[i] = xi_f_temp
#            #print(u)
#        return u,row


#    u_eq,row_pred0 = pred_variables(f_pred)
#    #fexact2= fexact2[:-1,:-1]
#    print('row_pred',row_pred0)

#    u_eqx,row_exact0 = pred_variables(f_exact)
#    #fexact2= fexact2[:-1,:-1]
#    print('row_pred',row_exact0)

#    u_eq_x0 = u_eq[:,0]
#    u_eq_x = torch.reshape(u_eq_x0,(172,101)).cpu().detach().numpy()

#    u_eq_y0 = u_eq[:,1]
#    u_eq_y = torch.reshape(u_eq_y0,(172,101)).cpu().detach().numpy()

#    u_eq_xx0 = u_eqx[:,0]
#    u_eq_xx = torch.reshape(u_eq_xx0,(172,101)).cpu().detach().numpy()

#    u_eq_yx0 = u_eqx[:,1]
#    u_eq_yx = torch.reshape(u_eq_yx0,(172,101)).cpu().detach().numpy()

#    row_pred = torch.reshape(row_pred0,(172,101)).cpu().detach().numpy()

#    row_exact = torch.reshape(row_exact0,(172,101)).cpu().detach().numpy()


#    error_v = torch.linalg.norm((u_Y0.cpu() - u_eq_y0),2)/torch.linalg.norm(u_Y0.cpu(),2)

#    print('y error',error_v)

#    error_u = torch.linalg.norm((u_X0.cpu() - u_eq_x0),2)/torch.linalg.norm(u_X0.cpu(),2)

#    print('X error',error_u)

#    error_row = torch.linalg.norm((row_exact0 - row_pred0),2)/torch.linalg.norm(row_exact0,2)

#    print('row error',error_row)

#    error_f = torch.linalg.norm((f_exact - f_pred),2)/torch.linalg.norm(f_exact,2)
#    print('f error',error_f)



#    # pick the desired colormap, sensible levels, and define a normalization
#    # instance which takes data values and translates those into levels.
#    #norm=colors.LogNorm(vmin=fe2_min, vmax=fe2_max)

#    filt_resi_err = lfilter(b2,a,res_err_cpu)
#    filt_ic_error = lfilter(b2,a,ic_error_cpu)
#    filt_bc_error = lfilter(b2,a,bc_error_cpu)

#    fig,axe =plt.subplots(1,1)
#    axe.plot(filt_resi_err, label='residual')
#    axe.plot(filt_ic_error, label='ic')
#    axe.plot(filt_bc_error, label='bc')
#    axe.set_xlim(4000,7000)
#    axe.set_ylim(0,1)
#    axe.set_yticks((0,0.1,0.2,0.3,0.4,0.5,0.8,1))
#    #axe.set_xticks((0,500,1000,1500,2000))
#    #axe.set_xticklabels((0,2500,5000,7500,10000), fontsize=20)
#    axe.set_ylabel('Relative L2 Error')
#    axe.set_xlabel('Iterations')
#    axe.xaxis.label.set_fontsize(25)
#    axe.yaxis.label.set_fontsize(25)
#    axe.grid()
#    axe.legend()


#    fig,ax = plt.subplots(1,2)
#    ax[0].plot(x1,error_cpu, label='f error')
#    #ax[0][0].set_yscale('log')
#    ax[0].set_yticks((0,5,10))
#    ax[0].set_xticks((0,500,1000,1500,2000))
#    ax[0].set_xticklabels((0,2500,5000,7500,10000), fontsize=20)
#    ax[0].set_ylabel('Relative L2 Error')
#    ax[0].set_xlabel('Iterations')
#    ax[0].xaxis.label.set_fontsize(25)
#    ax[0].yaxis.label.set_fontsize(25)
#    ax[0].grid()
#    ax[0].legend()


    
#    ax[1].plot(x1,filt_error2, label='f error')
#    ax[1].grid()
#    ax[1].legend()
#    ax[1].set_ylim(0,0.4)
#    ax[1].set_xlim(1500,2000)
#    ax[1].set_yticks((0,0.02,0.03,0.1,0.2,0.3,0.4))
#    ax[1].set_xticks((1500,1750,2000))
#    ax[1].set_xticklabels((7500,8750,10000), fontsize=20)
#    ax[1].set_ylabel('Relative L2 Error')
#    ax[1].set_xlabel('Iterations')
#    ax[1].xaxis.label.set_fontsize(25)
#    ax[1].yaxis.label.set_fontsize(25)

#    #plt.setp(ax[0].get_yticklabels(), Fontsize=20)
#    #plt.setp(ax[1].get_yticklabels(), Fontsize=20)


#    res_max2, res_min2 = res2.max(), res2.min()
#    res_max5, res_min5 = res5.max(), res5.min()

#    f_pred2_max, f_pred2_min = f_pred2.max(), f_pred2.min()
#    f_exact2_max, f_exact2_min = f_exact2.max(), f_exact2.min()
#    f_eq_pred2_max, f_eq_pred2_min = f_eq_pred2.max(), f_eq_pred2.min()

#    fig,ax1 = plt.subplots(2,1)
#    c1=ax1[0].pcolormesh(x0,y0, f_eq_pred2, shading = 'gouraud', label='f_pred', vmin=f_eq_pred2_min, vmax= f_eq_pred2_max, cmap=plt.get_cmap('rainbow'))
#    fig.colorbar(c1, ax=ax1[0])

#    c2=ax1[1].pcolormesh(x0,y0, f_exact2, shading = 'gouraud', label='f', vmin=f_exact2_min, vmax= f_exact2_max, cmap=plt.get_cmap('rainbow'))
#    fig.colorbar(c2, ax=ax1[1])


#    fig,ax2 = plt.subplots(2,1)
#    c1=ax2[0].pcolormesh(x0,y0, res2, shading = 'gouraud', label='residual', vmin=res_min2, vmax= res_max2, cmap=plt.get_cmap('rainbow'))
#    fig.colorbar(c1, ax=ax2[0])
#    ax2[0].axes.xaxis.set_visible(False)
#    ax2[0].axes.yaxis.set_visible(False)

#    c2=ax2[1].pcolormesh(x0,y0, res5, shading = 'gouraud', label='residual', vmin=res_min5, vmax= res_max5, cmap=plt.get_cmap('rainbow'))
#    fig.colorbar(c2, ax=ax2[1])
#    ax2[1].axes.xaxis.set_visible(False)
#    ax2[1].axes.yaxis.set_visible(False)

#    u_x_max,u_x_min= u_x.max(), u_x.min()
#    u_y_max,u_y_min= u_y.max(), u_y.min()

#    u_X_max,u_X_min= u_X.max(), u_X.min()
#    u_Y_max,u_Y_min= u_Y.max(), u_Y.min()

#    row_pred_max,row_pred_min = row_pred.max(), row_pred.min()
#    row_exact_max,row_exact_min = row_exact.max(), row_exact.min()

#    #fig,ax = plt.subplots(3,1)
#    #ax[0].plot(u_X0.cpu(), color='red')
#    #ax[0].plot(u_x0.cpu(), color='blue')

#    #ax[1].plot(u_Y0.cpu(), color='red')
#    #ax[1].plot(u_y0.cpu(), color='blue')

#    #ax[2].plot(row_exact0.cpu(), color='red')
#    #ax[2].plot(row_pred0.cpu(), color='blue')


#    fig,ax = plt.subplots(3,2)
#    c1=ax[0][0].pcolormesh(x0,y0, u_eq_x, shading = 'gouraud', label='u_x_pred', vmin=u_x_min, vmax= u_x_max, cmap=plt.get_cmap('rainbow'))
#    fig.colorbar(c1, ax=ax[0][0])
#    ax[0][0].set_title('u_pred', y=-0.1)
#    ax[0][0].axes.xaxis.set_visible(False)
#    ax[0][0].axes.yaxis.set_visible(False)

#    c2=ax[0][1].pcolormesh(x0,y0, u_eq_xx, shading = 'gouraud', label='u_x_exact', vmin=u_X_min, vmax= u_X_max, cmap=plt.get_cmap('rainbow'))
#    fig.colorbar(c2, ax=ax[0][1])
#    ax[0][1].set_title('u_exact', y=-0.1)
#    ax[0][1].axes.xaxis.set_visible(False)
#    ax[0][1].axes.yaxis.set_visible(False)

#    c1=ax[1][0].pcolormesh(x0,y0, u_eq_y, shading = 'gouraud', label='u_y_pred', vmin=u_y_min, vmax= u_y_max, cmap=plt.get_cmap('rainbow'))
#    fig.colorbar(c1, ax=ax[1][0])

#    c2=ax[1][1].pcolormesh(x0,y0, u_eq_yx, shading = 'gouraud', label='u_y_exact', vmin=u_Y_min, vmax= u_Y_max, cmap=plt.get_cmap('rainbow'))
#    fig.colorbar(c2, ax=ax[1][1])

#    c1=ax[2][0].pcolormesh(x0,y0, row_pred, shading = 'gouraud', label='density_pred', vmin=row_pred_min, vmax= row_pred_max, cmap=plt.get_cmap('rainbow'))
#    cb1=fig.colorbar(c1, ax=ax[2][0])
#    cb1.formatter.set_powerlimits((0, 0))
#    cb1.update_ticks()

#    c2=ax[2][1].pcolormesh(x0,y0, row_exact, shading = 'gouraud', label='density_exact', vmin=row_exact_min, vmax= row_exact_max, cmap=plt.get_cmap('rainbow'))
#    cb2=fig.colorbar(c2, ax=ax[2][1])
#    cb2.formatter.set_powerlimits((0, 0))
#    cb2.update_ticks()

#    plt.show()




class Potential_flow():
    def __init__(self,device,data,data2,data3):

        plt.rcParams.update({

            "font.family": "times new roman",
            "mathtext.default": "regular"
        })


        self.device = device


        self.error = data[2]
        self.loss = data[3]
        self.error2 = data2[2]
        self.loss2 = data2[3]
        self.error3 = data3[5]
        self.loss3 = data3[6]
        self.V_pred_norm = data[0]
        self.V_NN_pred_norm = data2[0]
        self.V_test = data[1]
        self.indices = data[4]
        print('Error',self.error[-1])
        print('Error3', self.error3[-1])
        #if pinn is True:
        self.bc_loss = data[5]
        self.in_loss = data[6]
        self.out_loss = data[7]
        self.cylBc_loss = data[8]
        self.domain_loss = data[9]
        self.V_pred = data[10]
        self.V_pred_NN = data2[10]
        self.V_domain = data[11]
        self.x_cyl = data[12]

        self.V_domain = torch.div(self.V_domain,10.0)

        self.bc_loss = torch.tensor(self.bc_loss).float().to(device)
        self.in_loss = torch.tensor(self.in_loss).float().to(device)
        self.out_loss = torch.tensor(self.out_loss).float().to(device)
        self.cylBc_loss = torch.tensor(self.cylBc_loss).float().to(device)
        self.domain_loss = torch.tensor(self.domain_loss).float().to(device)

        '''if pinn is True:
            self.bc_loss = data[5]
            self.in_loss = data[6]
            self.out_loss = data[7]
            self.cylBc_loss = data[8]
            self.domain_loss = data[9]

            self.bc_loss = torch.tensor(self.bc_loss).float().to(device)
            self.in_loss = torch.tensor(self.in_loss).float().to(device)
            self.out_loss = torch.tensor(self.out_loss).float().to(device)
            self.cylBc_loss = torch.tensor(self.cylBc_loss).float().to(device)
            self.domain_loss = torch.tensor(self.domain_loss).float().to(device)'''


        #self.V_pred = torch.from_numpy(self.V_pred_np).float().to(device)

        print(self.V_test.size())

        self.error_cpu = torch.tensor(self.error,device=self.device)
        #print('error',self.error_cpu[-1])
        self.loss_cpu = torch.tensor(self.loss, device=self.device)
        self.error2_cpu = torch.tensor(self.error2, device = self.device)
        self.loss2_cpu = torch.tensor(self.loss2, device = self.device)
        self.error3_cpu = torch.tensor(self.error3, device=self.device)[:1000]
        self.loss3_cpu = torch.tensor(self.loss3, device=self.device)[:1000]
        print(self.error_cpu.shape)
        print(self.error3_cpu.shape)
        self.epochs = 5000

        self.xmax = int(self.epochs/5)

        self.x1 = [*range(1,self.xmax+1)]

        x_values = np.arange(-8.0, 8.0, 0.05).tolist()
        y_values = np.arange(-4.0, 4.0, 0.05).tolist()
        x_values.append(8.0)
        y_values.append(4.0)

        self.x, self.y = np.meshgrid(x_values, y_values)

        self.X_in = np.hstack([self.x.flatten()[:, None], self.y.flatten()[:, None]])

        X_in1 = torch.from_numpy(self.X_in).float().to(device)

        v_test_norm_in = torch.zeros_like(X_in1, device= device, dtype=torch.float64)
        v_pred_norm_in = torch.zeros_like(X_in1, device = device, dtype=torch.float64)
        v_domain_in = torch.zeros_like(X_in1, device = device, dtype=torch.float64)
        v_pred_in = torch.zeros_like(X_in1, device = device, dtype=torch.float64)
        v_NN_pred_norm_in = torch.zeros_like(X_in1, device=device, dtype=torch.float64)


        print('indices',self.indices)
        print('size',X_in1.size()[0])
        count = 0
        for i in range(X_in1.size()[0]):
            if i in self.indices:
                v_test_norm_in[i]=0
                v_pred_norm_in[i]=0
                v_NN_pred_norm_in[i]=0
                v_domain_in[i] = 0
                v_pred_in[i] = 0
            else:
                v_test_norm_in[i]=self.V_test[count]
                v_pred_norm_in[i]=self.V_pred_norm[count]
                v_NN_pred_norm_in[i]=self.V_NN_pred_norm[count]
                v_domain_in[i] = self.V_domain[count]
                v_pred_in[i] = self.V_pred[count]
                count+=1

        u0 = v_domain_in[:, 0]
        v0 = v_domain_in[:, 1]

        U = np.sqrt(np.square(u0.cpu().detach().numpy()) + np.square(v0.cpu().detach().numpy()))
        # print('U',U)
        self.U_grid = np.reshape(U, self.x.shape)

        N_u = int(0.05 * len(self.X_in))
        idx = np.random.choice(self.X_in.shape[0], N_u, replace=False)

        X_in2 = self.X_in[idx, :]
        self.x2 = X_in2[:,0]
        self.y2 = X_in2[:,1]

        v_pred_in2 = v_pred_in[idx,:]
        v_test_in2 = v_domain_in[idx,:]

        self.u_pred = v_pred_in2[:,0].cpu().detach().numpy()
        self.v_pred = v_pred_in2[:,1].cpu().detach().numpy()
        self.u_test = v_test_in2[:,0].cpu().detach().numpy()
        self.v_test = v_test_in2[:,1].cpu().detach().numpy()

        self.u_pred_grid = torch.reshape(v_pred_in[:, 0], self.x.shape).cpu().detach().numpy()
        self.u_test_grid = torch.reshape(v_domain_in[:, 0], self.x.shape).cpu().detach().numpy()
        self.v_pred_grid = torch.reshape(v_pred_in[:, 1], self.x.shape).cpu().detach().numpy()
        self.v_test_grid = torch.reshape(v_domain_in[:, 1], self.x.shape).cpu().detach().numpy()

        #print(v_test_in)
        self.u_pred_norm_grid = torch.reshape(v_pred_norm_in[:,0], self.x.shape).cpu().detach().numpy()
        self.u_test_norm_grid = torch.reshape(v_test_norm_in[:,0], self.x.shape).cpu().detach().numpy()
        self.u_pred_NN_norm_grid = torch.reshape(v_NN_pred_norm_in[:,0], self.x.shape).cpu().detach().numpy()
        self.v_pred_norm_grid = torch.reshape(v_pred_norm_in[:, 1], self.x.shape).cpu().detach().numpy()
        self.v_test_norm_grid = torch.reshape(v_test_norm_in[:, 1], self.x.shape).cpu().detach().numpy()
        self.v_pred_NN_norm_grid = torch.reshape(v_NN_pred_norm_in[:,1], self.x.shape).cpu().detach().numpy()
        #print(self.u_test_grid)


    def error_loss_plot(self):
        e= int(self.epochs/5)
        fig,ax = plt.subplots(2,1)
        ax[0].plot(self.x1,self.error_cpu, label='PINN Potential flow cylinder error')
        ax[0].plot(self.x1, self.error2_cpu, label='NN Potential flow cylinder error')
        ax[0].plot(self.x1, self.error3_cpu, label='NN Rankine oval flow cylinder error')
        #ax[0][0].set_yscale('log')
        ax[0].set_ylim(0,0.5)
        ax[0].set_yticks((0,0.05,0.08,0.1,0.2,0.3,0.5))
        ax[0].set_xticks((0,e/5,2*e/5,3*e/5,4*e/5,e))
        ax[0].set_xticklabels((0,self.epochs/5,2*self.epochs/5,3*self.epochs/5,4*self.epochs/5,self.epochs), fontsize=10)
        ax[0].set_ylabel('L2 Error')
        ax[0].set_xlabel('Iterations')
        ax[0].xaxis.label.set_fontsize(15)
        ax[0].yaxis.label.set_fontsize(15)
        ax[0].grid()
        ax[0].legend()

        ax[1].plot(self.x1,self.loss_cpu, label='PINN Potential flow cylinder loss')
        ax[1].plot(self.x1, self.loss2_cpu, label='NN Potential flow cylinder loss')
        ax[1].plot(self.x1, self.loss3_cpu, label='NN Rankine oval flow cylinder loss')
        #ax[0][0].set_yscale('log')
        ax[1].set_ylim(0, 0.03)
        ax[1].set_yticks((0, 0.01, 0.02, 0.03))
        ax[1].set_xticks((0,e/5,2*e/5,3*e/5,4*e/5,e))
        ax[1].set_xticklabels((0,self.epochs/5,2*self.epochs/5,3*self.epochs/5,4*self.epochs/5,self.epochs), fontsize=10)
        ax[1].set_ylabel('MSE loss')
        ax[1].set_xlabel('Iterations')
        ax[1].xaxis.label.set_fontsize(15)
        ax[1].yaxis.label.set_fontsize(15)
        ax[1].grid()
        ax[1].legend()

        plt.show()

    def velocity_plot(self,V_pred_np,V_test):

        fig,ax = plt.subplots(1,2)
        V_test_np = V_test.cpu().detach().numpy()
        ax[0].scatter(V_pred_np[:,0],V_pred_np[:,1], label = 'predicted_velocity_pinn')
        ax[0].scatter(V_test_np[:,0], V_test_np[:,1], marker='x', label = 'exact_velocity_pinn')
        ax[0].grid()
        ax[0].legend()

        #ax[1].scatter(V1_pred_np[:, 0], V1_pred_np[:, 1], label='predicted_velocity_dnn')
        #ax[1].scatter(V_test_np[:, 0], V_test_np[:, 1], marker='x', label='exact_velocity_pinn')
        #ax[1].grid()
    #
        plt.show()

    def vector_plot(self,x,y,u_test,v_test,u_pred,v_pred):

        fig,ax = plt.subplots(1,2)

        ax[0].quiver(x,y,u_test, v_test, units='xy', label='Exact')
        #ax[0][0].legend(loc='upper right')
        ax[1].quiver(x,y,u_pred,v_pred, units='xy', color='red', label='PINN')
        #ax[0][0].legend(loc='upper right')
        #ax[1][1].quiver(x,y,u1_pred_grid,v1_pred_grid, units='xy', color='blue', label='DNN')
        #handles, labels = ax.get_legend_handles_labels()
        fig.legend(loc='upper right')
        plt.show()

    def density_plot(self):

        u_test_min = self.u_test_norm_grid.min()
        u_test_max = self.u_test_norm_grid.max()

        v_test_max = self.v_test_norm_grid.max()
        v_test_min = self.v_test_norm_grid.min()



        fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [3, 3]})
        c1=ax[0][0].pcolormesh(self.x,self.y, self.u_test_norm_grid, shading = 'gouraud', label='u_x_exact', vmin=u_test_min, vmax= u_test_max, cmap=plt.get_cmap('YlGnBu'))
        cbar=fig.colorbar(c1, ax=ax, aspect=50)
        cbar.ax.tick_params(labelsize=15)
        #ax[0][0].set_title('$U_{exact}$', y=-0.2)
        ax[0][0].add_patch(plt.Circle((0, 0), 2, color='Black', fill=False))
        ax[0][0].title.set_fontsize(15)
        ax[0][0].axes.xaxis.set_visible(False)
        ax[0][0].axes.yaxis.set_visible(False)

        c2=ax[0][1].pcolormesh(self.x,self.y, self.v_test_norm_grid, shading = 'gouraud', label='v_x_test', vmin=v_test_min, vmax= v_test_max, cmap=plt.get_cmap('YlGnBu'))
        #fig.colorbar(c2, ax=ax, aspect=50)
        #ax[0][1].set_title('$V_{exact}$', y=-0.2)
        ax[0][1].add_patch(plt.Circle((0, 0), 2, color='Black', fill=False))
        ax[0][1].title.set_fontsize(15)
        ax[0][1].axes.xaxis.set_visible(False)
        ax[0][1].axes.yaxis.set_visible(False)

        c3 = ax[1][0].pcolormesh(self.x, self.y, self.u_pred_norm_grid, shading='gouraud', label='u_x_pred', vmin=u_test_min, vmax=u_test_max,
                                 cmap=plt.get_cmap('YlGnBu'))
        #fig.colorbar(c3, ax=ax[1][0])
        #ax[1][0].set_title('$U_{pred}(PINN)$', y=-0.2)
        ax[1][0].add_patch(plt.Circle((0,0),2, color='Black', fill=False))
        ax[1][0].title.set_fontsize(15)
        ax[1][0].axes.xaxis.set_visible(False)
        ax[1][0].axes.yaxis.set_visible(False)

        c4 = ax[1][1].pcolormesh(self.x, self.y, self.v_pred_norm_grid, shading='gouraud', label='v_x_pred', vmin=v_test_min, vmax=v_test_max,
                                 cmap=plt.get_cmap('YlGnBu'))
        #fig.colorbar(c4, ax=ax[1][1])
        #ax[1][1].set_title('$V_{pred}(PINN)$', y=-0.2)
        ax[1][1].add_patch(plt.Circle((0, 0), 2, color='Black', fill=False))
        ax[1][1].title.set_fontsize(15)
        ax[1][1].axes.xaxis.set_visible(False)
        ax[1][1].axes.yaxis.set_visible(False)

        c5 = ax[2][0].pcolormesh(self.x, self.y, self.u_pred_NN_norm_grid, shading='gouraud', label='u_x_pred',
                                 vmin=u_test_min, vmax=u_test_max,
                                 cmap=plt.get_cmap('YlGnBu'))
        # fig.colorbar(c3, ax=ax[1][0])
        #ax[2][0].set_title('$U_{pred}(DNN)$', y=-0.2)
        ax[2][0].add_patch(plt.Circle((0, 0), 2, color='Black', fill=False))
        ax[2][0].title.set_fontsize(15)
        ax[2][0].axes.xaxis.set_visible(False)
        ax[2][0].axes.yaxis.set_visible(False)

        c6 = ax[2][1].pcolormesh(self.x, self.y, self.v_pred_NN_norm_grid, shading='gouraud', label='v_x_pred',
                                 vmin=v_test_min, vmax=v_test_max,
                                 cmap=plt.get_cmap('YlGnBu'))
        #fig.colorbar(c6, ax=ax[2][1])
        #ax[2][1].set_title('$V_{pred}(DNN)$', y=-0.2)
        ax[2][1].add_patch(plt.Circle((0, 0), 2, color='Black', fill=False))
        ax[2][1].title.set_fontsize(15)
        ax[2][1].axes.xaxis.set_visible(False)
        ax[2][1].axes.yaxis.set_visible(False)

        plt.show()

    def streamline(self):
        fig, ax = plt.subplots(2, 1)
        s1 = ax[0].streamplot(self.x, self.y, self.u_test_grid, self.v_test_grid, density=2, color=self.U_grid,
                              cmap='rainbow')
        c1 = fig.colorbar(s1.lines, ax=ax[0])
        c1.set_label('$u_{mag}$/U', rotation=0, labelpad=25, fontsize = 20, fontfamily = 'times new roman')
        c1.ax.tick_params(labelsize=15)
        ax[0].plot(self.x_cyl[:, 0], self.x_cyl[:, 1], color='black')
        #ax[0].plot([-2.0,2.0], [0.0,0.0])
        ax[0].annotate('<------ D ------>', xy=(-2.0, -0.1), xycoords='data', fontsize=20, fontfamily='times new roman')
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)


        s2 = ax[1].streamplot(self.x, self.y, self.u_pred_grid, self.v_pred_grid, density=2, color=self.U_grid, cmap='rainbow')
        fig.colorbar(s2.lines, ax=ax[1])
        ax[1].plot(self.x_cyl[:, 0], self.x_cyl[:, 1], color='black')


        plt.show()

############################################################################################################################


class Plotting():
    def __init__(self,data, data2):
    

        self.result = data
        self.result2 = data2

        self.V_pred = self.result[0]
        self.V_NN_pred = self.result2[0]
        self.V_domain = self.result[1]
                
        V_pred_norm = self.result[2]
        V_NN_pred_norm = self.result2[2]
        V_in_norm = self.result[3]
        indices = self.result[4]
        self.X_boundary_sort = self.result[7]
        self.V_in  = self.result[8]
        print(self.V_in)

        self.V_domain = torch.div(self.V_domain, 10.0)
        #print(self.V_in1)

        h = 0.02
        x_min = -8.0
        x_max = 8.0
        y_min = -4.0
        y_max = 4.0

        nox = int((x_max - x_min) / h)
        noy = int((y_max - y_min) / h)
        print(nox)
        print(noy)

        dec = 1

        plt.rcParams.update({

            "font.family": "times new roman",
            "mathtext.default": "regular"
        })

        #x_values = np.linspace(x_min, x_max, nox).tolist()
        #y_values = np.linspace(y_min, y_max, noy).tolist()

        #print(y_values)

        x_values1 = np.arange(-8.0, 8.0, h).tolist()
        y_values1 = np.arange(-5.0, 5.0, h).tolist()
        x_values1.append(8.0)
        y_values1.append(5.0)
        x_values = [round(elem, dec) for elem in x_values1]
        y_values = [round(elem, dec) for elem in y_values1]


        self.x, self.y = np.meshgrid(x_values, y_values)
        self.x1, self.y1= np.meshgrid(x_values1, y_values1)
        #plt.hist(V_in[:,0].cpu().detach().numpy())
        X_in = np.hstack([self.x.flatten()[:, None], self.y.flatten()[:, None]])
        X_in1 = torch.from_numpy(X_in).float().to(device)

        #X_in1 = torch.from_numpy(X_in).float().to(device)

        v_test_in_norm = torch.zeros_like(X_in1, device= device, dtype=torch.float64)
        v_pred_in_norm = torch.zeros_like(X_in1, device= device, dtype=torch.float64)
        v_test_in = torch.zeros_like(X_in1, device= device, dtype=torch.float64)
        v_pred_in = torch.zeros_like(X_in1, device= device, dtype=torch.float64)
        v_NN_pred_in_norm = torch.zeros_like(X_in1, device= device, dtype=torch.float64)

        c = 0

        for i in range(X_in1.size()[0]):
            if i in indices:
                v_test_in[i]=0
                v_pred_in[i]=0
                v_test_in_norm[i]=0
                v_pred_in_norm[i]=0
                v_NN_pred_in_norm[i]=0
            else:
                v_test_in[i]=self.V_domain[c]
                v_pred_in[i]=self.V_pred[c]
                v_test_in_norm[i]=V_in_norm[c]
                v_pred_in_norm[i]=V_pred_norm[c]
                v_NN_pred_in_norm[i]=V_NN_pred_norm[c]
                c+=1

                     
        up = v_pred_in[:,0]
        vp = v_pred_in[:,1]
        u0 = v_test_in[:,0]
        v0 = v_test_in[:,1]

        #normalized
        up_norm = v_pred_in_norm[:,0]
        vp_norm = v_pred_in_norm[:,1]
        up_NN_norm = v_NN_pred_in_norm[:,0]
        vp_NN_norm = v_NN_pred_in_norm[:,1]
        u0_norm = v_test_in_norm[:,0]
        v0_norm = v_test_in_norm[:,1]


        U = np.sqrt(np.square(u0.cpu().detach().numpy()) + np.square(v0.cpu().detach().numpy()))
        #print('U',U)
        self.U_grid = np.reshape(U, self.x.shape)

        self.u_grid_norm = torch.reshape(up_norm, self.x.shape).cpu().detach().numpy()
        self.v_grid_norm = torch.reshape(vp_norm, self.y.shape).cpu().detach().numpy()
        self.u_NN_grid_norm = torch.reshape(up_NN_norm, self.y.shape).cpu().detach().numpy()
        self.v_NN_grid_norm = torch.reshape(vp_NN_norm, self.y.shape).cpu().detach().numpy()

        self.u0_grid_norm = torch.reshape(u0_norm, self.x.shape).cpu().detach().numpy()
        self.v0_grid_norm = torch.reshape(v0_norm, self.y.shape).cpu().detach().numpy()

        self.u_grid = torch.reshape(up, self.x.shape).cpu().detach().numpy()
        self.v_grid = torch.reshape(vp, self.y.shape).cpu().detach().numpy()

        self.u0_grid = torch.reshape(u0, self.x.shape).cpu().detach().numpy()
        self.v0_grid = torch.reshape(v0, self.y.shape).cpu().detach().numpy()

        '''u_in = self.V_in1[:,0]
        v_in = self.V_in1[:,1]

        U_in = np.sqrt(np.square(u_in.cpu().detach().numpy()) + np.square(v_in.cpu().detach().numpy()))
        # print('U',U)
        self.U_in_grid = np.reshape(U_in, self.x.shape)

        self.u_in_grid = torch.reshape(u_in, self.x.shape).cpu().detach().numpy()
        self.v_in_grid = torch.reshape(v_in, self.y.shape).cpu().detach().numpy()'''

        

        '''fig, ax = plt.subplots(1,3)

        ax[0].plot(u0)
        ax[1].plot(v0)
        ax[2].plot(U)

        plt.show()'''
        

    def density_plot_norm(self):
        
        u_grid_max = self.u0_grid_norm.max()
        u_grid_min = self.u0_grid_norm.min()

        v_grid_max = self.v0_grid_norm.max()
        v_grid_min = self.v0_grid_norm.min()

        fig, ax = plt.subplots(3, 2,gridspec_kw={'width_ratios': [3, 3]})
        c1=ax[0][0].pcolormesh(self.x,self.y, self.u0_grid_norm, shading = 'gouraud', label='u_x_exact', vmin=u_grid_min, vmax= u_grid_max, cmap=plt.get_cmap('YlGnBu'))
        cbar = fig.colorbar(c1, ax=ax, aspect=50)
        cbar.ax.tick_params(labelsize=15)
        ax[0][0].plot(self.X_boundary_sort[:, 0], self.X_boundary_sort[:, 1], color='black')
        ax[0][0].hlines(y=0, xmin=-8.0, xmax=8.0, color='black')
        #ax[0][0].set_title('$U_{exact}$', y=-0.2)
        ax[0][0].title.set_fontsize(15)
        ax[0][0].axes.xaxis.set_visible(False)
        ax[0][0].axes.yaxis.set_visible(False)


        c2=ax[1][0].pcolormesh(self.x,self.y, self.u_grid_norm, shading = 'gouraud', label='u_x_pred', vmin=u_grid_min, vmax= u_grid_max, cmap=plt.get_cmap('YlGnBu'))
        #fig.colorbar(c2, ax=ax[1][0])

        ax[1][0].plot(self.X_boundary_sort[:, 0], self.X_boundary_sort[:, 1], color='black')
        ax[1][0].hlines(y=0, xmin=-8.0, xmax=8.0, color='black')
        #ax[1][0].set_title('$U_{pred}(PINN)$', y=-0.2)
        ax[1][0].title.set_fontsize(15)
        ax[1][0].axes.xaxis.set_visible(False)
        ax[1][0].axes.yaxis.set_visible(False)

        c3=ax[0][1].pcolormesh(self.x,self.y, self.v0_grid_norm, shading = 'gouraud', label='v_x_exact', vmin=v_grid_min, vmax= v_grid_max, cmap=plt.get_cmap('YlGnBu'))
        #fig.colorbar(c3, ax=ax[0][1])

        ax[0][1].plot(self.X_boundary_sort[:, 0], self.X_boundary_sort[:, 1], color='black')
        ax[0][1].hlines(y=0, xmin=-8.0, xmax=8.0, color='black')
        #ax[0][1].set_title('$V_{exact}$', y=-0.2)
        ax[0][1].title.set_fontsize(15)
        ax[0][1].axes.xaxis.set_visible(False)
        ax[0][1].axes.yaxis.set_visible(False)


        c4=ax[1][1].pcolormesh(self.x,self.y, self.v_grid_norm, shading = 'gouraud', label='v_x_pred', vmin=v_grid_min, vmax= v_grid_max, cmap=plt.get_cmap('YlGnBu'))
        #fig.colorbar(c4, ax=ax[1][1])

        ax[1][1].plot(self.X_boundary_sort[:, 0], self.X_boundary_sort[:, 1], color='black')
        ax[1][1].hlines(y=0, xmin=-8.0, xmax=8.0, color='black')
        #ax[1][1].set_title('$V_{pred}(PINN)$', y=-0.2)
        ax[1][1].title.set_fontsize(15)
        ax[1][1].axes.xaxis.set_visible(False)
        ax[1][1].axes.yaxis.set_visible(False)

        c5 = ax[2][0].pcolormesh(self.x, self.y, self.u_NN_grid_norm, shading='gouraud', label='v_x_exact',
                                 vmin=u_grid_min, vmax=u_grid_max, cmap=plt.get_cmap('YlGnBu'))
        #fig.colorbar(c3, ax=ax[2][0])

        ax[2][0].plot(self.X_boundary_sort[:, 0], self.X_boundary_sort[:, 1], color='black')
        ax[2][0].hlines(y=0, xmin=-8.0, xmax=8.0, color='black')
        #ax[2][0].set_title('$U_{pred}(DNN)$', y=-0.2)
        ax[2][0].title.set_fontsize(15)
        ax[2][0].axes.xaxis.set_visible(False)
        ax[2][0].axes.yaxis.set_visible(False)

        c6 = ax[2][1].pcolormesh(self.x, self.y, self.v_NN_grid_norm, shading='gouraud', label='v_x_pred', vmin=v_grid_min,
                                 vmax=v_grid_max, cmap=plt.get_cmap('YlGnBu'))
        #fig.colorbar(c6, ax=ax[2][1])

        ax[2][1].plot(self.X_boundary_sort[:, 0], self.X_boundary_sort[:, 1], color='black')
        ax[2][1].hlines(y=0, xmin=-8.0, xmax=8.0, color='black')
        #ax[2][1].set_title('$V_{pred}(DNN)$', y=-0.2)
        ax[2][1].title.set_fontsize(15)
        ax[2][1].axes.xaxis.set_visible(False)
        ax[2][1].axes.yaxis.set_visible(False)


        plt.show()

    def density_plot(self):
        #fig, ax = plt.subplots(1,2)
        plt.figure(figsize = (8,8))
        plt.imshow(self.u_grid_norm)
        plt.imshow(self.v_grid_norm)

        plt.show()

    def streamplot(self):

        fig, ax = plt.subplots(2,1)

        ##
        file0 = open('result_PINN_potential_flow.pkl', 'rb')
        file1 = open('result_potential_flow.pkl', 'rb')
        data0 = pickle.load(file0)
        data1 = pickle.load(file1)
        plot0 = Potential_flow(device, data0, data1, data2)
        s2 = ax[0].streamplot(plot0.x, plot0.y, plot0.u_test_grid, plot0.v_test_grid, density=2, color=plot0.U_grid,
                              cmap='rainbow')
        c2 = fig.colorbar(s2.lines, ax=ax, aspect=50)
        #c2.set_label('$u_{mag}$/U', rotation=0, labelpad=30, fontsize=20, fontfamily='times new roman')
        c2.ax.tick_params(labelsize=15)
        c2.set_ticks((0.0,0.5, 1.0,1.5,2.0))
        ax[0].plot(plot0.x_cyl[:, 0], plot0.x_cyl[:, 1], color='black')
        # ax[0].plot([-2.0,2.0], [0.0,0.0])
        #ax[0].annotate('<------ D ------>', xy=(-2.0, -0.1), xycoords='data', fontsize=20, fontfamily='times new roman')
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)
        '''plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.setp(ax[0].get_yticklabels(), visible=False)
        ax[0].tick_params(axis='both', which='both', length=0)
        ax[0].set_xlabel('x', rotation=0, labelpad=5, fontsize=15, fontfamily='times new roman')
        ax[0].set_ylabel('y', rotation=0, labelpad=10, fontsize=15, fontfamily='times new roman')'''
        ##


        s1 = ax[1].streamplot(self.x1,self.y1,self.u0_grid, self.v0_grid, density=2,color = self.U_grid, cmap='rainbow')
        #c1 = fig.colorbar(s1.lines, ax =ax[1])
        #c1.set_label(r'$u_{mag}/U$', rotation=0, labelpad=30,fontsize=20, fontfamily='times new roman')
        #c1.ax.tick_params(labelsize=15)
        #c1.set_ticks((0.0, 0.5,1.0, 1.5))
        ax[1].plot(self.X_boundary_sort[:,0], self.X_boundary_sort[:,1], color='black')
        ax[1].hlines(y=0,xmin=-8.0,xmax=8.0, color='black')
        ax[1].plot(2,0, color = 'red', marker = 'o')
        ax[1].plot(-2,0, color = 'red', marker = 'o')
        #ax[1].annotate('<------ 2a ------>', xy=(-2.0, 0.1), xycoords='data', fontsize=20, fontfamily='times new roman')
        ax[1].axes.xaxis.set_visible(False)
        ax[1].axes.yaxis.set_visible(False)
        '''plt.setp(ax[1].get_xticklabels(), visible=False)
        plt.setp(ax[1].get_yticklabels(), visible=False)
        ax[1].tick_params(axis='both', which='both', length=0)
        ax[1].set_xlabel('x', rotation=0, labelpad=5,fontsize=15, fontfamily='times new roman')
        ax[1].set_ylabel('y', rotation=0, labelpad=10, fontsize=15, fontfamily='times new roman')
        
        s2 = ax[1].streamplot(self.x1,self.y1,self.u_grid, self.v_grid, density=2, color = self.U_grid, cmap='rainbow')
        fig.colorbar(s2.lines, ax=ax[1])
        ax[1].plot(self.X_boundary_sort[:, 0], self.X_boundary_sort[:, 1], color='black')
        ax[1].hlines(y=0, xmin=-8.0, xmax=8.0, color='black')
        '''
        plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

file2 = open('result_rankine_oval_potential_flow.pkl', 'rb')
data2 = pickle.load(file2)
file3 = open('result_NN_rankine_oval_potential_flow.pkl', 'rb')
data3 = pickle.load(file3)

plot = Plotting(data2,data3)

plot.density_plot_norm()
#plot.streamplot()
#plot.density_plot()

####################################################################################################

file0 = open('result_PINN_potential_flow.pkl', 'rb')
file1 = open('result_potential_flow.pkl', 'rb')
data0 = pickle.load(file0)
data1 = pickle.load(file1)

plot0 = Potential_flow(device,data0,data1,data2)

#error_plot = plot0.error_loss_plot()
#density = plot0.density_plot()
#stream = plot0.streamline()

