
import numpy as np
import torch
import underlying_asset_simulation as ua
import scipy.stats
import matplotlib.pyplot as plt
import time
import torch.utils.data as tdata
import torch.nn as nn
import multiprocessing as mp
import sys
import torch.nn.functional as F
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def write_file(text):
    f = open("ended_output.txt", "a")
    f.write(text)
    f.close()
    
def write_table(model_name,N,P,CI_L, CI_U):
    f = open('ended_output_table_'+model_name+'.txt', 'a')
    text = str(N)+';'+str(P)+';'+str(CI_L)+';'+str(CI_U)+'\n'
    f.write(text)
    f.close()
def symmetric_case(sigma, symm, d):
    if(symm):
        # Create a matrix with diagonal vector 'sigma_diag_vec' and off-diagonal elements equal to 'rho'
        sigma_diag_vec = sigma * np.ones(d)
        rho = 0
        sigma_matrix = rho * np.ones((d, d))
        np.fill_diagonal(sigma_matrix, sigma_diag_vec)
        return sigma_matrix
    else:
        sigma_diag_vec = np.zeros(d)
        if(d > 5):
            for dimension in range(d):
                sigma_diag_vec[dimension] = 0.1 + (dimension+1)/(2*d)
        else:
            for dimension in range(d):
                sigma_diag_vec[dimension] = 0.08 + 0.32 * (dimension)/(d-1)
        rho = 0
        sigma_matrix = rho * np.ones((d, d))
        np.fill_diagonal(sigma_matrix, sigma_diag_vec)
        return sigma_diag_vec

def g(_t, _x, barrier = None, axis = 1):
    if barrier == []:
        payoff = torch.max(_x, dim=axis, keepdim = True).values
        return torch.exp(-r*_t)*torch.max(payoff-K,torch.tensor(0.))
    elif barrier == "MBRC":
        c = 7/12.
        B = 70
        F = 100
        Breach = torch.zeros(outer_batch_size, N)
        G = torch.zeros(outer_batch_size, 1, N)
    
        for j in range(outer_batch_size):
            Breach[j, 0] = torch.tensor(0.)
            for n in range(N):
                suma = 0
                H = torch.min(_x[j, :, n])
                if n-1 == 0:
                    Breach[j, n] = torch.where(H > B, torch.tensor([1]), torch.tensor([0]))
                else:
                    Breach[j, n] = torch.max(Breach[j,n-1], torch.where(H > B, torch.tensor([1]), torch.tensor([0])))
                if Breach[j, n] == torch.tensor([0]) or n < N-1:
                    for m in range(n):
                        suma += (torch.exp(-r*_t[m])*c)
                    suma += torch.exp(-r*_t[n])*F
                else:
                    for m in range(n):
                        suma += (torch.exp(-r*_t[m])*c)
                    H = torch.min(_x[j,:,-1])
                    a = torch.tensor([K])
                    if H > a:
                        H = K
                    suma += (torch.exp(-r*_t[-1]) * H) 
                G[j,0,n] = suma
        return G
def sample_paths(bsize, _d, x_0, rho, ua_model = 'BS', symm = True):

    if ua_model == 'BarrierBS':
        Ti = 1/2
        Tb = 1
        deltaib = 0.05
        rb = 0
        mu = rb- sigma**2/2
        time_step = dt = Tb/N
        gaussian = np.random.normal(size=(N,d))
        CHOL = np.linalg.cholesky(np.eye(d)+rho*(np.ones((d,d)) - np.eye(d)))
        pre_wi = torch.zeros(size = (bsize, d, N))
        
        for i in range(N):
            for j in range(bsize):
                scale_cholesky_gaussian = sigma*np.sqrt(dt)* np.dot(CHOL,gaussian[i])
                scale_cholesky_gaussian = torch.from_numpy(scale_cholesky_gaussian)
                if(i*dt < Ti):
                    for k in range(d):
                        pre_wi[j,k,i] = pre_wi[j,k,i] + scale_cholesky_gaussian[k]
                else:
                    for k in range(d):
                        pre_wi[j,k,i] = pre_wi[j,k,i] + scale_cholesky_gaussian[k]              

        
        
        _wi = torch.cumsum(pre_wi[:,:,:N//2], dim = 2)
        #_wi = torch.cumsum(pre_wi[:,:,:int(np.floor(N/2))], dim = 2)
        _ti = torch.linspace(Tb/N, Ti, N//2)
        _wo = torch.cumsum(pre_wi[:,:,N//2:], dim = 2)
        #_wo = torch.cumsum(pre_wi[:,:,int(np.ceil(N/2)):], dim = 2)
        _to = torch.linspace(Ti, Tb, N//2)
        _t = torch.linspace(Tb/N, Tb, N)
        
        
        _xi = torch.exp((rb - sigma ** 2 / 2.) * _ti + _wi) * x_0
        _xo = torch.exp((rb - sigma ** 2 / 2.) * _to + _wo) * x_0*(1-deltaib)
        _x = torch.cat((_xi, _xo), dim = 2)
        
        return _x, _t
        
    if ua_model == 'BS':
        if symm == False:
            time_step = T/N
            pre_w = torch.zeros(size = (bsize, d, N))
            symm_sigma = symmetric_case(sigma, symm,_d)
            gaussian = np.random.normal(size=(N+1,_d))
            path = torch.zeros(bsize, _d, N+1)
            mu = torch.zeros(_d)
            for i in range(_d):
                mu[i] = (r - delta - symm_sigma[i] ** 2 / 2.)
            
            CHOL = np.linalg.cholesky(np.eye(_d)+rho*(np.ones((_d,_d)) - np.eye(_d)))
            for j in range(bsize):
                for i in range(N):
                    for z in range(_d):
                        scale_cholesky_gaussian = symm_sigma[z]*np.sqrt(time_step)* np.dot(CHOL[z,:],gaussian[i])
                        pre_w[j,z,i] = pre_w[j,z,i] + scale_cholesky_gaussian
            _w = torch.cumsum(pre_w, dim = 2)
            _t = torch.linspace(T/N, T, N)
            for j in range(bsize):
                for i in range(1,N):
                    for z in range(_d):
                        if i == 1:
                            path[j,z,i] = x_0*torch.exp(mu[z]*time_step+ pre_w[j,z,i])
                        else:
                            path[j,z,i] = path[j,z,i-1]*torch.exp(mu[z]*time_step+ pre_w[j,z,i])
            
            _x = path[:,:,1:]
            return _x, _t
        else:
            pre_w = torch.normal(mean = 0.0, std = sigma*np.sqrt(T/N), size=(bsize, d, N))
            _w = torch.cumsum(pre_w, dim = 2)
            _t = torch.linspace(T/N, T, N)
            _x = torch.exp((r - delta - sigma ** 2 / 2.) * _t + _w) * x_0
            
            return _x, _t
    elif ua_model == 'Bach':
        pre_w = torch.normal(mean = 0.0, std = sigma*np.sqrt(T/N), size=(bsize, d, N))
    
        _w = torch.cumsum(pre_w, dim = 2)
        _t = torch.linspace(T/N, T, N)
        _x = x_0 + ((r - delta - sigma ** 2 / 2.) * _t + _w) 
        return _x, _t
    elif ua_model == 'Kou':
        dt = T / N
        lambdat = 0.08
        p = 0.4
        lambdam = lambdap = 1000
        _t = torch.linspace(T/N, T, N)
        mu = (r - delta - sigma ** 2 / 2.)
        t = np.linspace(0, T, N+1)
        X = np.zeros((bsize, d, N+1))
        NT = scipy.stats.poisson.ppf(np.random.rand(bsize, 1), lambdat * T)
        for j in range(bsize):
            Tj = np.sort(T * np.random.rand(int(NT[j])))
            Z = np.random.randn(N, d)
            for i in range(1, N+1):
                X[j,:,i] = X[j,:,i-1] + mu * dt + sigma * np.sqrt(dt) * Z[i-1, :]
                for k in range(int(NT[j])):
                    if (int(Tj[k]) > int(t[i-1])) & (int(Tj[k]) <= int(t[i])):
                        uniform_value = np.random.rand(1)
                        if uniform_value < p:
                            Y = scipy.stats.expon.ppf(uniform_value, 1/lambdap)
                        else:
                            Y = -scipy.stats.expon.ppf(uniform_value, 1/lambdam)
                        J = np.zeros(d)
                        for l in range(d):
                            J[l] = Y * np.random.randn(1)
                        X[j,:,i ] = X[j,:,i ] + J
        X = X[:,:,1:]
        X = torch.from_numpy(X)
        return x_0*torch.exp(X),  _t
    elif ua_model == 'Merton':
        pre_w = torch.normal(mean = 0.0, std = sigma*np.sqrt(T/N), size=(bsize, d, N))
        lambdat, muJ, deltaJ = 3, 0, 0.25
        dt = T / N
        lambdat = 0.08
        p = 0.4
        _t = torch.linspace(T/N, T, N)
        mu = (r - delta - sigma ** 2 / 2.)
        t = np.linspace(0, T, N+1)
        X = np.zeros((bsize, d, N+1))
        NT = scipy.stats.poisson.ppf(np.random.rand(bsize, 1), lambdat * T)
        for j in range(bsize):
            Tj = np.sort(T * np.random.rand(int(NT[j])))
            Z = np.random.randn(N, d)
            for i in range(1, N+1):
                X[j,:,i] = X[j,:,i-1] + mu * dt + sigma * np.sqrt(dt) * Z[i-1, :]
                for k in range(int(NT[j])):
                    if (int(Tj[k]) > int(t[i-1])) & (int(Tj[k]) <= int(t[i])):
                        Y = np.random.normal(muJ, deltaJ, d)
                        X[j,:,i] = X[j,:,i] + Y
        X = X[:,:,1:]
        X = torch.from_numpy(X)
        return x_0*torch.exp(X),  _t
    elif ua_model == 'Heston':
        dt = T/N
        _t = torch.linspace(T/N, T, N)
        time_step = T / N
        size = d
        epsilon = 0.05
        k = 3
        theta = 0.2
        rho = -0.2
        
        V = np.zeros((bsize, N+1, d))
        X = np.zeros((bsize, N+1, d))
        X[:,:,0] = x_0
        V[:,:,0] = 0.2
        
        Feller_condition = (epsilon**2 - 2*k*theta)
        dt = T / N
        
        if Feller_condition > 0: # QE Scheme
            Psi_cutoff = 1.5
            for i in range(N): # Discretise V (volatility) and calculate m, Psi, s2 for each dimension
                m = theta + (V[:,i,:] - theta)*np.exp(-k*dt)
                m2 = m**2
                s2 = V[:,i,:]*epsilon**2*np.exp(-k*dt)*(1-np.exp(-k*dt))/k + theta*epsilon**2*(1-np.exp(-k*dt))**2/(2*k)
                Psi = (s2)/(m2)
                index = np.where(Psi_cutoff < Psi)
                
                # Exponential approx scheme if Psi > Psi_cutoff
                p_exp = (Psi[index]-1)/(Psi[index]+1)
                beta_exp = (1-p_exp)/m[index]
                U = np.random.rand(np.size(index))
                V[index,i+1,:] = (np.log((1-p_exp)/(1-U)/beta_exp)*(U>p_exp))
                
                # Quadratic approx scheme if 0 < Psi < Psi_cutoff
                index = np.where(Psi <= Psi_cutoff)
                invPsi = 1/Psi[index]
                b2_quad = 2*invPsi - 1 + np.sqrt(2*invPsi)*np.sqrt(2*invPsi-1)
                a_quad  = m[index]/(1+b2_quad)
                V[index, i+1,:] = a_quad*(np.sqrt(b2_quad)+ np.random.randn(np.size(index), d))**2
            
            # Central discretisation scheme
                gamma1 = 0.5
                gamma2 = 0.5
                k0 = r*dt -rho*k*theta*dt/epsilon
                k1 = gamma1*dt*(k*rho/epsilon-0.5)-rho/epsilon
                k2 = gamma2*dt*(k*rho/epsilon-0.5)+rho/epsilon
                k3 = gamma1*dt*(1-rho**2)
                k4 = gamma2*dt*(1-rho**2)
                for j in range(bsize):
                    for i in range(N):
                            X[j,i+1,:] = np.exp(np.log(X[j,i,:]))+k0+k1*V[j,i,:]+k2*V[j,i+1,:]+np.sqrt(k3*V[j,i,:])+k4*V[j,i+1,:]*np.random.randn(1,bsize)
                X = X[:,1:,:]
                X = X.permute(0, 2, 1)
                return X, _t
        else:
                mu = [0, 0]
                VC = [[1,rho],[rho,1]]
                for i in range(N):
                    Z = np.random.multivariate_normal(mu,VC,bsize)
                    X[:,i+1,:] = X[:,i,:] + (r-V[:,i,:]/2)*dt+np.sqrt(V[:,i,:]*dt)*Z[:,0]
                    V[:,i+1] = V[:,i] + k*(theta-V[:,i])*dt+epsilon*np.sqrt(V[:,i]*dt)*Z[:,1]
                X = X[:, 1:]
                X = X.permute(0, 2, 1)
                return np.exp(X)

class Neural_Net_NN(torch.nn.Module):
    def __init__(self, M,shape_input):
        super(Neural_Net_NN, self).__init__()
        self.dense1 = nn.Linear(shape_input,M)
        self.dense2 = nn.Linear(M,1)
        self.bachnorm1 = nn.BatchNorm1d(M)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.bachnorm1(self.relu(self.dense1(x.float())))
        x = self.relu(self.dense2(x))
        return x
    
def train_and_price(x, p, n, batch_size, num_neurons, train_steps, mc_runs,
                    lr_boundaries, path, barrier, epsilon=0.1, dtype=torch.float32):

    neural_net = Neural_Net_NN(num_neurons,d+1).to(device)
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_boundaries, gamma=0.1)
    px_hist = []
    print('\n\n Training phase:\n\n')
    max_px = 0
    for train_step in range(0,training_steps+1):
        X,p_,g_tau = x, p, p[:,:,n-1]
        X,p_,g_tau = X.to(device), p_.to(device),g_tau.to(device)
        state = torch.cat((X,p_),axis = 1)
        loss = np.zeros(1)
        loss = torch.tensor(loss).to(device)
        for n in range(N-2, -1, -1):
            net_n = neural_net(state[:,:,n])
            F_n   = torch.sigmoid(net_n)
            if barrier != []:
                loss = torch.mean(p_[:, :, n] * F_n + g_tau * (1. - F_n))
            loss -= torch.mean(p_[:, :, n] * F_n + g_tau * (1. - F_n)) 
            g_tau = torch.where(net_n > 0, p_[:, :, n], g_tau)
        
        px_mean_batch = torch.mean(g_tau)
        loss = torch.mean(loss)
        px_hist.append(px_mean_batch.item())
        
        if train_step > 10:
            if px_mean_batch.item()>max_px and px_mean_batch.item()<np.mean(px_hist[-10:-1])*1.50:
                torch.save(neural_net.state_dict(), 'best_model.pt')
                max_px = px_mean_batch.item()
        
        
        if train_step%100 == 0:
            print('| Train step: {:5.0f} | Loss: {:3.3f} | V: {:3.3f} | Lr: {:1.6f} |'.format(train_step,loss.item(),px_mean_batch.item(),optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
       
    print('\n\n Evaluation phase:\n\n')
    px_vec = []
    px_var_vec = []
    neural_net = Neural_Net_NN(num_neurons,d+1).to(device)
    neural_net.load_state_dict(torch.load('best_model.pt'))
    neural_net.eval()  
    
    
    for mc_step in range(0,mc_runs+1):
        X,p_,g_tau = x, p, p[:,:,n-1]
        X,p_,g_tau = X.to(device), p_.to(device),g_tau.to(device)
        state = torch.cat((X,p_),axis = 1)
        for n in range(N-2, -1, -1): # loop from T-T/N to T/N
            net_n = neural_net(state[:,:,n])
            g_tau = torch.where(net_n > 0, p_[:, :, n], g_tau) 
        
        px_mean_batch = torch.mean(g_tau)
        px_var_batch = torch.var(g_tau)

        if px_mean_batch.item() < torch.mean(torch.max(p_,2)[0]).item():
            px_vec.append(px_mean_batch)
            px_var_vec.append(px_var_batch)
        if mc_step%100 == 0:
            print('| MC run step: {:5.0f} | V: {:3.3f} |'.format(mc_step,px_mean_batch.item()))
    
    if px_vec == []:
        px_mean = torch.tensor([0])
        px_std = torch.tensor([0])
    else:
        px_mean = torch.mean(torch.stack(px_vec))
        px_std = torch.std(torch.stack(px_vec))
    np_px_mean = px_mean.detach().numpy()
    
    p_v = np.sum(np.array(px_var_vec)) / mc_runs + np.var(np_px_mean)
    tmp = scipy.stats.norm.ppf(0.975) * np.sqrt(p_v / (mc_runs * batch_size - 1.))
    return px_mean.item(), px_mean.item() - tmp, px_mean.item()+tmp
    del neural_net
    torch.cuda.empty_cache()

path = os.getcwd()
K,T,N,r,delta,sigma,batch_size, = 100,3,9,0.05,0.1,0.2,8192
#N = 12
#ua_model = 'BarrierBS'
ua_model = 'BS'
runs = 3
#barrier = 'MBRC'
barrier = []
symm = True

for d in [2, 3, 5, 10, 20, 30, 50, 100, 200, 500]:
    num_neurons = d+40
    training_steps, mc_runs_px = (3000+d), 500
    lr_boundaries = [500 + d // 5, 1500 + 3 * d // 5]
    if d <= 30:
        bsize = batch_size // 512 * 8
    elif d <= 100:
        bsize = batch_size // 2048 * 8
    elif d <= 200:
        bsize = batch_size // 4096 * 8
    else:
        bsize = batch_size // 8192 * 8
    if barrier == []:
        for s_0 in [90., 100., 110.]:
            V_L_S = 0 
            C_L_S = 0
            C_U_S = 0
            print(f'For d = {d}, S0 = {s_0}:')
            t0 = time.time()
            for _ in range(runs):
                
                X, t = sample_paths(bsize, d, s_0, 0, ua_model, symm)
                
                V_L, C_L, C_U = train_and_price(X, g(t,X, barrier), N, bsize, num_neurons, training_steps, mc_runs_px,
                                    lr_boundaries, path, barrier, epsilon=0.1, dtype=torch.float32)
                
                V_L_S += V_L
                C_L_S += C_L
                C_U_S += C_U
                text = '\n Params: N = '+str(N)+' | d = '+str(d)+' | V = '+ str(V_L)+' CI_L = '+str(C_L)+' C_I_U = '+str(C_U)
            t1 = time.time()
            write_file('\n For d = {:5.0f}, S0 = {:3.3f}:\n'.format(d, s_0))
            text = '\n Params: time = '+str(int(t1-t0))+' seconds' + 'N = '+str(N)+' | d = '+str(d)+' | V = '+ str(V_L_S/runs)+' CI_L = '+str(C_L_S/runs)+' C_I_U = '+str(C_U_S/runs)
            write_file(text)
            write_table(ua_model, N,V_L_S/runs,C_L_S/runs, C_U_S/runs)
    else:
        s_0 = 100.
        for rho in [0.6, 0.1]:
             V_L_S = 0 
             C_L_S = 0
             C_U_S = 0
             print(f'For d = {d}, rho = {rho}:')
             t0 = time.time()
             for _ in range(runs):
                 X, t = sample_paths(bsize, d, s_0, rho, ua_model = "BarrierBS", symm = False)
                 V_L, C_L, C_U = train_and_price(X, g(t,X, barrier), N, bsize, num_neurons, training_steps, mc_runs_px,
                                     lr_boundaries, path, barrier, epsilon=0.1, dtype=torch.float32)
                 V_L_S += V_L
                 C_L_S += C_L
                 C_U_S += C_U
                 text = '\n Params: N = '+str(N)+' | d = '+str(d)+' | V = '+ str(V_L)+' CI_L = '+str(C_L)+' C_I_U = '+str(C_U)
             
             t1 = time.time()
             write_file('\n For d = {:5.0f}, rho = {:3.3f}:\n'.format(d, rho))
             text = '\n Params: time = '+str(int(t1-t0))+' seconds' + 'N = '+str(N)+' | d = '+str(d)+' | V = '+ str(V_L_S/runs)+' CI_L = '+str(C_L_S/runs)+' C_I_U = '+str(C_U_S/runs)
             write_file(text)
             write_table(ua_model, N,V_L_S/runs,C_L_S/runs, C_U_S/runs)