import numpy as np
from scipy.stats import expon, poisson, norm

def Bach_X_BS(N, Nsim, T, mu, sigma, S0):
    XT = mu * T + sigma * np.sqrt(T) * np.random.randn(Nsim, 1)
    ST = S0 * np.exp(XT)
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros((N+1,Nsim))
    X[0,:] = S0*np.ones(Nsim)
    for i in range(1, N+1):
        X[i,:] = X[i-1,:] + mu*dt + sigma*np.sqrt(dt)*np.random.randn(Nsim)
    return X

def X_BS(N, Nsim, T, mu, sigma, S0):
    XT = mu * T + sigma * np.sqrt(T) * np.random.randn(Nsim, 1)
    ST = S0 * np.exp(XT)
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros((N+1,Nsim))
    for i in range(1, N+1):
        X[i,:] = X[i-1,:] + mu*dt + sigma*np.sqrt(dt)*np.random.randn(Nsim)
    return S0*np.exp(X)

def MD_X_Bach(N, T, mu, sigma, S0):
    time_step = T/N
    size = len(S0)
    rho = sigma[0][1]
    sigma_diag = np.diagonal(sigma)
    gaussian = np.random.normal(size=(N+1,size))
    path = np.zeros((N+1,size))
    path[0] = S0
    drift = (mu*time_step)
    CHOL = np.linalg.cholesky(np.eye(size)+rho*(np.ones((size,size)) - np.eye(size)))
    for i in range(1, N+1):
        for d in range(size):
            scale_cholesky_gaussian = np.dot(CHOL[d,:], gaussian[i])
            computed_share = path[i-1, d] + drift[d] + np.exp(sigma_diag[d]*np.sqrt(time_step)*scale_cholesky_gaussian)
            path[i,d] = computed_share
    return path

def MD_X_BS(N, T, mu, sigma, S0):
    time_step = T/N
    size = len(S0)
    rho = sigma[0][1]
    sigma_diag = np.diagonal(sigma)
    gaussian = np.random.normal(size=(N+1,size))
    path = np.zeros((N+1,size))
    path[0] = S0
    expo = np.exp(mu*time_step)
    CHOL = np.linalg.cholesky(np.eye(size)+rho*(np.ones((size,size)) - np.eye(size)))
    for i in range(1, N+1):
        for d in range(size):
            scale_cholesky_gaussian = np.dot(CHOL[d,:], gaussian[i])
            computed_share = path[i-1, d]*expo[d]*np.exp(sigma_diag[d]*np.sqrt(time_step)*scale_cholesky_gaussian)
            path[i,d] = computed_share
    return path

def X_BS_Merton(N,Nsim, T,mu,sigma, S0, lambdat, muJ, deltaJ):
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros((N+1, Nsim))
    NT = poisson.ppf(np.random.rand(Nsim, 1), lambdat * T)
    for j in range(Nsim):
        Tj = np.sort(T * np.random.rand(int(NT[j])))
        Z = np.random.randn(1, N)
        for i in range(1, N+1):
            X[i,j] = X[i-1,j] + mu * dt + sigma * np.sqrt(dt) * Z[0, i-1]
            for k in range(int(NT[j])):
                if (int(Tj[k]) > int(t[i-1])) & (int(Tj[k]) <= int(t[i])):
                    Y = norm.ppf(q = np.random.rand(1), loc= muJ, scale= deltaJ)
                    X[i,j] = X[i,j] + Y
    return S0*np.exp(X)

def X_BS_Kou(N,Nsim, T,mu,sigma, S0, lambdat, p, lambdap, lambdam):
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros((N+1, Nsim))
    NT = poisson.ppf(np.random.rand(Nsim, 1), lambdat * T)
    for j in range(Nsim):
        Tj = np.sort(T * np.random.rand(int(NT[j])))
        Z = np.random.randn(1, N)
        for i in range(1, N+1):
            X[i,j] = X[i-1,j] + mu * dt + sigma * np.sqrt(dt) * Z[0, i-1]
            for k in range(int(NT[j])):
                if (int(Tj[k]) > int(t[i-1])) & (int(Tj[k]) <= int(t[i])):
                    uniform_value = np.random.rand(1)
                    if uniform_value < p:
                        Y = expon.ppf(uniform_value, 1/lambdap)
                    else:
                        Y = -expon.ppf(uniform_value, 1/lambdam)
                    X[i,j] = X[i,j] + Y
    return S0*np.exp(X)


def MD_X_BS_Kou(N, Nsim, T, mu, sigma, S0, lambdat, p, lambdap, lambdam, d):
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros((N+1, Nsim, d))
    NT = poisson.ppf(np.random.rand(Nsim, 1), lambdat * T)
    for j in range(Nsim):
        Tj = np.sort(T * np.random.rand(int(NT[j])))
        Z = np.random.randn(N, d)
        for i in range(1, N+1):
            X[i,j, :] = X[i-1,j, :] + mu * dt + sigma * np.sqrt(dt) * Z[i-1, :]
            for k in range(int(NT[j])):
                if (int(Tj[k]) > int(t[i-1])) & (int(Tj[k]) <= int(t[i])):
                    uniform_value = np.random.rand(1)
                    if uniform_value < p:
                        Y = expon.ppf(uniform_value, 1/lambdap)
                    else:
                        Y = -expon.ppf(uniform_value, 1/lambdam)
                    J = np.zeros(d)
                    for l in range(d):
                        J[l] = Y * np.random.randn(1)
                    X[i,j,:] = X[i,j,:] + J
    
    return S0*np.exp(X)

def MD_X_BS_Merton(N, Nsim, T, mu, sigma, S0, lambdat, muJ, deltaJ):
    dt = T / N
    t = np.linspace(0, T, N+1)
    d = len(mu)
    X = np.zeros((N+1, Nsim, d))
    NT = poisson.ppf(np.random.rand(Nsim, 1), lambdat * T)
    for j in range(Nsim):
        Tj = np.sort(T * np.random.rand(int(NT[j])))
        Z = np.random.randn(d, N)
        for i in range(1, N+1):
            X[i,j] = X[i-1,j] + mu * dt + sigma * np.sqrt(dt) * Z[:, i-1]
            for k in range(int(NT[j])):
                if (int(Tj[k]) > int(t[i-1])) & (int(Tj[k]) <= int(t[i])):
                    Y = np.random.normal(muJ, deltaJ, d)
                    X[i,j] = X[i,j] + Y
    return S0*np.exp(X)


def Heston(N, Nsim, T, theta, k, epsilon, r, X0, rho, V0):
    V = np.zeros((N+1, Nsim))
    X = np.zeros((N+1, Nsim))
    X[0,:] = X0
    V[0,:] = V0
    Feller_condition = (epsilon**2 - 2*k*theta)
    dt = T / N
    if Feller_condition > 0: # QE Scheme
        Psi_cutoff = 1.5
        for i in range(N): # Discretise V (volatility) and calculate m, Psi, s2
            m = theta + (V[i,:] - theta)*np.exp(-k*dt)
            m2 = m**2
            s2 = V[i,:]*epsilon**2*np.exp(-k*dt)*(1-np.exp(-k*dt))/k + theta*epsilon**2*(1-np.exp(-k*dt))**2/(2*k)
            Psi = (s2)/(m2)
            index = np.where(Psi_cutoff < Psi)[0]
            
            # Exponential approx scheme if Psi > Psi_cutoff
            p_exp = (Psi[index]-1)/(Psi[index]+1)
            beta_exp = (1-p_exp)/m[index]
            U = np.random.rand(np.size(index))
            V[i+1, index] = (np.log((1-p_exp)/(1-U)/beta_exp)*(U>p_exp))
            
            # Quadratic approx scheme if 0 < Psi < Psi_cutoff
            index = np.where(Psi <= Psi_cutoff)[0]
            invPsi = 1/Psi[index]
            b2_quad = 2*invPsi - 1 + np.sqrt(2*invPsi)*np.sqrt(2*invPsi-1)
            a_quad  = m[index]/(1+b2_quad)
            V[i+1, index] = a_quad*(np.sqrt(b2_quad)+ np.random.randn(np.size(index)))**2
            
        # Central discretisation scheme
        gamma1 = 0.5
        gamma2 = 0.5
        k0 = r*dt -rho*k*theta*dt/epsilon
        k1 = gamma1*dt*(k*rho/epsilon-0.5)-rho/epsilon
        k2 = gamma2*dt*(k*rho/epsilon-0.5)+rho/epsilon
        k3 = gamma1*dt*(1-rho**2)
        k4 = gamma2*dt*(1-rho**2)
        for i in range(N):
            X[i+1,:] = np.exp(np.log(X[:,i]))+k0+k1*V[i,:]+k2*V[i+1,:]+np.sqrt(k3*V[i,:])+k4*V[i+1,:]*np.random.randn(1,Nsim)
        return X, V
    else:
        mu = [0, 0]
        X = np.zeros((N+1, Nsim))
        VC = [[1,rho],[rho,1]]
        for i in range(N):
            Z = np.random.multivariate_normal(mu,VC,Nsim)
            X[i+1,:] = X[i,:] + (r-V[i,:]/2)*dt+np.sqrt(V[i,:]*dt)*Z[:,0]
            V[i+1,:] = V[i,:] + k*(theta-V[i,:])*dt+epsilon*np.sqrt(V[i,:]*dt)*Z[:,1]
        return X0*np.exp(X), V





def MD_Heston(N, Nsim, T, theta, k, epsilon, r, X0, rho, V0):
    d = len(X0)
    V = np.zeros((Nsim, N, d))
    X = np.zeros((Nsim, N+1, d))
    X[:,:,0] = X0
    V[:,:,0] = V0
    # Feller condition
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
            for i in range(N):
                X[:,i+1] = np.exp(np.log(X[:,i]))+k0+k1*V[:,i]+k2*V[:,i+1]+np.sqrt(k3*V[:,i])+k4*V[:,i+1]*np.random.randn(1,Nsim)
            return X
    else:
            mu = [0, 0]
            VC = [[1,rho],[rho,1]]
            for i in range(N):
                Z = np.random.multivariate_normal(mu,VC,Nsim)
                X[:,i+1] = X[:,i] + (r-V[:,i]/2)*dt+np.sqrt(V[:,i]*dt)*Z[:,1]
                V[:,i+1] = V[:,i] + k*(theta-V[:,i])*dt+epsilon*np.sqrt(V[:,i]*dt)*Z[:,2]
            return np.exp(X)