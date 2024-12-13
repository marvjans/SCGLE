#Required Packages
import numpy as np

## Truncate the fourier series
def P_N(u,N):
    u_shape=np.asarray(u.shape)
    N_0=u_shape[0]
    P_N_shape=u_shape
    P_N_shape[0]=N
    P_Nu=np.zeros(P_N_shape,dtype=complex)
    N_half=N//2
    if 2*N_half-N==0:
        P_Nu[:N_half]=u[:N_half]
        P_Nu[N_half:]=u[N_0-N_half:]
    else:
        P_Nu[:N_half+1]=u[:N_half+1]
        P_Nu[N_half+1:]=u[N_0-N_half:]
    return P_Nu    

## Residue of the truncation
def I_min_P_N(u,N):
    N_0=len(u)
    I_min_P_Nu=np.copy(u)
    N_half=N//2
    if 2*N_half-N==0:
        I_min_P_Nu[:N_half]=np.zeros(N_half)
    else:
        I_min_P_Nu[:N_half+1]=np.zeros(N_half+1)
    I_min_P_Nu[N_0-N_half:]=np.zeros(N_half)
    return I_min_P_Nu   

## Generating matrices for discretized A and covariance matrix of stochastic convolution
def generate_matrices(dt,nu,sigma,N):
    # Calculating e^-lambda_ell and variance of the noise
    
    diag=np.zeros(N,dtype=complex)
    A_diag=np.zeros(N,dtype=complex)
    var=np.zeros(N)
    diag[0]=1
    A_diag[0]=dt
    var[0]=dt
    
    for i in range(N//2):     #Creating diagonal elements for solving diffusion part exactly
        diag[i+1]=np.e**(-(2*np.pi*(i+1))**2*dt*(1+1j*nu))
        A_diag[i+1]=(1-diag[i+1])/(2*np.pi*(i+1))**2/(1+1j*nu)
        var[i+1]=(1-np.e**(-2*(2*np.pi*(i+1))**2*dt))/(2*(2*np.pi*(i+1))**2)
    for i in range(N//2):     #Creating diagonal elements for solving diffusion part exactly
        diag[N-i-1]=np.e**(-(2*np.pi*(i+1))**2*dt*(1+1j*nu))
        A_diag[N-i-1]=(1-diag[i+1])/(2*np.pi*(i+1))**2/(1+1j*nu)
        var[N-i-1]=(1-np.e**(-2*(2*np.pi*(i+1))**2*dt))/(2*(2*np.pi*(i+1))**2)
    return diag,A_diag,var

## Generates q_k such that regularity assumptions are satified
def sqrt_q_generate(reg,epsilon,N):
    q=np.ones(N)
    r=2*reg+1+epsilon #Calculate rate of q to satisfy regularity condition
    for i in range(N//2):
        q[i+1]=1/(i+1)**(r)
    for i in range(N//2):
        q[N-i-1]=1/(i+1)**(r)
    return np.sqrt(q)

def get_sqrt_qvar(sqrt_q,var):
    return sqrt_q*np.sqrt(var)

def get_sqrt_qdt(sqrt_q,dt):
    return sqrt_q*np.sqrt(dt)

#Gives Stochastic Convolution term drawing from W
def get_W(N,sqrt_qdt,M,sigma):
    
    W=np.zeros([M+1,N],dtype='complex')
    W[0]=np.zeros(N)
    for i in range(M):
        noise_R=np.random.normal(0,sqrt_qdt)
        noise_C=np.random.normal(0,sqrt_qdt)
        W[i+1]=(noise_R+1j*noise_C)*N
    return sigma*W


#Gives Stochastic Convolution term drawing from stochastic convolution
def Stoch_conv(N,diag,sqrt_qvar,M,sigma):
    
    stoch_conv=np.zeros([M+1,N],dtype='complex')
    stoch_conv[0]=np.zeros(N)
    for i in range(M):
        noise_R=np.random.normal(0,sqrt_qvar)
        noise_C=np.random.normal(0,sqrt_qvar)
        stoch_conv[i+1]=diag*stoch_conv[i]+(noise_R+1j*noise_C)*N
    return sigma*stoch_conv

def Exact_splitting_method(dt,M,N1,N2,R,mu,sigma,diag,sqrt_qvar,u_0,k):  
    # Initial value
    hu_t1=np.copy(u_0)  #initial value
    hu_t2=P_N(u_0,N2)/N1*N2
    alpha1=np.e**(-2*R*dt)
    beta1=(np.e**(2*R*dt)-1)/R      
    alpha2=np.e**(-2*k*R*dt)
    beta2=(np.e**(2*k*R*dt)-1)/R
    diag2=P_N(diag,N2)        
    #Solver
    for i in range(M//k):
        #Create noise
        Noise=Stoch_conv(N1,diag,sqrt_qvar,4,sigma)
        for j in range(k):
            #Non-linear part for N_exact
            u1=np.fft.ifft(hu_t1)
            Rad_old1=np.abs(u1)
            Rad_new1=Rad_old1*np.sqrt(R/(Rad_old1**2-alpha1*(Rad_old1**2-R)))
            Deg_new1=np.angle(u1)-mu*0.5*np.log(Rad_old1**2*beta1+1)
            u1=Rad_new1*np.e**(1j*Deg_new1)
            hu_t1=np.fft.fft(u1)

            #Diffusion and Noise part
            hu_t1=diag*(hu_t1-Noise[j])+Noise[j+1]
        
        u2=np.fft.ifft(hu_t2)
        Rad_old2=np.abs(u2)
        Rad_new2=Rad_old2*np.sqrt(R/(Rad_old2**2-alpha2*(Rad_old2**2-R)))
        Deg_new2=np.angle(u2)-mu*0.5*np.log(Rad_old2**2*beta2+1)
        u2=Rad_new2*np.e**(1j*Deg_new2)
        hu_t2=np.fft.fft(u2)

        #Diffusion and Noise part
        hu_t2=(diag2**k)*hu_t2+P_N(Noise[-1],N2)*N2/N1
    P_error=np.sum(np.abs(I_min_P_N(hu_t1/N1, N2)**2))
    N_error=np.sum(np.abs(P_N(hu_t1/N1, N2)-hu_t2/(N2))**2)
    return P_error,N_error

def Exact_splitting_accelerated_method(dt,M,N1,N2,R,mu,sigma,diag,A_diag1,A_diag2,sqrt_qvar,u_0):  
    # Initial value
    hu_t1=np.copy(u_0)  #initial value
    hu_t2=P_N(u_0,N2)/N1*N2  
    alpha1=np.e**(-2*R*dt)
    beta1=(np.e**(2*R*dt)-1)/R      
    alpha2=np.e**(-4*R*dt)
    beta2=(np.e**(4*R*dt)-1)/R
    diag2=P_N(diag,N2)
    #Solver
    for i in range(M//2):
        #Create noise
        Noise=Stoch_conv(N1,diag,sqrt_qvar,2,sigma)
        #Non-linear part for N_exact
        u1=np.fft.ifft(hu_t1)
        Rad_old1=np.abs(u1)
        Rad_new1=Rad_old1*np.sqrt(R/(Rad_old1**2-alpha1*(Rad_old1**2-R)))
        Deg_new1=np.angle(u1)-mu*0.5*np.log(Rad_old1**2*beta1+1)
        u1=Rad_new1*np.e**(1j*Deg_new1)
        hu_tf1=np.fft.fft(u1)

        #Diffusion and Noise part
        hu_t1=diag*(hu_t1)+A_diag1*(hu_tf1-hu_t1)/dt+Noise[1]
        
        u1=np.fft.ifft(hu_t1)
        Rad_old1=np.abs(u1)
        Rad_new1=Rad_old1*np.sqrt(R/(Rad_old1**2-alpha1*(Rad_old1**2-R)))
        Deg_new1=np.angle(u1)-mu*0.5*np.log(Rad_old1**2*beta1+1)
        u1=Rad_new1*np.e**(1j*Deg_new1)
        hu_tf1=np.fft.fft(u1)

        #Diffusion and Noise part
        hu_t1=diag*(hu_t1-Noise[1])+A_diag1*(hu_tf1-hu_t1)/dt+Noise[2]
        
        u2=np.fft.ifft(hu_t2)
        Rad_old2=np.abs(u2)
        Rad_new2=Rad_old2*np.sqrt(R/(Rad_old2**2-alpha2*(Rad_old2**2-R)))
        Deg_new2=np.angle(u2)-mu*0.5*np.log(Rad_old2**2*beta2+1)
        u2=Rad_new2*np.e**(1j*Deg_new2)
        hu_tf2=np.fft.fft(u2)

        #Diffusion and Noise part
        hu_t2=diag2*diag2*hu_t2+A_diag2*(hu_tf2-hu_t2)/(2*dt)+P_N(Noise[2],N2)*N2/N1
    P_error=np.sum(np.abs(I_min_P_N(hu_t1/N1, N2)**2))
    N_error=np.sum(np.abs(P_N(hu_t1/N1, N2)-hu_t2/(N2))**2)
    return P_error,N_error

def Explicit_splitting_method(dt,M,N1,N2,R,mu,sigma,diag,A_diag1,A_diag2,sqrt_qdt,u_0,k):  
    # Initial value
    hu_t1=np.copy(u_0)  #initial value
    hu_t2=P_N(u_0,N2)/N1*N2   
    alpha1=np.e**(-2*R*dt)
    beta1=(np.e**(2*R*dt)-1)/R      
    alpha2=np.e**(-2*k*R*dt)
    beta2=(np.e**(2*k*R*dt)-1)/R
    diag2=P_N(diag,N2)     
    #Solver
    for i in range(M//k):
        #Create noise
        Noise=get_W(N1,sqrt_qdt,k,sigma)
        for j in range(k):
            #Non-linear part for N_exact
            u1=np.fft.ifft(hu_t1)
            Rad_old1=np.abs(u1)
            Rad_new1=Rad_old1*np.sqrt(R/(Rad_old1**2-alpha1*(Rad_old1**2-R)))
            Deg_new1=np.angle(u1)-mu*0.5*np.log(Rad_old1**2*beta1+1)
            u1=Rad_new1*np.e**(1j*Deg_new1)
            hu_t1=np.fft.fft(u1)

            #Diffusion and Noise part
            hu_t1=diag*(hu_t1)+diag*Noise[j+1]
            
        
        u2=np.fft.ifft(hu_t2)
        Rad_old2=np.abs(u2)
        Rad_new2=Rad_old2*np.sqrt(R/(Rad_old2**2-alpha2*(Rad_old2**2-R)))
        Deg_new2=np.angle(u2)-mu*0.5*np.log(Rad_old2**2*beta2+1)
        u2=Rad_new2*np.e**(1j*Deg_new2)
        hu_t2=np.fft.fft(u2)
        for j in range(k):
            hu_t2+=P_N(Noise[j+1],N2)*N2/N1

        #Diffusion and Noise part
        hu_t2=(diag2**k)*hu_t2
    P_error=np.sum(np.abs(I_min_P_N(hu_t1/N1, N2)**2))
    N_error=np.sum(np.abs(P_N(hu_t1/N1, N2)-hu_t2/(N2))**2)
    return P_error,N_error

def Tamed_method(dt,M,N1,N2,R,mu,sigma,diag,A_diag1,A_diag2,sqrt_qvar,u_0,k):  
    # Initial value
    hu_t1=np.copy(u_0)  #initial value
    hu_t2=P_N(u_0,N2)/N1*N2
    diag2=P_N(diag,N2)        
    #Solver
    for i in range(M//k):
        #Create noise
        Noise=Stoch_conv(N1,diag,sqrt_qvar,k,sigma)
        for j in range(k):
            #Non-linear part for N_exact
            u1=np.fft.ifft(hu_t1)
            f_dt=R*u1-(1+1j*mu)*u1*np.abs(u1)**2
            hu_tf1=np.fft.fft(f_dt)

            #Diffusion and Noise part
            Reaction=A_diag1*hu_tf1/(1+dt*np.sqrt(np.sum(np.abs(hu_tf1/N1)**2)))
            hu_t1=diag*(hu_t1-Noise[j])+Reaction+Noise[j+1]
        
        u2=np.fft.ifft(hu_t2)
        f_dt=R*u2-(1+1j*mu)*u2*np.abs(u2)**2
        hu_tf2=np.fft.fft(f_dt)

        #Diffusion and Noise part
        Reaction=A_diag2*hu_tf2/(1+k*dt*np.sqrt(np.sum(np.abs(hu_tf2/N2) **2)))
        hu_t2=(diag2**k)*hu_t2+Reaction+P_N(Noise[-1],N2)*N2/N1
    P_error=np.sum(np.abs(I_min_P_N(hu_t1/N1, N2)**2))
    N_error=np.sum(np.abs(P_N(hu_t1/N1, N2)-hu_t2/(N2))**2)
    return P_error,N_error