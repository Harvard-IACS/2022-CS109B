import numpy as np
import matplotlib.pyplot as plt


def comparison_plot(rms=0,ada=0,lrs=0):
    # Plot the effective learning rate over epochs
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.plot(rms, linewidth=3, color='#332288',label = f'RMS prop: Steps = {len(rms)}')

    ax.plot(lrs[:len(rms)], linewidth=3, color='#882255',label = f'LR Decay: Steps = {len(lrs)}',alpha = 0.6)

    ax.plot(ada[:len(rms)], linewidth=3, color='#88CCEE',label = f'AdaGrad: Steps = {len(ada)}')
    ax.grid(alpha=0.3)
    ax.set_ylabel('$\epsilon$', fontsize=16)
    ax.set_xlabel('$Steps$', fontsize=16)
    ax.legend(fontsize=12)


    
def landscapes_plot(x,y,Ws_decay,Ws_ada,Ws_rms,rms=0,ada=0,lrs=0):
    
    fig, axes = plt.subplots(1,3,figsize=(14,4),sharey=True)
    
    names = ['LR Decay','Adagrad','RMS Prop']
    Ws = [Ws_decay, Ws_ada,Ws_rms]
    steps = [lrs,ada,rms]
    for i,ax in enumerate(axes):

        # Plot the original data
        ax.plot(x, y, color='black', alpha=0.6, linewidth=2)
        ax.scatter(np.array(Ws[i]), get_response_data(np.array(Ws[i])), s=150, label='Transition', color='#FDB6AA', alpha=0.6)

        # Plot the starting point
        ax.scatter(Ws[i][0], get_response_data(Ws[i][0]), c='#009193', s=150, label='Start', alpha=0.5, edgecolor='black')

        # Plot the ending point
        ax.scatter(Ws[i][-1], get_response_data(Ws[i][-1]), c='#7A81FF', s=150, label='End',edgecolor='black')
        ax.set_xlabel("$x$", fontsize=16)
        ax.set_ylabel("$y$", fontsize=16)
        ax.legend(loc='best');
        ax.set_title(f'Loss landscape for {names[i]} \n For {len(steps[i])} number of steps' ,fontsize=14);    
    
def der_polynomial(coeffs,u):
    y = 0
    for i,coeff in enumerate(coeffs[1:]):
        y -= -(i+1)*(-u)**i*coeff
    return y/10**10

def polynomial(coeffs,u):
    y = 0
    for i,coeff in enumerate(coeffs):
        y -= (-u)**i*coeff
    return y/10**10


# Function to compute the response data given the predictor data
def get_response_data(x):
    return polynomial((2059200,-68101200,57193,+40020520,57884673,-68554668,12592256.25,8308217.25,-2505009.75,
                      -375392.75,131853.75,9486.75,-2588.25,-164.25,15,1),x)

# Function to compute the derivative
def derivative(x):
    return der_polynomial((2059200,-68101200,57193,+40020520,57884673,-68554668,12592256.25,8308217.25,-2505009.75,
                      -375392.75,131853.75,9486.75,-2588.25,-164.25,15,1),x)

# # Function to compute the response data given the predictor data
# def get_response_data(x):
#     return np.cos(x) * np.exp(-x/10)

# # Function to compute the derivative
# def derivative(x):
#     return (-0.1 * ((np.exp(-x/10))* (10*np.sin(x) + np.cos(x)) ))


def lr_decay(W, epsilon,decay_rate = 0.1,delta=1e-8):

    #Variable to store the W value before update. This will help to check for convergence. 
    W_prev = None
    t = 0
    
    Ws = [W]
    lrs = [epsilon]
    
    # Perform the update until convergence
    # Convergence is said to have taken place if the previous and updated weights are the same
    while (W_prev != W):
        
        # Increment the counter t for each iteration
        t += 1
        
        # Compute the gradient by calling the derivative function
        g = derivative(W)  
         
        
        # Save the W value in W_prev before update
        W_prev = W   
        
        epsilon = (1-decay_rate)*epsilon
        
        # Update the parameters based on the equations given in the instructions
        W = W - epsilon*g
        
        # Append the new weight list with the udpated weight value
        Ws.append(W)
        lrs.append(epsilon)
        
    return Ws,lrs

def rms_prop(W, epsilon, rho=0.999, delta=1e-8):

    #Variable to store the W value before update. This will help to check for convergence. 
    W_prev = 0
    
    #Inititalise v and r to zero
    r = 0 
    
    # t is the iteration counter that will be used in the bias correction equations 
    t = 0

    # Save the current weights to a new list and append the updated weights in each iteration to the same
    Ws = [W]
    lrs = [epsilon]
    
    # Perform the update until convergence
    # Convergence is said to have taken place if the previous and updated weights are the same
    while (np.abs(W_prev -W) > delta):
        
        # Increment the counter t for each iteration
        t += 1
        
        # Compute the gradient by calling the derivative function
        g = derivative(W)  
         
        
        # Update the r, the moving average of the  sqaured gradient according to the equation given in the instructions
        r = rho*r + (1-rho)*(g**2)
        
        # According the the bias correct equations get the corrected v and r values     
        r_bias_corr = r/(1-(rho**t))        
        
        # Save the W value in W_prev before update
        W_prev = W                            
        
        # Update the parameters based on the equations given in the instructions
        W = W - (epsilon*g/(np.sqrt(r_bias_corr)+delta))   
        
        # Append the new weight list with the udpated weight value
        Ws.append(W)
        lrs.append((epsilon/(np.sqrt(r_bias_corr)+delta)))
        
    return Ws,lrs


def adagrad(W, epsilon, delta=1e-8):

    #Variable to store the W value before update. This will help to check for convergence. 
    W_prev = 0
    
    #Inititalise v and r to zero
    r = 0 
    
    # t is the iteration counter that will be used in the bias correction equations 
    t = 0

    # Save the current weights to a new list and append the updated weights in each iteration to the same
    Ws = [W]
    lrs = [epsilon]
    
    # Perform the update until convergence
    # Convergence is said to have taken place if the previous and updated weights are the same
    while (np.abs(W_prev -W) > delta):
        
        # Increment the counter t for each iteration
        t += 1
        
        # Compute the gradient by calling the derivative function
        g = derivative(W)  
         
        
        # Update the r, the moving average of the  sqaured gradient according to the equation given in the instructions
        r += (g**2)
        

        # Save the W value in W_prev before update
        W_prev = W                            
        
        # Update the parameters based on the equations given in the instructions
        W = W - (epsilon*g/np.sqrt(r+delta))   
        
        # Append the new weight list with the udpated weight value
        Ws.append(W)
        
        
        lrs.append((epsilon/(np.sqrt(r)+delta)))
        
    
        
    return Ws,lrs