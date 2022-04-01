######## Plotting neural networks using matplotlib #######

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Matplotlib function to plot ellipse on a plot

def ellipse(x, y, width=0.25/1.25,height=0.7/1.25,color = '#7A7A7A'):
    from matplotlib.patches import Ellipse
    from matplotlib.patheffects import withStroke
    ellipse = Ellipse((x, y), width=width,height=height, clip_on=False, zorder=10, linewidth=1,
                    edgecolor='black', facecolor=color,
                    path_effects=[withStroke(linewidth=5, foreground='w')])
    ax = plt.gca()
    ax.add_artist(ellipse)
    

# Custom function that takes network weights and makes a parallel coordinates plot

def plot_weights(mlp,epochnum=0,n_hidden=50):
    
    # First get the weights from the network 
    # Note: We choose only even numbers because we skip the bias terms
    
    weights = {}
    for i in range(0,5,2):
        weights[i] = mlp.get_weights()[i]
        
    # Next make a dataframe with weight values and associated layers  
    
    df = pd.DataFrame(columns= ['Layer 1','Layer 2','Layer 3','y'])
    df['Layer 1'] = np.array(list(weights[0].flatten())*n_hidden).reshape(n_hidden,n_hidden).T.reshape(n_hidden**2,)
    df['Layer 2'] = weights[2].flatten()
    df['Layer 3'] = list(weights[4].flatten())*n_hidden  
    
    # Value of input and output chosen as 3 inorder to faciliate plotting 
    
    input_location = 3
    
    df['Input'] = input_location
    
    df['Hidden 1'] = np.array(list(range(1,n_hidden+1))*n_hidden).reshape(n_hidden,n_hidden).T.reshape(n_hidden**2,)

    df['Hidden 2'] = list(range(1,n_hidden+1))*n_hidden

    df['Output'] = input_location
    
    # Naming the 'y' column as 'Weights' because parallel coordinates plot is for categorical data 
    
    df.y = 'Weights'
    
    # Below code is for the parallel coordinates plot    
    
    with plt.xkcd(scale=0.3):
        
        # Setting figure size and fontsize         
        fig = plt.figure(figsize=(10,6))
        plt.rcParams.update({'font.size': 16})

        numweights = 0        
        if epochnum == 0:
            for i in range(1,4): 
                numweights+=len(df[f'Layer {i}'].unique()) 

            pd.plotting.parallel_coordinates(df, "y",
                                             color=["#595959"],
                                             cols = ['Input','Hidden 1','Hidden 2','Output'],
                                             alpha=1,lw=3) 
            if n_hidden == 5:
                plt.title(f'{numweights} non-zero weights before training')

                # Plotting ellipses because matplotlib figure is a rectangle, hence circles will become ovals 

                # Input 
                ellipse(0, input_location,color = '#E6B9B8')
                # Output
                ellipse(3, input_location,color = '#C3D69B')

                # 2 Hidden layers, each with two nodes

                for i in range(5):
                    ellipse(1, 1+i,color = '#C6D9F1')
                    ellipse(2,1+i,color = '#C6D9F1')

                plt.yticks([])
                plt.ylim([0.5,5.5])

            
            
        # Numweights computation based on the remaining significant weight values 
        # i ranges from 1 - 4 because we have 3 layers
        else:
            for i in range(1,4): 
                numweights+=len(df[(df['Layer 1'].abs() > 0.1) & (df['Layer 2'].abs() > 0.1)][f'Layer {i}'].unique()) 

            pd.plotting.parallel_coordinates(df[(df['Layer 1'].abs() > 0.1) & (df['Layer 2'].abs() > 0.1)], "y",
                                             color=["#595959"],
                                             cols = ['Input','Hidden 1','Hidden 2','Output'],
                                             alpha=1,lw=3) 

            if n_hidden == 5:
                plt.title(f'{numweights} non-zero weights after {epochnum} epochs ')

                # Plotting ellipses because matplotlib figure is a rectangle, hence circles will become ovals 

                # Input 
                ellipse(0, input_location,color = '#E6B9B8')
                # Output
                ellipse(3, input_location,color = '#C3D69B')

                # 2 Hidden layers, each with two nodes

                for i in range(5):
                    ellipse(1, 1+i,color = '#C6D9F1')
                    ellipse(2,1+i,color = '#C6D9F1')

                plt.yticks([])
                plt.ylim([0.5,5.5])
            
        