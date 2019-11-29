#A code for print network status when different data set input samples are applied
# Plot of Individual neural state for the interation that you defined in load and print
# Plot of SVD in 2 and 3D
# Plot of PCA in 3D

import time
import numpy as np
import matplotlib.pyplot as plt
from pylab import grid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import grid
from keras import backend as K
from keras.models import Sequential, Model

# pca part:
from sklearn.decomposition import PCA
import sklearn.decomposition
from scipy import signal

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def plot_sample(sample_number,input_number,neurons,x_train,y_train,model,seq_dur,i,plot_dir,f):

    frecuencias=[]
    
    seq_dur                        = len(x_train[sample_number, :, 0])
    test                           = x_train[sample_number:sample_number+1,:,:]
    colors                         = cm.rainbow(np.linspace(0, 1, neurons+1))
    y_pred                         = model.predict(test)

    ###################################

    # Status for the sample value at the layer indicated
    capa=0

    #Primer capa:
    get_0_layer_output = K.function([model.layers[capa].input], [model.layers[capa].output])
    
    layer_output= get_0_layer_output([test])[capa]
        
    #segunda capa:
    get_1_layer_output = K.function([model.layers[capa].input], [model.layers[capa].output])
    #layer_output_1     = get_1_layer_output([test])[capa]
                
    #tercer capa
    #get_2_layer_output = K.function([model.layers[0].input], [model.layers[2].output])
    #layer_output_2     = get_2_layer_output([test])[0]
    
    layer_output_T       = layer_output.T

    print"layer_output",layer_output_T
    #layer_output_1_T     = layer_output_1.T
    #layer_output_2_T    = layer_output_2.T
    array_red_list       = []

    ####################################
    y_pred              = model.predict(test)

    # To generate the Populational Analysis

    for ii in np.arange(0,neurons,1):
        neurona_serie = np.reshape(layer_output_T[ii], len(layer_output_T[ii]))
        array_red_list.append(neurona_serie)
    
    array_red = np.asarray(array_red_list)
    sdv       = sklearn.decomposition.TruncatedSVD(n_components=2)
    sdv_3d    = sklearn.decomposition.TruncatedSVD(n_components=3)
    X_2d      = sdv.fit_transform(array_red.T)
    X_3d      = sdv_3d.fit_transform(array_red.T)
    pca       = PCA(n_components=3)
    X_pca_    = pca.fit(array_red)
    X_pca     = pca.components_

    ####################################

    print"------------"
    ordeno_primero_x=X_pca[0]
    ordeno_primero_y=X_pca[1]
    ordeno_primero_z=X_pca[2]
              
    ####################################
    # How many 3d angular views you want to define
    yy        = np.arange(70,80,10)

    ####################################   
    
    kk=70       

    fig     = plt.figure(figsize=cm2inch(19,7))

    ###
    plt.subplot(2, 2, 1) 
    plt.plot(test[0,:,0],color='g',label='Input Reset')
    plt.plot(test[0,:,1],color='pink',label='Input Set')
    plt.plot(y_train[sample_number,:, 0],color='grey',linewidth=3,label='Target Output')  
    plt.plot(y_pred[0,:, 0], color='r',linewidth=2,label=' Output')
    plt.xlim(0,seq_dur+1)
    plt.ylim([-0.5, 1.5])
    plt.yticks([])
    plt.xticks(np.arange(0,seq_dur+1,50),fontsize = 8)
    plt.legend(fontsize= 4,loc=1)

    ### 
    plt.subplot(2, 2, 3) 
    plt.plot(test[0,:,0],color='g',label='Input A')
    #plt.plot(test[0,:,0],color='pink',label='Input B')
    
    for ii in np.arange(0,int(neurons/2),1):

        imprimir=layer_output_T[ii].T[0]
        plt.plot(layer_output_T[ii],color=colors[ii],linewidth=1)
        plt.xlim(-1,seq_dur+1)         
        plt.ylim([-1.5, 1.6])
        plt.xlabel('time [mS]',fontsize = 10)
        plt.yticks([])
        plt.xticks(np.arange(0,seq_dur+1,50),fontsize = 8)
    plt.plot(y_pred[0,:, 0], color='r',linewidth=2,label=' Output\n 25 individual states')    
    plt.legend(fontsize= 3.5,loc=3)

    
    #plt.subplot(2, 2, 4) 
    fig.suptitle("Time series and PCA 3D plot",fontsize = 12)
    ax = fig.add_subplot(122, projection='3d')
    
    if sample_number==0:
       ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=70,c='r',marker="^",label=' Start ')     
       ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=70,c='b',marker="^",label=' Stop ')
    if sample_number==3:
       ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=70,c='r',marker="^",label=' Start ')     
       ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=70,c='b',marker="^",label=' Stop ')
    if sample_number !=3 and sample_number!=0:
       ax.plot(X_pca[0],X_pca[1],X_pca[2],color='salmon',marker="p",zorder=2,markersize=2,label="3d plot")
       ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=70,c='r',marker="^",label=' Start ')     
       ax.scatter(ordeno_primero_x[-1],ordeno_primero_y[-1],ordeno_primero_z[-1],s=70,c='b',marker="^",label=' Stop ')
    '''   

    ax.plot(X_pca[0],X_pca[1],X_pca[2],color='salmon',marker="p",zorder=2,markersize=2,label="3d plot")
    ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=70,c='r',marker="^",label=' Start ')     
    ax.scatter(ordeno_primero_x[-1],ordeno_primero_y[-1],ordeno_primero_z[-1],s=70,c='b',marker="^",label=' Stop ')
    '''
    ax.set_xlabel('pca 1 [arb. units]',size=10)
    ax.set_ylabel('pca 2 [arb. units]',size=10)
    ax.set_zlabel('pca 3 [arb. units]',size=10)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_zticks(())
    ax.view_init(elev=10, azim=kk)

    ax.legend(fontsize= 5)
    fig.text(0.1, 0.5, 'Amplitude [Arb. Units]', va='center', ha='center', rotation='vertical', fontsize=10)
    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_pca_3D_"+str(capa)+"_individual_neurons_state_"+str(i)+'_'+str(kk)+"_"+str(f)+".png"
    plt.savefig(figname,dpi=300, bbox_inches = 'tight') 
    plt.close()      
    ####################################
 
