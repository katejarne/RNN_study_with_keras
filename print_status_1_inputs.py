#A code for print network status when different data set input samples are applied
# Plot of Individual neural state for the interation that you defined in load and print
# Plot of SVD in 2 and 3D
# Plot of PCA in 3D

import numpy as np
import matplotlib.pyplot as plt
from pylab import grid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import grid
from keras import backend as K

# pca part:
from sklearn.decomposition import PCA
import sklearn.decomposition

def plot_sample(sample_number,input_number,neurons,x_train,y_train,model,seq_dur,i,plot_dir):

    #seq_dur                        = 1200
    test                           = x_train[sample_number:sample_number+1,:,:]
    colors                         = cm.rainbow(np.linspace(0, 1, neurons+1))
    y_pred                         = model.predict(test)

    ###################################

    # Status for the sample value at the layer indicated
    capa=0

    #Primer capa:
    get_0_layer_output = K.function([model.layers[capa].input], [model.layers[capa].output])
    layer_output= get_0_layer_output([test])[capa]
    #layer_output.append(layer_output_m)
    
    #segunda capa:
    get_1_layer_output = K.function([model.layers[capa].input], [model.layers[capa].output])
    layer_output_1     = get_1_layer_output([test])[capa]
  
    #tercer capa
    #get_2_layer_output = K.function([model.layers[0].input], [model.layers[2].output])
    #layer_output_2     = get_2_layer_output([test])[0]

    
    layer_output_T       = layer_output.T
    layer_output_1_T     = layer_output_1.T
    #layer_output_2_T    = layer_output_2.T
    array_red_list       = []

    ####################################
    print("len",len(layer_output[capa]))
    y_pred              = model.predict(test)

    # To generate the Populational Analysis

    for ii in np.arange(0,neurons,1):
        neurona_serie = np.reshape(layer_output_T[ii], len(layer_output_T[ii]))
        print"neurona_serie",neurona_serie
        array_red_list.append(neurona_serie)
    
    print"array_red_list",array_red_list
    array_red = np.asarray(array_red_list)
    sdv       = sklearn.decomposition.TruncatedSVD(n_components=2)
    sdv_3d    = sklearn.decomposition.TruncatedSVD(n_components=3)
    X_2d      = sdv.fit_transform(array_red.T)
    X_3d      = sdv_3d.fit_transform(array_red.T)
    print('size X_2d',X_2d.size)
    print('size X_3d',X_3d.size)

    pca     = PCA(n_components=3)
    X_pca_  = pca.fit(array_red)
    X_pca   = pca.components_

    print("X_pca: ",X_pca)

    ####################################

    fig = plt.figure()
    fig.suptitle("SDV Network Population Analysis",fontsize = 20)
    ax1 = fig.add_subplot(111)
    
    #ax1.scatter(X_vs_2d[:,0],X_vs_2d[:,1],c='m',label='sdv pesos')
    #ax1.scatter(X_2d[:,0],X_2d[:,1],c='g', lw=3,label='Dim Reduction of the network')
    ax1.plot(X_2d[:,0],X_2d[:,1],c='c',marker="p",zorder=2,label='Dim Reduction of the network')
    ax1.scatter(X_2d[0,0],X_2d[0,1],c='r',marker='^',s=70,label='start')
    ax1.scatter(X_2d[-1,0],X_2d[-1,1],c='b',marker='^',s=70,label='stop')
    plt.legend(loc='upper left',fontsize= 'x-small');
    plt.ylabel('C1')
    plt.xlabel('C2')
    
    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_sdv_"+str(capa)+"_individual_neurons_state_"+str(i)+".png" 
    #plt.savefig(figname,dpi=200)
    plt.close()
    #pp.show()

    ####################################

    print"------------"
    ordeno_primero_x=X_pca[0]
    ordeno_primero_y=X_pca[1]
    ordeno_primero_z=X_pca[2]

    fig = plt.figure()
    fig.suptitle("PCA Network Population Analysis",fontsize = 20)
    ax1 = fig.add_subplot(111)
    
    ax1.scatter(X_pca[0],X_pca[1],c='g',marker="p",zorder=2,label='Dim Reduction of the network')    
    ax1.scatter(ordeno_primero_x[0],ordeno_primero_y[0],s=70,c='r',marker="^",label='start')
    ax1.scatter(ordeno_primero_x[-1],ordeno_primero_y[-1],s=70,c='b',marker="^",label='stop')
    plt.legend(loc='upper left',fontsize= 'x-small');
    plt.ylabel('C1')
    plt.xlabel('C2')
    
    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_pca_"+str(capa)+"_individual_neurons_state_"+str(i)+".png" 
    #plt.savefig(figname,dpi=200)
    plt.close()
    #pp.show()
              
    ####################################
    # How many 3d angular views you want to define

    #yy        = np.arange(0,360,10)
    yy        = np.arange(70,80,10)
    for ii,kk in enumerate(yy):
        print "ii: ",ii," kk: ",kk   
        fig     = plt.figure(figsize=(10,8))
        fig.suptitle("3D plot SDV Network Population Analysis",fontsize = 20)
        
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X_3d[:,0],X_3d[:,1],X_3d[:,2],color='g',marker="p",zorder=2,label="3 d plot")
        ax.scatter(X_3d[0,0],X_3d[0,1],X_3d[0,2],c='r',marker="^",label='start')
        ax.scatter(X_3d[-1,0],X_3d[-1,1],X_3d[-1,2],c='b',marker="^",label='stop')
        #ax.scatter(X_3d[:,0],X_3d[:,1],X_3d[:,2],color='g', zorder=2,marker="p",label="3 d plot")
        ax.set_xlabel('comp 1 (arb. units)',size=16)
        ax.set_ylabel('comp 2 (arb. units)',size=16)
        ax.set_zlabel('comp 3 (arb. units)',size=16)
        ax.legend()
        ax.view_init(elev=10, azim=kk)
        figname = str(plot_dir)+"/sample_"+str(sample_number)+"_sdv_3d_"+str(capa)+"_individual_neurons_state_"+str(i)+'_'+str(kk)+".png" 
        #plt.savefig(figname,dpi=200)
        plt.close()
    
    ####################################   

            
    for ii,kk in enumerate(yy):
       print "ii: ",ii," kk: ",kk   
       fig     = plt.figure(figsize=(10,8))
       fig.suptitle("3D plot PCA",fontsize = 20)
       ax = fig.add_subplot(111, projection='3d')
       ax.plot(X_pca[0],X_pca[1],X_pca[2],color='c',marker="p",zorder=2,label="3 d plot")
       ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=100,c='r',marker="^",label='start')
       ax.scatter(ordeno_primero_x[-1],ordeno_primero_y[-1],ordeno_primero_z[-1],s=100,c='b',marker="^",label='stop')
       #ax.scatter(X_3d[-1,0],X_3d[-1,1],X_3d[-1,2],c='b',marker="^",label='stop')
       #ax.scatter(X_3d[:,0],X_3d[:,1],X_3d[:,2],color='g', zorder=2,marker="p",label="3 d plot")
       ax.set_xlabel('pca 1 (arb. units)',size=16)
       ax.set_ylabel('pca 2 (arb. units)',size=16)
       ax.set_zlabel('pca 3 (arb. units)',size=16)
       ax.set_zlim(-0.42,0.42)
       ax.set_ylim(-0.15,0.15)
       ax.set_xlim(-0.25,0.25)

       #### part: 
       first_in         = 30 #time to start the first stimulus   #30 #60
       stim_dur         = 20 #stimulus duration #20 #30
       out_gap          = 130 #how much lenth add to the sequence duration    #140 #100
       marker_          = 5
       gap              = 20
       ax.plot(ordeno_primero_x[0:30],ordeno_primero_y[0:30],ordeno_primero_z[0:30],markersize=marker_,color='y',zorder=2,marker="o",label='start-part')
       ax.plot(ordeno_primero_x[30:30+20],ordeno_primero_y[30:30+20],ordeno_primero_z[30:30+20],markersize=marker_,zorder=2,color='m',marker="o",label='stim-part')
       ax.plot(ordeno_primero_x[30+20:30+20+gap],ordeno_primero_y[30+20:30+20+gap],ordeno_primero_z[30+20:30+20+gap],markersize=marker_,zorder=2,color='r',marker="o",label='gap-part')
       #ax.plot(ordeno_primero_x[30+20+gap:],ordeno_primero_y[30+20+gap:],ordeno_primero_z[30+20+gap:],markersize=marker_,zorder=2,color='g',marker="o",label='last-part')

       ax.plot(ordeno_primero_x[30+20+gap:30+20+gap+20],ordeno_primero_y[30+20+gap:30+20+gap+20],ordeno_primero_z[30+20+gap:30+20+gap+20],markersize=marker_,zorder=2,color='orange',marker="o",label='pulse-duration')

       ax.plot(ordeno_primero_x[30+20+gap+20:],ordeno_primero_y[30+20+gap+20:],ordeno_primero_z[30+20+gap+20:],markersize=marker_,zorder=2,color='g',marker="o",label='last-part')
       ax.view_init(elev=10, azim=kk)
       ax.legend()
       figname = str(plot_dir)+"/sample_"+str(sample_number)+"_pca_3D_"+str(capa)+"_individual_neurons_state_"+str(i)+'_'+str(kk)+".png" 
       plt.savefig(figname,dpi=200) 
       plt.close()

             
     
    ####################################
    fig     =plt.figure(figsize=(14,12))
    fig.suptitle("Neural Network status Sample "+str(sample_number)+"\n (amplitude in arb. units time [mSec])",fontsize = 20) 

    plt.subplot(3, 1, 1) 
    #plt.plot(test[0,:,1],color='g',label='input A')
    plt.plot(test[0,:,0],color='b',label='input A')
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output')  
    plt.plot(y_pred[0,:, 0], color='r',label=' output')
    plt.xlim(0,seq_dur)
    #plt.ylim([-0.1, 1.5])
    plt.ylim([-2.5, 2.5])
    #plt.ylim([-0.1, 2.5])
    plt.ylabel('Activity',fontsize = 20)
    plt.xlabel('Time',fontsize = 20)
    plt.xticks(np.arange(0,seq_dur,20),fontsize = 8)
    plt.legend(fontsize= 'x-small')

    plt.subplot(3, 1, 2) 
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output')  
    for ii in np.arange(0,int(neurons/2),1):
        plt.plot(layer_output_T[ii],color=colors[ii],linewidth=1,label='Neuron'+ str(ii))         
        plt.xlim(0,seq_dur)
        #plt.ylim([-0.1, 1.5])
        plt.ylim([-2.5, 2.5])
        #plt.ylim([-0.1, 2.5])
        plt.ylabel('Activity',fontsize = 20)
        plt.xlabel('Time',fontsize = 20)
        plt.xticks(np.arange(0,seq_dur,20),fontsize = 8)
        plt.legend(fontsize= 'xx-small')


    plt.subplot(3, 1, 3)      
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output')  
    for ii in np.arange(int(neurons/2),neurons,1):
        plt.plot(layer_output_T[ii],color=colors[ii],linewidth=1,label='Neuron '+ str(ii))         
        plt.xlim(0,seq_dur)
        #plt.ylim([-0.1, 1.5])
        plt.ylim([-2.5, 2.5])
        #plt.ylim([-0.1, 2.5])
        plt.ylabel('Activity',fontsize = 20)
        plt.xlabel('Time',fontsize = 20)
        plt.xticks(np.arange(0,seq_dur,20),fontsize = 8)
        plt.legend(fontsize= 'xx-small')


    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_layer_"+str(capa)+"_individual_neurons_state_"+str(i)+".png" 
    plt.savefig(figname,dpi=200)
    #plt.show()

    fig     =plt.figure(figsize=(14,12))
    fig.suptitle("Neural Network PCA var evolution Sample "+str(sample_number)+"\n (amplitude in arb. units time [mSec])",fontsize = 20) 

    plt.subplot(3, 1, 1) 
    #plt.plot(test[0,:,1],color='g',label='input A')
    plt.plot(test[0,:,0],color='b',label='input A')
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output')
    plt.plot(y_pred[0,:, 0], color='r',label=' output')
    plt.plot(X_pca[0],color='g',marker="p",zorder=2,label="v0")  
    plt.plot(y_pred[0,:, 0], color='r',label=' output')
    plt.xlim(0,seq_dur)
    #plt.ylim([-0.1, 1.5])
    plt.ylim([-2.5, 2.5])
    #plt.ylim([-0.1, 2.5])
    plt.ylabel('Activity',fontsize = 20)
    plt.xlabel('Time',fontsize = 20)
    plt.xticks(np.arange(0,seq_dur,20),fontsize = 8)
    plt.legend(fontsize= 'x-small')

    plt.subplot(3, 1, 2)
    plt.plot(test[0,:,0],color='b',label='input A') 
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output')
    plt.plot(y_pred[0,:, 0], color='r',label=' output')  
    plt.plot(X_pca[1],color='g',marker="p",zorder=2,label="v1")       
    plt.xlim(0,seq_dur)
    #plt.ylim([-0.1, 1.5])
    plt.ylim([-2.5, 2.5])
    #plt.ylim([-0.1, 2.5])
    plt.ylabel('Activity',fontsize = 20)
    plt.xlabel('Time',fontsize = 20)
    plt.xticks(np.arange(0,seq_dur,20),fontsize = 8)
    plt.legend(fontsize= 'xx-small')


    plt.subplot(3, 1, 3)
    plt.plot(test[0,:,0],color='b',label='input A')      
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output')  
    plt.plot(y_pred[0,:, 0], color='r',label=' output')
    plt.plot(X_pca[2],color='g',marker="p",zorder=2,label="v2")     
    plt.xlim(0,seq_dur)
    #plt.ylim([-0.1, 1.5])
    plt.ylim([-2.5, 2.5])
    #plt.ylim([-0.1, 2.5])
    plt.ylabel('Activity',fontsize = 20)
    plt.xlabel('Time',fontsize = 20)
    plt.xticks(np.arange(0,seq_dur,20),fontsize = 8)
    plt.legend(fontsize= 'xx-small')


    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_layer_"+str(capa)+"_pca_time_ev_"+str(i)+".png" 
    plt.savefig(figname,dpi=200)
    #plt.show()
    plt.close()
    

