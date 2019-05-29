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
from keras.models import Sequential, Model

# pca part:
from sklearn.decomposition import PCA
import sklearn.decomposition
from scipy import signal


def plot_sample(sample_number,input_number,neurons,x_train,y_train,model,seq_dur,i,plot_dir,f):
    #neurons=50
    import time
    frecuencias=[]
    #print"model.layers[0].states[0]",model.layers[0].states
    #time.sleep(15.5)
    
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
    
    #layer_output= model.layers[0].output

    
    #print"layer_output",layer_output
    #time.sleep(15.5)
    ''''
    me_fijo=layer_output[0]
    print "me_fijo[0]", me_fijo[0]
    me_fijo_2=me_fijo[0]
    layer_output=me_fijo_2[0]
    '''
    #get_0_layer_output =  [K.get_value(s) for s,_ in model.state_updates]    
    #layer_output.append(layer_output_m)
    
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
    #print("len",len(layer_output[capa]))
    y_pred              = model.predict(test)

    # To generate the Populational Analysis

    for ii in np.arange(0,neurons,1):
        neurona_serie = np.reshape(layer_output_T[ii], len(layer_output_T[ii]))
        #print"neurona_serie: ",ii," aca: ",neurona_serie
        #time.sleep(15.5)
        array_red_list.append(neurona_serie)
    
    #print"array_red_list",array_red_list
    array_red = np.asarray(array_red_list)
    sdv       = sklearn.decomposition.TruncatedSVD(n_components=2)
    sdv_3d    = sklearn.decomposition.TruncatedSVD(n_components=3)
    X_2d      = sdv.fit_transform(array_red.T)
    X_3d      = sdv_3d.fit_transform(array_red.T)
    #print('size X_2d',X_2d.size)
    #print('size X_3d',X_3d.size)

    pca     = PCA(n_components=3)
    X_pca_  = pca.fit(array_red)
    X_pca   = pca.components_

    #print("X_pca: ",X_pca)

    ####################################

    fig = plt.figure()
    fig.suptitle("SDV Network Population Analysis",fontsize = 20)
    ax1 = fig.add_subplot(111)
    
    #ax1.scatter(X_vs_2d[:,0],X_vs_2d[:,1],c='m',label='sdv pesos')
    #ax1.scatter(X_2d[:,0],X_2d[:,1],c='g', lw=3,label='Dim Reduction of the network')
    ax1.plot(X_2d[:,0],X_2d[:,1],c='g',marker="p",zorder=2,label='Dim Reduction of the network')
    ax1.scatter(X_2d[0,0],X_2d[0,1],c='r',marker='^',s=70,label='start')
    ax1.scatter(X_2d[-1,0],X_2d[-1,1],c='b',marker='^',s=70,label='stop')
    plt.legend(loc='upper left',fontsize= 'x-small');
    plt.ylabel('C1')
    plt.xlabel('C2')
    plt.ylim([-3,3])
    plt.xlim([-4, 4])
    
    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_sdv_"+str(capa)+"_individual_neurons_state_"+str(i)+".png" 
    #plt.savefig(figname,dpi=200)
    plt.close()
    #pp.show()

    print"------------"
    ordeno_primero_x=X_pca[0]
    ordeno_primero_y=X_pca[1]
    ordeno_primero_z=X_pca[2]

    fig = plt.figure()
    fig.suptitle("PCA Network Population Analysis",fontsize = 20)
    ax1 = fig.add_subplot(111)
    
    ax1.plot(X_pca[0],X_pca[1],c='c',marker="p",zorder=2,label='Dim Reduction of the network')    
    ax1.scatter(ordeno_primero_x[0],ordeno_primero_y[0],s=70,c='r',marker="^",label='start')
    ax1.scatter(ordeno_primero_x[-1],ordeno_primero_y[-1],s=70,c='b',marker="^",label='stop')
    plt.legend(loc='upper left',fontsize= 'x-small');
    plt.ylabel('C1')
    plt.xlabel('C2')
    plt.ylim([-0.3,0.3])
    plt.xlim([-0.15, 0.15])
    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_pca_"+str(capa)+"_individual_neurons_state_"+str(i)+".png" 
    #plt.savefig(figname,dpi=200)
    plt.close()
    #pp.show()
              
    ####################################
    # How many 3d angular views you want to define

    #yy        = np.arange(0,360,10)
    yy        = np.arange(70,80,10)
    '''
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
        figname = "plots_and/sample_"+str(sample_number)+"_sdv_3d_"+str(capa)+"_individual_neurons_state_"+str(i)+'_'+str(kk)+".png" 
        plt.savefig(figname,dpi=200)
        plt.close()
    '''
    ####################################   
    
    kk=70        
    #for ii,kk in enumerate(yy):
   #print "ii: ",ii," kk: ",kk   

    fig     = plt.figure(figsize=(18,7))
    plt.subplot(2, 2, 1) 
    plt.plot(test[0,:,1],color='g',label='input A')
    plt.plot(test[0,:,0],color='b',label='input B')
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output')  
    plt.plot(y_pred[0,:, 0], color='r',linewidth=2,label=' output')
    plt.xlim(0,seq_dur+1)
    plt.ylim([-1.5, 1.5])
    plt.ylabel('Activity [arb. units]',fontsize = 16)
    plt.xlabel('time [mSec]',fontsize = 16)
    plt.xticks(np.arange(0,seq_dur+1,20),fontsize = 8)
    plt.legend(fontsize= 'x-small',loc=3)

    import time
    
    plt.subplot(2, 2, 3) 
    plt.plot(test[0,:,1],color='g',label='input A')
    plt.plot(test[0,:,0],color='b',label='input B')
    
    #plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output\n Individual states')  
    for ii in np.arange(0,int(neurons/2),1):
        a_ver=layer_output_T[ii]

        t_= np.arange(len(a_ver))
        freq=0
        print "print t",t_
    
        peakind_min,peakind_min_2      = signal.argrelmin(a_ver,axis=0)
        amp_pico_min_x   = a_ver[peakind_min]

        print "peakind_min: ",peakind_min
        print "amp_pico_min_x: ",amp_pico_min_x

        if len(amp_pico_min_x)>1:
            pepe         =t_[peakind_min]
            print "pepe",pepe
            if len(pepe)>6:
                freq         =1/(0.001*float(pepe[-2]-pepe[-3]))
        else:
            freq =0   

        ff='%.4f'%freq
        print("t",t_)
        print("Frequency",ff)
        frecuencias.append(ff)
        
        #print"aver",a_ver[0]
        #time.sleep(15.5)  
        #plt.plot(layer_output_T[ii],color=colors[ii],linewidth=1)#,label='Neuron'+ str(ii))
        imprimir=layer_output_T[ii].T[0]
        plt.plot(layer_output_T[ii],color=colors[ii],linewidth=1)#, label="frec: "+str(ff))
        #plt.scatter(peakind_min,amp_pico_min_x,color="k")
        #plt.scatter(np.arange(len(layer_output_T[ii])),imprimir,color=colors[ii],linewidth=1, label="frec: "+str(ff))
        plt.xlim(-1,seq_dur+1)         
        #plt.xlim(0,seq_dur)
        plt.ylim([-1.5, 1.5])
        plt.ylabel('Activity [arb. units]',fontsize = 16)
        plt.xlabel('time [mSec]',fontsize = 16)
        plt.xticks(np.arange(0,seq_dur+1,20),fontsize = 8)
    plt.plot(y_pred[0,:, 0], color='r',linewidth=2,label=' output\n 25 individual states')    
    plt.legend(fontsize= 'x-small',loc=3)

    
    #plt.subplot(2, 2, 4) 
    fig.suptitle("Time series and PCA 3D plot",fontsize = 20)
    ax = fig.add_subplot(122, projection='3d')
    ax.plot(X_pca[0],X_pca[1],X_pca[2],color='g',marker="p",zorder=2,markersize=5,label="3 d plot")
    ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=250,c='r',marker="^",label='start')     
    ax.scatter(ordeno_primero_x[-1],ordeno_primero_y[-1],ordeno_primero_z[-1],s=250,c='b',marker="^",label='stop')

       
    #### part: 
    first_in         = 30 #time to start the first stimulus   #30 #60
    stim_dur         = 20 #stimulus duration #20 #30
    out_gap          = 130 #how much lenth add to the sequence duration    #140 #100
    marker_          = 5
    gap              = 20
    #ax.plot(ordeno_primero_x[0:30],ordeno_primero_y[0:30],ordeno_primero_z[0:30],markersize=marker_,color='y',zorder=2,marker="o",label='start')
    #ax.plot(ordeno_primero_x[30:30+20],ordeno_primero_y[30:30+20],ordeno_primero_z[30:30+20],markersize=marker_,zorder=2,color='m',marker="o",label='stimulus')
    #ax.plot(ordeno_primero_x[30+20:30+20+gap],ordeno_primero_y[30+20:30+20+gap],ordeno_primero_z[30+20:30+20+gap],markersize=marker_,zorder=2,color='r',marker="o",label='reaction gap')
    #ax.plot(ordeno_primero_x[30+20+gap:],ordeno_primero_y[30+20+gap:],ordeno_primero_z[30+20+gap:],markersize=marker_,zorder=2,color='g',marker="o",label='final state')

    #ax.scatter(X_3d[-1,0],X_3d[-1,1],X_3d[-1,2],c='b',marker="^",label='stop')
    #ax.scatter(X_3d[:,0],X_3d[:,1],X_3d[:,2],color='g', zorder=2,marker="p",label="3 d plot")
    ax.set_xlabel('pca 1 (arb. units)',size=16)
    ax.set_ylabel('pca 2 (arb. units)',size=16)
    ax.set_zlabel('pca 3 (arb. units)',size=16)
    #ax.set_zlim(-0.42,0.35)
    ax.set_ylim(-0.12,0.12)
    #ax.set_xlim(-0.25,0.2)
    ax.set_zlim(-0.35,0.2)
    ax.set_xlim(-0.12,0.12)
    ax.view_init(elev=10, azim=kk)
    ax.legend(fontsize= 'small')
    
    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_pca_3D_"+str(capa)+"_individual_neurons_state_"+str(i)+'_'+str(kk)+"_"+str(f)+".png" 
    plt.savefig(figname,dpi=200) 
    plt.close()      
    
    ####################################
    '''
      for ii,kk in enumerate(yy):
       #print "ii: ",ii," kk: ",kk   
       fig     = plt.figure(figsize=(10,8))
       fig.suptitle("3D plot PCA",fontsize = 20)
       ax = fig.add_subplot(111, projection='3d')
       ax.plot(X_pca[0],X_pca[1],X_pca[2],color='c',marker="p",zorder=2,markersize=5,label="3 d plot")
       ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=250,c='r',marker="^",label='start')     
       ax.scatter(ordeno_primero_x[-1],ordeno_primero_y[-1],ordeno_primero_z[-1],s=250,c='b',marker="^",label='stop')

       
       #### part: 
       first_in         = 30 #time to start the first stimulus   #30 #60
       stim_dur         = 20 #stimulus duration #20 #30
       out_gap          = 130 #how much lenth add to the sequence duration    #140 #100
       marker_          = 5
       gap              = 20
       ax.plot(ordeno_primero_x[0:30],ordeno_primero_y[0:30],ordeno_primero_z[0:30],markersize=marker_,color='y',zorder=2,marker="o",label='start')
       ax.plot(ordeno_primero_x[30:30+20],ordeno_primero_y[30:30+20],ordeno_primero_z[30:30+20],markersize=marker_,zorder=2,color='m',marker="o",label='stimulus')
       ax.plot(ordeno_primero_x[30+20:30+20+gap],ordeno_primero_y[30+20:30+20+gap],ordeno_primero_z[30+20:30+20+gap],markersize=marker_,zorder=2,color='r',marker="o",label='reaction gap')
       ax.plot(ordeno_primero_x[30+20+gap:],ordeno_primero_y[30+20+gap:],ordeno_primero_z[30+20+gap:],markersize=marker_,zorder=2,color='g',marker="o",label='final state')

       #ax.scatter(X_3d[-1,0],X_3d[-1,1],X_3d[-1,2],c='b',marker="^",label='stop')
       #ax.scatter(X_3d[:,0],X_3d[:,1],X_3d[:,2],color='g', zorder=2,marker="p",label="3 d plot")
       ax.set_xlabel('pca 1 (arb. units)',size=16)
       ax.set_ylabel('pca 2 (arb. units)',size=16)
       ax.set_zlabel('pca 3 (arb. units)',size=16)
       #ax.set_zlim(-0.42,0.35)
       ax.set_ylim(-0.12,0.12)
       #ax.set_xlim(-0.25,0.2)
       ax.set_zlim(-0.35,0.2)
       ax.set_xlim(-0.12,0.12)
       ax.view_init(elev=10, azim=kk)
       ax.legend()
       figname = str(plot_dir)+"/sample_"+str(sample_number)+"_pca_3D_"+str(capa)+"_individual_neurons_state_"+str(i)+'_'+str(kk)+"_"+str(f)+".png" 
       plt.savefig(figname,dpi=200) 
       plt.close()     
    '''

    '''
    fig     =plt.figure(figsize=(14,12))
    fig.suptitle("Neural Network status Sample "+str(sample_number)+"\n (amplitude in arb. units time [mSec])",fontsize = 20) 

    plt.subplot(3, 1, 1) 
    plt.plot(test[0,:,1],color='g',label='input A')
    plt.plot(test[0,:,0],color='b',label='input B')
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output')  
    plt.plot(y_pred[0,:, 0], color='r',linewidth=2,label=' output')
    plt.xlim(0,seq_dur)
    plt.ylim([-1.5, 1.5])
    plt.ylabel('Activity [arb. units]',fontsize = 20)
    plt.xlabel('time [mSec]',fontsize = 20)
    plt.xticks(np.arange(0,seq_dur,20),fontsize = 8)
    plt.legend(fontsize= 'x-small')

    import time

    
    plt.subplot(3, 1, 2) 
    plt.plot(test[0,:,1],color='g',label='input A')
    plt.plot(test[0,:,0],color='b',label='input B')
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output\n Individual states')  
    for ii in np.arange(0,int(neurons/2),1):
        a_ver=layer_output_T[ii]

        t_= np.arange(len(a_ver))
        freq=0
        print "print t",t_
    
        peakind_min,peakind_min_2      = signal.argrelmin(a_ver,axis=0)
        amp_pico_min_x   = a_ver[peakind_min]

        print "peakind_min: ",peakind_min
        print "amp_pico_min_x: ",amp_pico_min_x

        if len(amp_pico_min_x)>1:
            pepe         =t_[peakind_min]
            print "pepe",pepe
            if len(pepe)>6:
                freq         =1/(0.001*float(pepe[-2]-pepe[-3]))
        else:
            freq =0   

        ff='%.4f'%freq
        print("t",t_)
        print("Frequency",ff)
        frecuencias.append(ff)
        
        #print"aver",a_ver[0]
        #time.sleep(15.5)  
        #plt.plot(layer_output_T[ii],color=colors[ii],linewidth=1)#,label='Neuron'+ str(ii))
        imprimir=layer_output_T[ii].T[0]
        plt.plot(layer_output_T[ii],color=colors[ii],linewidth=1, label="frec: "+str(ff))
        plt.scatter(peakind_min,amp_pico_min_x,color="k")
        #plt.scatter(np.arange(len(layer_output_T[ii])),imprimir,color=colors[ii],linewidth=1, label="frec: "+str(ff))
        plt.xlim(-1,seq_dur)         
        #plt.xlim(0,seq_dur)
        plt.ylim([-1.5, 1.5])
        plt.ylabel('Activity [arb. units]',fontsize = 20)
        plt.xlabel('time [mSec]',fontsize = 20)
        plt.xticks(np.arange(0,seq_dur,20),fontsize = 8)
        plt.legend(fontsize= 'xx-small')


    plt.subplot(3, 1, 3)      
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output \n Individual states')  
    for ii in np.arange(int(neurons/2),neurons,1):
        imprimir=layer_output_T[ii].T[0]
        #print"aca",(np.arange(len(layer_output_T[ii]))),"aca",imprimir
        #time.sleep(15.5)
        #plt.plot(layer_output_T[ii],color=colors[ii],linewidth=1)#,label='Neuron '+ str(ii))
        plt.plot(layer_output_T[ii],color=colors[ii],linewidth=1, label="frec: "+str(ff))
        #plt.scatter(np.arange(len(layer_output_T[ii])),imprimir,color=colors[ii],linewidth=1)
        plt.xlim(-1,seq_dur)
        plt.ylim([-1.5, 1.5])
        plt.ylabel('Activity [arb. units]',fontsize = 20)
        plt.xlabel('time [mSec]',fontsize = 20)
        plt.xticks(np.arange(0,seq_dur,20),fontsize = 8)
        plt.legend(fontsize= 'xx-small')


    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_layer_"+str(capa)+"_individual_neurons_state_"+str(i)+"_"+str(f)+".png" 
    plt.savefig(figname,dpi=200)
    #plt.show()
    plt.close()
    '''


    '''    
    fig     =plt.figure(figsize=(14,12))
    fig.suptitle("Neural Network PCA var evolution Sample "+str(sample_number)+"\n (amplitude in arb. units time [mSec])",fontsize = 20) 

    plt.subplot(3, 1, 1) 
    #plt.plot(test[0,:,1],color='g',label='input A')
    plt.plot(test[0,:,0],color='g',label='input A')
    plt.plot(test[0,:,1],color='b',label='input B')
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output')
    plt.plot(X_pca[0],color='g',marker="p",zorder=2,label="v0")  
    plt.plot(y_pred[0,:, 0], color='r',label=' output')
    plt.xlim(0,seq_dur)
    #plt.ylim([-0.1, 1.5])
    plt.ylim([-1.5, 1.5])
    #plt.ylim([-0.1, 2.5])
    plt.ylabel('Activity',fontsize = 20)
    plt.xlabel('Time',fontsize = 20)
    plt.xticks(np.arange(0,seq_dur,20),fontsize = 8)
    plt.legend(fontsize= 'x-small')

    plt.subplot(3, 1, 2)
    plt.plot(test[0,:,0],color='g',label='input A')
    plt.plot(test[0,:,1],color='b',label='input B') 
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output')
    plt.plot(y_pred[0,:, 0], color='r',label=' output')  
    plt.plot(X_pca[1],color='g',marker="p",zorder=2,label="v1")       
    plt.xlim(0,seq_dur)
    #plt.ylim([-0.1, 1.5])
    plt.ylim([-1.5, 1.5])
    #plt.ylim([-0.1, 2.5])
    plt.ylabel('Activity',fontsize = 20)
    plt.xlabel('Time',fontsize = 20)
    plt.xticks(np.arange(0,seq_dur,20),fontsize = 8)
    plt.legend(fontsize= 'xx-small')


    plt.subplot(3, 1, 3)
    plt.plot(test[0,:,0],color='g',label='input A')
    plt.plot(test[0,:,1],color='b',label='input B')      
    plt.plot(y_train[sample_number,:, 0],color='k',linewidth=3,label='Target Output')
    plt.plot(y_pred[0,:, 0], color='r',label=' output')  
    plt.plot(X_pca[2],color='g',marker="p",zorder=2,label="v2")     
    plt.xlim(0,seq_dur)
    #plt.ylim([-0.1, 1.5])
    plt.ylim([-1.5, 1.5])
    #plt.ylim([-0.1, 2.5])
    plt.ylabel('Activity',fontsize = 20)
    plt.xlabel('Time',fontsize = 20)
    plt.xticks(np.arange(0,seq_dur,20),fontsize = 8)
    plt.legend(fontsize= 'xx-small')


    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_layer_"+str(capa)+"_pca_time_ev_"+str(i)+"_"+str(f)+".png" 
    #plt.savefig(figname,dpi=200)
    #plt.show()
    plt.close()
    '''
    #print"frecuencias",frecuencias
