import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import grid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import grid
from scipy.stats import norm
from numpy import linalg as LA
import matplotlib.mlab as mlab
from keras.utils import CustomObjectScope

from keras.models import Sequential,load_model
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.layers.recurrent import SimpleRNN
from keras.layers import TimeDistributed, Dense, Activation, Dropout
from keras.utils import plot_model
from keras import metrics
from keras import optimizers
from keras import regularizers
from keras import initializers
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

# Para coustomizar el constraint!!!!
from keras.constraints import Constraint

import tensorflow as tf

# taking dataset from function:

from generate_data_set_and import *
#from generate_data_set_or import *
#from generate_data_set_xor import *
#from generate_data_set_not import *
#from generate_data_set_time_pulse import *

#from generate_data_set_oscilator_ver_2 import *
#from generate_data_set_oscilator import *
#from generate_data_set import *


# To print network status
from print_status_2_inputs_paper import *

#Parameters:
sample_size_3       = 6
mem_gap             = 20
sample_size         = 64 # Data set to print some results
lista_distancia_all =[]

# Generate a data Set to study the Network properties:

x_train,y_train, mask,seq_dur  = generate_trials(sample_size,mem_gap) 
test                           = x_train[0:1,:,:] # Here you select from the generated data set which is used for test status
test_set                       = x_train[0:20,:,:]
y_test_set                     = y_train[0:20,:,0]
con_matrix_list_pos            = []
con_matrix_list_neg            = []
con_matrix_list                = []


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

#para el paper

#### AND:
#r_dir="Networks_and_files/networks/AND/.."

##### OR:
#r_dir="Networks_and_files/networks/OR/.."

##### XOR:
#r_dir="Networks_and_files/networks/Xor/.."

#### NOT:
#r_dir="Networks_and_files/networks/Not/.."

### Pulse:

#r_dir="Networks_and_files/networks/time/.."

### Osc:
#r_dir="Networks_and_files/networks/Osc/.."

#### Flip Flop:

#r_dir="Networks_and_files/networks/FF/.."

plot_dir="plots_paper"

lista_neg     =[]
lista_pos     =[]
total         =[]
lista_neg_porc=[]
lista_pos_porc=[]
lista_tot_porc=[]

#N_rec=50
g=1


for root, sub, files in os.walk(r_dir):
    files = sorted(files)
    for i,f in enumerate(files):
       print("file: ",f)
 
       #General network model construction:
       
       model = Sequential()
       
       #model = load_model(r_dir+"/"+f)      
       seed(1)

       #model.reset_states()
       model = load_model(r_dir+"/"+f)   #custom_objects={'NonNegLast':NonNegLast})
       # Compiling model for each file:
       model.compile(loss = 'mse', optimizer='Adam', sample_weight_mode="temporal")
  
       print"-------------",i
       #Esto me permite imprimir los pesos de la red y la estructura!!!
       for i, layer in enumerate(model.layers):
           print"i-esima capa: ",i 
           print(layer.get_config(), layer.get_weights())
       
       pesos     = model.layers[0].get_weights()
       pesos__   = model.layers[0].get_weights()[0]
       pesos_in  = pesos[0]
       pesos_out = model.layers[1].get_weights()
       pesos     = model.layers[0].get_weights()[1] 
      # biases   = model.layers[0].get_weights()[2]
       '''
       
       pesos    = model.layers[1].get_weights()
       pesos__  = model.layers[1].get_weights()[0]
       pesos_in = pesos[1]
       pesos    = model.layers[1].get_weights()[1] 
       #biases   = model.layers[1].get_weights()[2]
       #pepe_4 = model.layers[0].get_weights()[3] 
       '''

       N_rec                          =len(pesos_in[0])  # it has to match the value of the recorded trained network
       neurons                        = N_rec
       colors                         = cm.rainbow(np.linspace(0, 1, neurons+1))


       print "h",model.layers[0].states[0]

       print"-------------\n-------------"   
       print"pesos:\n:",pesos
       print"-------------\n-------------"
       print"N_REC:",N_rec
       unidades        = np.arange(len(pesos))
       conection       = pesos
  
       con_matrix_list.append(pesos)
       print("array: ",np.arange(len(pesos)))       
       #print("biases: ",biases)
       
       print"##########################\n ##########################"
       print"conection",conection       
       print"##########################\n ##########################"
     
       histo_lista    =[]
       array_red_list =[]

       peso_mask  = 0.001 # 0.1# 0.011
       peso_mask_2=-0.001#-0.1#-0.011
      
       # Test anulando las conecciones que son mas debiles que el x% del max o el minimo
       
       conection_usar =conection
       #conection_usar[(conection_usar < peso_mask) & (conection_usar > peso_mask_2)] = 0
       #np.fill_diagonal(conection_usar, 0.0)
       #model.layers[0].set_weights([pesos_in,conection_usar])
       
       w, v = LA.eig(conection_usar)
       print"Autovalores:\n", w
       print"Autovectores:\n",v
       print"Distancia:", np.sqrt(w.real*w.real+w.imag*w.imag)

       lista_dist  = np.c_[w,w.real]
       lista_dist_2= np.c_[w,abs(w.real)]
       maximo      = max(lista_dist, key=lambda item: item[1])
       
       maximo_2= max(lista_dist_2, key=lambda item: item[1])
       marcar  = maximo[0]
       marcar_2= maximo_2[0]
       
       print"Primer elemento",maximo
       print"Maximo marcar",marcar
       
       frecuency=0
       if marcar_2.imag==0:
           frecuency =0
       else: 
           frecuency =abs(float(marcar_2.imag)/(3.14159*float(marcar_2.real)))
           
       print "frecuency",frecuency
 
       ################ Fig Eigenvalues ########################

       plt.figure(figsize=cm2inch(8.5,7.5))
       plt.scatter(w.real,w.imag,color="hotpink",label="Eigenvalue spectrum\n ",s=2)#Total of: "+str(len(w.real))+" values")
       #plt.scatter(w.real,w.imag,color="plum",label="Eigenvalue spectrum\n Total of: "+str(len(w.real))+" values")
       # for plotting circle line:
       a = np.linspace(0, 2*np.pi, 500)
       cx,cy = np.cos(a), np.sin(a)
       plt.plot(cx, cy,'--', alpha=.5, color="dimgrey") # draw unit circle line
       #plt.plot(2*cx, 2*cy,'--', alpha=.5)
       plt.scatter(marcar.real,marcar.imag,color="red", label="Eigenvalue maximum real part",s=5)
       #plt.scatter(marcar_2.real,marcar_2.imag,color="blue", label="Eigenvalue with maximum abs(Real part)\n"+"Frecuency: "+str(frecuency))
       plt.plot([0,marcar.real],[0,marcar.imag],'-',color="grey")
       #plt.plot([0,marcar_2.real],[0,marcar_2.imag],'k-')
       plt.axvline(x=1,color="salmon",linestyle='--')
       plt.xticks(fontsize=4)
       plt.yticks(fontsize=4)
       plt.xlabel(r'$Re( \lambda)$',fontsize = 11)
       plt.ylabel(r'$Im( \lambda)$',fontsize = 11)
       plt.legend(fontsize= 5,loc=1)
       plt.savefig(plot_dir+"/autoval_"+str(i)+"_"+str(f)+"_"+str(peso_mask)+"_.png",dpi=300, bbox_inches = 'tight')
       plt.close()
       
       
       ########## Here we plot iner state of the network with the desierd stimuli: 
       
       for sample_number in np.arange(sample_size_3):
          print ("sample_number",sample_number)
          print_sample = plot_sample(sample_number,2,neurons,x_train,y_train,model,seq_dur,i,plot_dir,f)

       ##########    
                 
       for ii in unidades:
           histo_lista.extend(pesos[ii])
           
       media= np.average(histo_lista)
       

       # best fit of data
       (mu, sigma) = norm.fit(histo_lista)
       # the histogram of the data
       n, bins, patches = plt.hist(histo_lista, 200, normed=1, facecolor='green', alpha=0.75)

       # add a 'best fit' line
       y = mlab.normpdf( bins, mu, sigma)
       
       plt.figure(figsize=(14,10)) 
       #if i==0:
       #    plt.title('Initial Histogram Weights', fontsize = 18)
       #else:
       plt.title('Histogram Weights after \"AND\" learning', fontsize = 18)

       plt.hist(histo_lista, bins=200,color="mistyrose",normed=1,label="Weight Value \n Mu= "+str(mu)+"\n Sigma= "+str(sigma))
       plt.plot(bins, y, 'r-', linewidth=3)
       plt.axvline(mu, color='r', linestyle='dashed', linewidth=2)
       plt.vlines(x=sigma, ymin=0, ymax=900, linewidth=2, color='k')
       plt.vlines(x=-sigma,ymin=0, ymax=900, linewidth=2, color='k')
       plt.xlabel('Weight strength [arb. units]',fontsize = 16)
       plt.ylim([0,6])
       plt.xlim([-0.5,0.5])
       plt.legend(fontsize= 5,loc=1)
       plt.savefig(plot_dir+"/weight_histo_"+str(i)+"_"+str(f)+'_'+str(peso_mask)+"_0.png",dpi=300, bbox_inches = 'tight')
       plt.close()
            
       ######   Conectivity matrix: positive weights #################
       #0.1
       plt.figure(figsize=(14,12)) 
       plt.title('Conection matrix iteration:'+str(i), fontsize = 18)#, weight="bold")
       grid(True)

       cmap           = plt.cm.gist_ncar # Tipos de mapeo #OrRd # 'gist_ncar'#"plasma"
       #conection_pos  = np.ma.masked_where(conection < peso_mask, conection)
       conection_pos = np.ma.masked_where(conection_usar < peso_mask, conection)
       #conection_pos  = np.ma.masked_where(conection > peso_mask, conection)
       #conection_pos = np.ma.masked_where(conection < 0.0, conection)
       
       cmap.set_bad(color='white')
       plt.imshow(conection_pos,cmap='gist_ncar',interpolation="none",label='Conection matrix')
            
       cbar_max  = 0.75
       cbar_min  = -0.75
       cbar_step = 0.025
       
       cbar=plt.colorbar(ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step))
       cbar.ax.set_ylabel('Weights [arbitrary unit]', fontsize = 16, weight="bold")
       
       plt.xlim([-1,N_rec +1])
       plt.ylim([-1,N_rec +1])
       
       plt.xticks(np.arange(0, 51, 2))
       plt.yticks(np.arange(0, 51, 2))
       plt.ylabel('Unit [i]',fontsize = 16)
       plt.xlabel('Unit [j]',fontsize = 16)
       #plt.legend(fontsize= 'medium',loc=1)
       #plt.savefig(plot_dir+"/Pos_conection_matrix_"+str(i)+"_"+str(f)+'_'+str(peso_mask)+"_0.png",dpi=200)
       plt.close()
      
       ######   Conectivity matrix: Negative  weights ######
       plt.figure(figsize=(14,12)) 
       plt.title('Conection matrix iteration:'+str(i), fontsize = 18)#, weight="bold")
       grid(True)
       cmap = plt.cm.gist_ncar # Tipos de mapeo OrRd # 'gist_ncar'#"plasma"
       #conection_neg = np.ma.masked_where(conection > -peso_mask, conection)
       conection_neg = np.ma.masked_where(conection_usar > peso_mask_2, conection)
       #conection_neg = np.ma.masked_where(conection > -peso_mask, conection)
       #conection_neg = np.ma.masked_where(conection < -peso_mask, conection)
       #conection_neg = np.ma.masked_where(conection > 0.0, conection)
       pepe_mask     = conection_neg.compressed().shape
       pepe_mask_2   = conection_pos.compressed().shape
       pepe_sin_mask = conection.shape
       
       cmap.set_bad(color='white')

       plt.imshow(conection_neg,cmap='gist_ncar',interpolation="none",label='Conection matrix')
            
       cbar_max  = 0.75
       cbar_min  = -0.75
       cbar_step = 0.025
       
       cbar_2=plt.colorbar(ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step))       
       cbar_2.ax.set_ylabel('Weights [arbitrary unit]', fontsize = 16, weight="bold")
 
       plt.xlim([-1,N_rec +1])
       plt.ylim([-1,N_rec +1])
       plt.xticks(np.arange(0, 51, 2))
       plt.yticks(np.arange(0, 51, 2))
       plt.ylabel('Unit [i]',fontsize = 16)
       plt.xlabel('Unit [j]',fontsize = 16)
       #plt.legend(fontsize= 'medium',loc=1)
       #plt.savefig(plot_dir+"/Neg_conection_matrix_"+str(i)+"_"+str(f)+'_'+str(peso_mask)+"0.png",dpi=300, bbox_inches = 'tight')       
       plt.close()

       zzz=np.zeros(50)

       conection_filt =conection_neg+conection_pos
       
       # Sparsity
       print("total elements: ",pepe_sin_mask)
       print(" negative Elements total: ",pepe_mask[0])
       print(" positive Elements total: ",pepe_mask_2[0])
       lista_neg.append(pepe_mask[0])
       lista_pos.append(pepe_mask_2[0])
       total.append(N_rec*N_rec)
       lista_neg_porc.append(100*pepe_mask[0]/(N_rec*N_rec))
       lista_pos_porc.append(100*pepe_mask_2[0]/(N_rec*N_rec))
       lista_tot_porc.append(100*(pepe_mask[0]+pepe_mask_2[0])/(N_rec*N_rec))


       #################################### Conectivity matrix: positive or excitatory weights ###################################
       #0.1
       plt.figure(figsize=(14,12)) 
       plt.title('Conection matrix iteration:'+str(i), fontsize = 18)#, weight="bold")
       grid(True)

       cmap           = plt.cm.gist_ncar # Tipos de mapeo #OrRd # 'gist_ncar'#"plasma"
       
       conection[conection < peso_mask ] = 0
       conection_filt= conection
       print("conection_filt",conection_filt)
       conection_filt = np.ma.masked_where(conection_filt ==0, conection_filt)
       plt.imshow(conection_filt,cmap='gist_ncar',interpolation="none",label='Conection matrix')
            
       cbar_max  = 0.75
       cbar_min  = -0.75
       cbar_step = 0.025
       
       cbar=plt.colorbar(ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step))
       cbar.ax.set_ylabel('Weights [arbitrary unit]', fontsize = 16, weight="bold")
       
       plt.xlim([-1,N_rec +1])
       plt.xticks(np.arange(0, 51, 2))
       plt.yticks(np.arange(0, 51, 2))
       plt.ylim([-1,N_rec +1])
       
       plt.ylabel('Unit [i]',fontsize = 16)
       plt.xlabel('Unit [j]',fontsize = 16)
       #plt.legend(fontsize= 'medium',loc=1)
       #plt.savefig(plot_dir+"/filt_conection_matrix_"+str(i)+"_"+str(f)+"_0.png",dpi=300, bbox_inches = 'tight')
       plt.close()
       
       # Model Testing: 
       x_pred = x_train[0:10,:,:]
       y_pred = model.predict(x_pred)

       print("x_train shape:\n",x_train.shape)
       print("x_pred shape\n",x_pred.shape)
       print( "y_train shape\n",y_train.shape)

       lista_distancia=[]
       
       #########################################
       fig= plt.figure(figsize=cm2inch(8,7))
       #fig.suptitle("\"And\" Data Set Trainined Output \n (amplitude in arb. units time in mSec)",fontsize = 20)
       for ii in np.arange(6):
           plt.subplot(3, 2, ii + 1)    
           plt.plot(x_train[ii, :, 0],color='g',label="Input A")
           plt.plot(x_train[ii, :, 1],color='pink',label="Input B")
           plt.plot(y_train[ii, :, 0],color='gray',linewidth=2,label="Expected Output")
           plt.plot(y_pred[ii, :, 0], color='r',label="Predicted Output")
           plt.ylim([-2, 2])
           #plt.xlim([0, 205])
           plt.xlim([0, 205])
           plt.xticks(np.arange(0,205,100),fontsize = 8)
           plt.legend(fontsize= 2.75,loc=3)
           #plt.xticks([])
           plt.yticks([])
           plt.xticks(fontsize=5)
           plt.yticks(fontsize=5)
           a=y_train[ii, :, 0]
           b=y_pred[ii, :, 0]
 
           a_min_b = np.linalg.norm(a-b)      
           lista_distancia.append(a_min_b)
       fig.text(0.5, 0.03, 'time [mS]',fontsize=5, ha='center')
       fig.text(0.1, 0.5, 'Amplitude [Arb. Units]', va='center', ha='center', rotation='vertical', fontsize=5)
       figname =plot_dir+"/data_set_"+str(peso_mask)+'_'+str(f)+".png"       
       plt.savefig(figname,dpi=300, bbox_inches = 'tight')
       
       lista_distancia.insert(0,N_rec)
       lista_distancia_all.append(lista_distancia)
       K.clear_session()
todo         = np.c_[lista_neg,lista_pos,total,lista_neg_porc,lista_pos_porc,lista_tot_porc]
todo_2       = lista_distancia_all
print (todo_2)



