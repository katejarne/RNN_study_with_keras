import os
import time

import numpy as np
import matplotlib.pyplot as plt
from pylab import grid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import grid
from scipy.stats import norm
import matplotlib.mlab as mlab
from numpy import linalg as LA
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

import tensorflow as tf

# taking dataset from function
from generate_data_set_time_pulse import *
# To print network status
from print_status_1_inputs_paper import *


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

#Parameters:
sample_size_3       = 3
mem_gap             = 20
sample_size         = 11 # Data set to print some results
lista_distancia_all =[]

# Generate a data Set to study the Network properties:

x_train, y_train,mask,seq_dur  = generate_trials(sample_size,mem_gap) 
test                           = x_train[0:1,:,:] # Here you select from the generated data set which is used for test status
test_set                       = x_train[0:20,:,:]
y_test_set                     = y_train[0:20,:,0]


plot_dir="plots_paper"

r_dir="Networks_and_files/networks/.."



lista_neg=[]
lista_pos=[]
total    =[]
lista_neg_porc=[]
lista_pos_porc=[]
lista_tot_porc=[]


for root, sub, files in os.walk(r_dir):
    files = sorted(files)
    for i,f in enumerate(files):
       print("file: ",f)

       #General network model construction:
       model = Sequential()
       model = load_model(r_dir+"/"+f)

       # Compiling model for each file:

    
       model.compile(loss = 'mse', optimizer='Adam', sample_weight_mode="temporal")
       model.load_weights(r_dir+"/"+f)
       print"-------------",i

       #Esto me permite imprimir los pesos de la red y la estructura!!!

       pesos    = model.layers[0].get_weights()
       pesos__  = model.layers[0].get_weights()[0]
       pesos_in = pesos[0]
       pesos    = model.layers[0].get_weights()[1] 
       
       N_rec    =len(pesos_in[0])  # it has to match the value of the recorded trained network
       neurons  = N_rec
       colors   = cm.rainbow(np.linspace(0, 1, neurons+1))

       print"-------------\n-------------"   
       print"pesos:\n:",pesos
       print"-------------\n-------------"
 
       unidades        = np.arange(len(pesos))
       conection       = pesos

       
       print("array: ",np.arange(len(pesos)))       
       #print("biases: ",biases)
       print"##########################"       
       print"conection",conection      
       print"##########################\n ##########################"
     
       histo_lista    =[]
       array_red_list =[]
  
       conection_usar =conection
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
       plt.savefig(plot_dir+"/autoval_"+str(i)+"_"+str(f)+"_.png",dpi=300, bbox_inches = 'tight')
       plt.close()
       # Plots 

       ################################### Here we plot iner state of the network with the desierd stimulus
       
       for sample_number in np.arange(sample_size_3):
            print ("sample_number",sample_number)
            print_sample = plot_sample(sample_number,1,neurons,x_train,y_train,model,seq_dur,i,plot_dir)


