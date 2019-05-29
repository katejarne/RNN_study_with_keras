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

# taking dataset from function
from generate_data_set_and_rand_place import *
#from generate_data_set_and import *

# To print network status
from print_status_2_inputs import *

#Parameters:
sample_size_3       = 10
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


#Pruebas 05/04
#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-And-03-29-19_test/con_ruido/orto/weights_and_0_N_50_gap_20/aca"
#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-And-03-29-19_test/sin_ruido/orto-500/weights_and_0_N_500_gap_20/aca"
#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-And-03-29-19_test/sin_ruido/normal_dist/weights_and_19_N_50_gap_20/aca"
#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-And-03-29-19_test/sin_ruido/orto-500/weights_and_9_N_500_gap_20/aca"
#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-04-12-19/ran_normal/weights_and_32_N_50_gap_20/aca"
#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-04-12-19/ran_normal/weights_and_26_N_50_gap_20/aca"


#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-04-13-19/ran_nor/weights_and_39_N_50_gap_20/aca"
#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights_and_20_N_500_gap_20/aca"
#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-04-14-19/weights-200/weights_and_1_N_200_gap_20/aca"

#para el paper

#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-And-03-29-19_test/con_ruido/orto/weights_and_14_N_50_gap_20/aca"
r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-And-03-29-19_test/con_ruido/orto/weights_and_4_N_50_gap_20/aca"
##r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-And-03-29-19_test/sin_ruido/normal_dist/weights_and_13_N_50_gap_20/aca"
##r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-04-12-19/ran_normal_ruido/weights_and_32_N_50_gap_20/aca"
#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-And-03-29-19_test/sin_ruido/normal_dist/weights_and_7_N_50_gap_20/aca"
#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights-And-03-29-19_test/sin_ruido/normal_dist/weights_and_18_N_50_gap_20/aca"

#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code/weights_and/weights_and_20_N_50_gap_5/aca"

plot_dir="plots_and"
f_out         = open(plot_dir+'/%s.txt' %('sparcisity_gap_'+str(mem_gap)), 'w')
f_out_dist    = open(plot_dir+'/%s.txt' %('distance_gap_' + str(mem_gap)), 'w')

lista_neg     =[]
lista_pos     =[]
total         =[]
lista_neg_porc=[]
lista_pos_porc=[]
lista_tot_porc=[]

#N_rec=50
g=1

def my_init_rec(shape, name=None):
    value = np.random.random(shape)
    #min_=-1
    #max_=1
    mu=0
    sigma=np.sqrt(1/(N_rec))#0.01#/(N_rec*N_rec)#0.05
    #value = np.random.uniform(low=min_, high=max_, size=shape)
    value= g*np.random.normal(mu, sigma, shape)
    return K.variable(value, name=name)

def get_states(model):
    return [K.get_value(s) for s,_ in model.state_updates]

'''
class NonNegLast(Constraint):
    def __call__(self, w):
        last_row = w[-1, :] * K.cast(K.greater_equal(w[-1, :], 0.), K.floatx())
        last_row = K.expand_dims(last_row, axis=0)
        full_w = K.concatenate([w[:-1, :], last_row], axis=0)
        return full_w
'''

'''
class NonNegLast(Constraint):
    def __call__(self, w):
        last_row_2 = w[:, -1] * K.cast(K.less_equal(w[:, -1], 0.), K.floatx())
        last_row_2 = K.expand_dims(last_row_2, -1)
        full_w = K.concatenate([w[:, :-1], last_row_2], -1)

        return full_w
'''
'''
class NonNegLast(Constraint):
    def __call__(self, w):
        last_row = w[:, -1] * K.cast(K.greater_equal(w[:, -1], 0.), K.floatx())
        last_row = K.expand_dims(last_row, axis=1)
        full_w = K.concatenate([w[:, :-1], last_row], axis=1)
        return full_w
'''
class NonNegLast(Constraint):
    def __call__(self, w):
        last_row = w[:, -1] * K.cast(K.greater_equal(w[:, -1], 0.), K.floatx())
        #last_row = w[:, -1] * K.cast(K.less_equal(w[:, -1], 0.), K.floatx())
        last_row = K.expand_dims(last_row, axis=1)
        full_w__ = K.concatenate([w[:, :-3], last_row], axis=1)
        full_w_ = K.concatenate([full_w__, last_row], axis=1)  
        full_w=   K.concatenate([full_w_, last_row], axis=1)
        #last_row_2 = w[:, -2] * K.cast(K.less_equal(w[:, -2], 0.), K.floatx())
        #last_row_2 = K.expand_dims(last_row_2, -1)
        #full_w = K.concatenate([w[:, :-2], last_row_2], -2)
        #full_w = K.concatenate([w[:, :-1], last_row], -1)
        return full_w

class My_init_rec:
    def __call__(self, shape, name=None):
        return my_init_rec(shape, name=name)

#get_custom_objects().update({'my_init_rec': My_init_rec})

#custom_objects={'my_init_rec': my_init_rec}


for root, sub, files in os.walk(r_dir):
    files = sorted(files)
    for i,f in enumerate(files):
       print("file: ",f)
 
       #General network model construction:
       
       model = Sequential()
       
       model = load_model(r_dir+"/"+f)      
       seed(1)

       #model.reset_states()
       #model = load_model(r_dir+"/"+f,  custom_objects={'NonNegLast':NonNegLast})
       # Compiling model for each file:
       #model.layers[0].states[0] = np.zeros(N_rec)
       #model.reset_states()
       #model.layers[0].states[0] =np.ones(6N_rec)

       #model.layers[0].states[0] =np.ones(N_rec)#np.random.rand(N_rec)
       
       '''
       statesAll=[]
       for layer in model.layers:
           print" 1"
           if getattr(layer,'stateful',False):
               if hasattr(layer,'states'):
                   print "a c a "
                   for state in layer[0].states:
                       statesAll.append(K.get_value(state))
                       #statesAll.append(K.set_value(1,state))
                       print state
       print"STATES",statesAll  
       '''
       
       #print"STATES shape",statesAll     
       #model.layers[0].reset_states(states=(64,np.ones(N_rec)))
       #model.layers[0].states[0] =np.random.rand(N_rec)


       #model.reset_states()
       model.compile(loss = 'mse', optimizer='Adam', sample_weight_mode="temporal")
  
       print"-------------",i
       #print"model.layers[0].states[0]",model.layers[0].states[0]
       #print"model get",K.get_value(model.layers[0].)

       #time.sleep(15.5) 
       #Esto me permite imprimir los pesos de la red y la estructura!!!
       for i, layer in enumerate(model.layers):
           print"i-esima capa: ",i 
           print(layer.get_config(), layer.get_weights())

       #time.sleep(15.5)
       
       pesos    = model.layers[0].get_weights()
       pesos__  = model.layers[0].get_weights()[0]
       pesos_in = pesos[0]
       pesos    = model.layers[0].get_weights()[1] 
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

      # wi#th tf.Session() as sess:
      #    tf.global_variables_initializer().run()
      #    h = sess.run(model.layers[0].states[0])

       print "h",model.layers[0].states[0]

       #new_h = K.variable(value=np.ones((1, N_rec))+1)
       #new_h = K.variable(value=np.zeros((1, N_rec)))     
       #model.layers[0].states[0] = new_h
       
       #with tf.Session() as sess:
       #    tf.global_variables_initializer().run()
       #    h = sess.run(model.layers[0].states[0])
       
       #print "h",h
      
       #time.sleep(15)
       #model.layers[0].reset_states([new_h])
       #print "h2",h

       print"-------------\n-------------"   
       print"pesos:\n:",pesos
       print"-------------\n-------------"
       print"N_REC:",N_rec
       unidades        = np.arange(len(pesos))
       conection       = pesos
       #determinante    = tf.matrix_determinant(pesos)

       con_matrix_list.append(pesos)
       print("array: ",np.arange(len(pesos)))       
       #print("biases: ",biases)
       print"##########################"       
       #with tf.Session() as sess:
       #    print (sess.run(determinante))
       print"##########################\n ##########################"
       print"conection",conection       

       #print" determinante", determinante
       
       print"##########################\n ##########################"
     
       histo_lista    =[]
       array_red_list =[]

       peso_mask  =  0.011
       peso_mask_2= -0.011
      
       # Test anulando las conecciones que son mas debiles que el 10% del max o el minimo
       
       conection_usar=conection
       #conection_usar[(conection_usar < peso_mask) & (conection_usar > peso_mask_2)] = 0

       #np.fill_diagonal(conection_usar, 0.0)
       model.layers[0].set_weights([pesos_in,conection_usar])

       from numpy import linalg as LA
       
       w, v = LA.eig(conection_usar)
       print"autovalores\n", w
       print"autovectores\n",v

       print"distancia", np.sqrt(w.real*w.real+w.imag*w.imag)
       lista_dist=np.c_[w,w.real]
       lista_dist_2=np.c_[w,abs(w.real)]
       maximo=max(lista_dist, key=lambda item: item[1])
       print"primer elemento",maximo
       maximo_2=max(lista_dist_2, key=lambda item: item[1])
       marcar=maximo[0]
       marcar_2=maximo_2[0]
       print"maximo marcar",marcar
       
       frecuency=0
       if marcar_2.imag==0:
           frecuency =0
       else: 
           frecuency =abs(float(marcar_2.imag)/(3.14159*float(marcar_2.real)))
           
       print "frecuency",frecuency
       #time.sleep(15.5) 

       plt.figure(figsize=(12,10))
       plt.scatter(w.real,w.imag,color="hotpink",label="Eigenvalue spectrum\n ")#Total of: "+str(len(w.real))+" values")
       #plt.scatter(w.real,w.imag,color="plum",label="Eigenvalue spectrum\n Total of: "+str(len(w.real))+" values")

       # for plotting circle line:
       a = np.linspace(0, 2*np.pi, 500)
       cx,cy = np.cos(a), np.sin(a)
       plt.plot(cx, cy,'--', alpha=.5, color="dimgrey") # draw unit circle line
       #plt.plot(2*cx, 2*cy,'--', alpha=.5)
       plt.scatter(marcar.real,marcar.imag,color="red", label="Eigenvalue with maximum real part")
       #plt.scatter(marcar_2.real,marcar_2.imag,color="blue", label="Eigenvalue with maximum abs(Real part)\n"+"Frecuency: "+str(frecuency))
       plt.plot([0,marcar.real],[0,marcar.imag],'k-')
       #plt.plot([0,marcar_2.real],[0,marcar_2.imag],'k-')
       plt.axvline(x=1,color="salmon",linestyle='--')


       plt.xlabel(r'$Re( \lambda)$',fontsize = 20)
       plt.ylabel(r'$Im( \lambda)$',fontsize = 20)
       plt.legend(fontsize= 'large',loc=1)
       plt.savefig("plots_and/autoval_"+str(i)+"_"+str(f)+"_"+str(peso_mask)+"_.png",dpi=200)
       plt.close()
       
        # Plots 
       ################################### Here we plot iner state of the network with the desierd stimulus
       
       for sample_number in np.arange(sample_size_3):
          print ("sample_number",sample_number)
          print_sample = plot_sample(sample_number,2,neurons,x_train,y_train,model,seq_dur,i,plot_dir,f)

       ##################################   

       #################################### Bias & Weights as scatter plot ######################
       '''       
       plt.figure(figsize=(14,12)) 
       plt.subplot(2, 1, 1)        
       #plt.scatter(unidades,biases, label="Bias value for each unit")
       plt.hlines(y=0.01, xmin=-1, xmax=N_rec +1, linewidth=2, color='k')
       plt.hlines(y=-0.01, xmin=-1, xmax=N_rec +1, linewidth=2, color='k')
       plt.xlim([-1,N_rec +1])
       plt.legend(fontsize= 'medium',loc=1)
       plt.xlabel('Unit',fontsize = 20)
       plt.ylim([-0.065,0.065])

       plt.subplot(2, 1, 2)               
       for ii in unidades:
           histo_lista.extend(pesos[ii])
           plt.scatter(unidades,pesos[ii],color=colors[ii])
       plt.hlines(y=peso_mask, xmin=-1, xmax=N_rec +1, linewidth=2, color='k')
       plt.hlines(y=-peso_mask, xmin=-1, xmax=N_rec +1, linewidth=2, color='k')
       plt.xlim([-1,N_rec +1])
       plt.ylim([-0.45,0.45])
       plt.xlabel('Unit',fontsize = 20)
       plt.legend(fontsize= 'medium',loc=1)
       #plt.savefig("plots_and/weight_bias_"+str(i)+"_"+str(f)+"_0.png",dpi=200)
       plt.close()
       
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

       plt.hist(histo_lista, bins=200,color='c',normed=1,label="Weight Value \n Mu= "+str(mu)+"\n Sigma= "+str(sigma))
       plt.plot(bins, y, 'r-', linewidth=3)
       plt.axvline(mu, color='r', linestyle='dashed', linewidth=2)
       plt.vlines(x=sigma, ymin=0, ymax=900, linewidth=2, color='k')
       plt.vlines(x=-sigma,ymin=0, ymax=900, linewidth=2, color='k')
       plt.xlabel('Weight strength [arb. units]',fontsize = 16)
       plt.ylim([0,10])
       plt.legend(fontsize= 'medium',loc=1)
       plt.savefig("plots_and/weight_histo_"+str(i)+"_"+str(f)+'_'+str(peso_mask)+"_0.png",dpi=200)
       plt.close()
     
      #################################### Conectivity matrix: positive or excitatory weights ###################################
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
       plt.ylabel('Unit [i]',fontsize = 16)
       plt.xlabel('Unit [j]',fontsize = 16)
       #plt.legend(fontsize= 'medium',loc=1)
       #plt.savefig("plots_and/Pos_conection_matrix_"+str(i)+"_"+str(f)+'_'+str(peso_mask)+"_0.png",dpi=200)
       plt.close()
      
       #################################### Conectivity matrix: Negative or inhibitory weights ##################################
       
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
       plt.ylabel('Unit [i]',fontsize = 16)
       plt.xlabel('Unit [j]',fontsize = 16)
       #plt.legend(fontsize= 'medium',loc=1)
       #plt.savefig("plots_and/Neg_conection_matrix_"+str(i)+"_"+str(f)+'_'+str(peso_mask)+"0.png",dpi=200)
       plt.close()
       zzz=np.zeros(50)
       #print ("biases shape",conection.shape,zzz.shape)
       #print(model.layers[0].get_config())
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
       
       conection[abs(conection) < peso_mask ] = 0
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
       plt.ylim([-1,N_rec +1])
       plt.ylabel('Unit [i]',fontsize = 16)
       plt.xlabel('Unit [j]',fontsize = 16)
       #plt.legend(fontsize= 'medium',loc=1)
       #plt.savefig("plots_and/filt_conection_matrix_"+str(i)+"_"+str(f)+"_0.png",dpi=200)
       plt.close()
       '''
       # Model Testing: 
       x_pred = x_train[0:10,:,:]
       y_pred = model.predict(x_pred)

       print("x_train shape:\n",x_train.shape)
       print("x_pred shape\n",x_pred.shape)
       print( "y_train shape\n",y_train.shape)

       lista_distancia=[]
       
       fig     = plt.figure(figsize=(6,8))
       fig.suptitle("\"And\" Data Set Trainined Output \n (amplitude in arb. units time in mSec)",fontsize = 20)
       for ii in np.arange(10):
           plt.subplot(5, 2, ii + 1)    
           plt.plot(x_train[ii, :, 0],color='g',label="inpt A")
           plt.plot(x_train[ii, :, 1],color='b',label="input B")
           plt.plot(y_train[ii, :, 0],color='k',linewidth=3,label="Desierd output")
           plt.plot(y_pred[ii, :, 0], color='r',label="Predicted Output")
           plt.ylim([-2.5, 2.5])
           plt.legend(fontsize= 5,loc=3)
           plt.xticks(fontsize=8)
           plt.yticks(fontsize=8)
           a=y_train[ii, :, 0]
           b=y_pred[ii, :, 0]
           a_min_b = np.linalg.norm(a-b)      
           lista_distancia.append(a_min_b)
           figname =plot_dir+"/data_set_and_sample_trained_"+str(peso_mask)+'_'+str(f)+".png" 
           figname = "plots_and/data_set_and_sample_trained.png" 
           plt.savefig(figname,dpi=200)
       
       lista_distancia.insert(0,N_rec)
       lista_distancia_all.append(lista_distancia)
'''
todo         = np.c_[lista_neg,lista_pos,total,lista_neg_porc,lista_pos_porc,lista_tot_porc]
todo_2       = lista_distancia_all
print (todo_2)

np.savetxt(f_out,todo,fmt='%f %f %f %f %f %f',delimiter='\t',header="Negative<"+str(peso_mask_2)+" #Positive>"+str(peso_mask)+" #total #%_neg #%_pos #%_total")

np.savetxt(f_out_dist,todo_2,fmt='%f %f %f %f %f %f %f %f %f %f %f',delimiter='\t',header="Nrec #S1 #S2 #S3 #S4 #S5 #S6 #S7 #S8 #S9 #S10")
'''
# Difference between initial and final state
'''
matrix_diff = 100*(con_matrix_list[-1]-con_matrix_list[0])/con_matrix_list[0]

print("con_matrix_list: ", matrix_diff)

#ax = plt.f#igure(figsize=(14,12)) 
fig, ax = plt.subplots(figsize=(14,12))
plt.title('Conection matrix \% difference betwwen initial and final state', fontsize = 17)#, weight="bold")
grid(True)

cmap          = plt.cm.gist_ncar # Tipos de mapeo #OrRd # 'gist_ncar'#"plasma"
matrix_diff_2   = np.ma.masked_where(abs(matrix_diff)< 10, matrix_diff)

#conection_pos = np.ma.masked_where(conection < 0.0, conection)
cmap.set_bad(color='white')

matrix_diff_3   = np.ma.masked_where(abs(matrix_diff_2)> 200, matrix_diff_2)
cmap.set_bad(color='white')

cax =ax.imshow(matrix_diff_3,cmap=cmap,interpolation="none",label='Conection matrix')
       
cbar_max  =100 #0.75
cbar_min  =-100# -0.75
cbar_step = 10 #0.1#0.025
       
cbar = fig.colorbar(cax)#,ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step))
#cbar = fig.colorbar(cax,ticks= [-0.25,-0.1,0,0.1,0.25])
#plt.colorbar(ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step))
cbar.ax.set_ylabel('% Weight variation', fontsize = 16, weight="bold")
       
plt.xlim([-1,N_rec +1])
plt.ylim([-1,N_rec +1])
plt.ylabel('Unit [i]',fontsize = 16)
plt.xlabel('Unit [j]',fontsize = 16)
#plt.legend(fontsize= 'medium',loc=1)
plt.savefig("plots_and/Diff_conection_matrix_"+str(i)+"_0.png",dpi=200)
plt.close()
'''



