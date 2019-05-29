##########################################################
#                 Author C. Jarne                        #
#            binary_and_recurrent_main.py  (ver 2.0)     #                       
#  Based on a Keras-Cog task from Alexander Atanasov     #
#  A or task (high to low edge triggered)                #                
#                                                        #
# MIT LICENCE                                            #
##########################################################

import numpy as np
import matplotlib.pyplot as plt
import time

from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint, Callback, warnings
from keras.layers.recurrent import SimpleRNN
from keras.layers import TimeDistributed, Dense, Activation, Dropout
from keras.utils import plot_model
from keras import metrics
from keras import optimizers
from keras import regularizers
from keras import initializers
import keras
# taking dataset from function
from generate_data_set_or import *

#start_time = time.time()

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value   = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print(" Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def and_fun(t,N_rec,base,base_plot):
    lista_distancia=[]
    #Parameters

    sample_size      = 15050#8242 # (2^11 samples +50 for testing its a good value 2^13 +50 is better)
    epochs           = 20
    #N_rec            = 50 #100
    p_connect        = 0.9

    #to be used in the Simple rnn redefined (not yet implemented)
    dale_ratio       = 0.8
    tau              = 100
    mem_gap          = t

    pepe=keras.initializers.RandomNormal(mean=0.0, stddev=1*np.sqrt(float(1)/float((N_rec))), seed=None)
    x_train,y_train, mask,seq_dur = generate_trials(sample_size,mem_gap) #time loop
    #x_train,y_train, mask,seq_dur = generate_trials(sample_size) #no time loop

    #Network model construction
    seed(None)# cambie el seed    
    model = Sequential()
    model.add(SimpleRNN(units=N_rec,return_sequences=True, input_shape=(None, 2), kernel_initializer='glorot_uniform',      recurrent_initializer='orthogonal'  ,activation='tanh',use_bias=False))
    #model.add(SimpleRNN(units=N_rec,return_sequences=True, input_shape=(None, 2), kernel_initializer='glorot_uniform',      recurrent_initializer='orthogonal',activation='tanh',use_bias=True,bias_initializer='zeros')) #defaults for the recurrent model!
    model.add(Dense(units=1,input_dim=N_rec))#,activation="linear"softplus
    #model.compile(loss = 'mse', optimizer='Adam', sample_weight_mode="temporal")
    #model.compile(loss ='mean_absolute_percentage_error', optimizer='Adam', sample_weight_mode="temporal")

    #model.save('weights_and/initial.hdf5')
    model.save(base+'/00_initial.hdf5')
    # Model Compiling:
    ADAM           = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0001)
    model.compile(loss = 'mse', optimizer=ADAM, sample_weight_mode="temporal")

    # Saving weigths
    filepath       = base+'/and_weights-{epoch:02d}.hdf5'
    #checkpoint    = ModelCheckpoint(filepath, monitor='accuracy')
    #checkpoint     = ModelCheckpoint(filepath)
    callbacks      = [EarlyStoppingByLossVal(monitor='loss', value=0.0001, verbose=1), ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, verbose=1),]


    #from keras.callbacks import TensorBoard
    #tensorboard = TensorBoard(log_di r='./logs', histogram_freq=0, write_graph=True, write_images=False)

    history      = model.fit(x_train[50:sample_size,:,:], y_train[50:sample_size,:,:], epochs=epochs, batch_size=64, callbacks = callbacks,     sample_weight=mask[50:sample_size,:])

    #callbacks=[tensorboard]
    #callbacks = [checkpoint]

    # Model Testing: 
    x_pred = x_train[0:50,:,:]
    y_pred = model.predict(x_pred)

    print("x_train shape:\n",x_train.shape)
    print("x_pred shape\n",x_pred.shape)
    print("y_train shape\n",y_train.shape)

    fig     = plt.figure(figsize=(6,8))
    fig.suptitle("\"Or\" Data Set Trainined Output \n (amplitude in arb. units time in mSec)",fontsize = 20)
    for ii in np.arange(10):
        plt.subplot(5, 2, ii + 1)    
        plt.plot(x_train[ii, :, 0],color='g',label="input A")
        plt.plot(x_train[ii, :, 1],color='b',label="input B")
        plt.plot(y_train[ii, :, 0],color='k',linewidth=3,label="Desierd output")
        plt.plot(y_pred[ii, :, 0], color='r',label="Predicted Output")
        plt.ylim([-2.5, 2.5])
        plt.legend(fontsize= 5,loc=3)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        figname =  base_plot+"/data_set_and_sample_trained.png" 
        #figname = "plots_and/data_set_and_sample_trained.png" 
        plt.savefig(figname,dpi=200)
        a=y_train[ii, :, 0]
        b=y_pred[ii, :, 0]
        a_min_b = np.linalg.norm(a-b)      
        lista_distancia.append(a_min_b)
    #plt.close()
    #plt.show()


    print(model.summary())
    plot_model(model, to_file='plots_or/model.png')

    print ("history keys",(history.history.keys()))

    #print("--- %s to train the network seconds ---" % (time.time() - start_time))

    fig     = plt.figure(figsize=(8,6))
    plt.grid(True)
    plt.plot(history.history['loss'])
    plt.title('Model loss during training')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #figname = "plots_and/model_loss.png" 
    figname = base_plot+"/model_loss.png" 
    plt.savefig(figname,dpi=200)

    '''
    plt.figure()  
    plt.grid(True)
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_loss'])
    plt.title('accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    figname = "plots/accuracy.png" 
    plt.savefig(figname,dpi=200)
    '''
    #plt.show()
    return lista_distancia



