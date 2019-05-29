#########################################
#                                       #
#     A "Flip Flop" data set generator  #
#            of samples                 #
#     with adjutable parameters         #
#                                       #
#   Mit License C. Jarne V. 1.0 2018    #
#                                       #
#########################################

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.random import seed

start_time = time.time()

def generate_data_set(sample_size, mem_gap):

    #Parameters of the data set
    seed(1)
    nturns           = 8
    input_wait       = 50
    quiet_gap        = 100
    stim_dur         = 20 #stimulus duration
    var_delay_length = 0#200 #change for a variable length stimulus
    stim_noise       = 0.1 #noise
    rec_noise        = 0    
    #mem_gap          = 10 # output reaction time
        
    #To control the length if it is fix or variable

    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1
     
    input_times  = np.zeros([sample_size, nturns],dtype=np.int)
    output_times = np.zeros([sample_size, nturns],dtype=np.int)    
    turn_time    = np.zeros(sample_size, dtype=np.int)
    
    for sample in np.arange(sample_size):
        turn_time[sample] =  stim_dur + quiet_gap + var_delay[sample]
        for i in np.arange(nturns): 
            input_times[sample, i]  = input_wait + i * turn_time[sample]
            output_times[sample, i] = input_wait + i * turn_time[sample] + stim_dur

    seq_dur = int(max([output_times[sample, nturns-1] + quiet_gap, sample in np.arange(sample_size)]))
    x_train = np.zeros([sample_size, seq_dur, 2])
    y_train = 0.01 * np.ones([sample_size, seq_dur, 1])

    for sample in np.arange(sample_size):
        for turn in np.arange(nturns):
            firing_neuron = np.random.randint(2)                # 0 or 1
            
            x_train[sample, input_times[sample, turn]:(input_times[sample, turn] + stim_dur),firing_neuron] = 1 #Puts 1 on Set or Reset at Random
       
            y_train[sample, output_times[sample, turn]+mem_gap:(input_times[sample, turn] + turn_time[sample]+ stim_dur)+mem_gap,0] = firing_neuron    
            #output value is firing rate
            #If reset is 1(0# component of the vector ) output is zero, if 1 is in set (vector 1# component) output is 1
     
    #Here we add noise on the input

    x_train = x_train + stim_noise * np.random.randn(sample_size, seq_dur, 2)
    #y_train = y_train + stim_noise * np.random.randn(sample_size, seq_dur, 1)
    mask = np.zeros((sample_size, seq_dur))

    for sample in np.arange(sample_size):        
        mask[sample,:] = [1 for x in y_train[sample,:,:]]
    print("--- %s seconds to generate dataset---" % (time.time() - start_time))
    
    return x_train,y_train, seq_dur,mask

#To see how is the training data set uncoment these lines

sample_size=10

x_train,y_train, mask,seq_dur = generate_data_set(sample_size,20) 

#print ("x",x_train)
#print ("y",y_train)

fig     = plt.figure(figsize=(6,8))
fig.suptitle("\"Flip-Flop\" Data Set Training Sample\n (amplitude in arb. units time in mSec)",fontsize = 20)
for ii in np.arange(10):
    plt.subplot(5, 2, ii + 1)
    
    plt.plot(x_train[ii, :, 0],color='g',label="input Set")
    plt.plot(x_train[ii, :, 1],color='b',label="input Reset")
    plt.plot(y_train[ii, :, 0],color='k',label="output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize= 5,loc=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    figname = "plots_ff/data_set_flip_flop_sample.png" 
    plt.savefig(figname,dpi=200)
plt.show()
    

