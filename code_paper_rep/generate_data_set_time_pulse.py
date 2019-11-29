#########################################
#                                       #
#     A "time pulse" data set generator #
#            of samples                 #
#     with adjutable parameters         #
#                                       #
#   Mit License C. Jarne V. 1.0 2018    #
#                                       #
#########################################

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal
from numpy.random import seed
from scipy import signal

start_time = time.time()

def generate_trials(size,mem_gap):
    seed(2)
    # mem_gap          = 200 # output reaction length
    first_in         = 50#time to start the first stimulus  
    stim_dur         = 20  #stimulus duration
    stim_noise       = 0.03 #noise
    var_delay_length = 0    #change for a variable length stimulus
    out_gap          = 250-20#-60#50 #how much lenth add to the sequence duration   
    sample_size      = size # sample size
    rec_noise        = 0
    
    xor_seed_A = np.array([[0],[1]])
    
    xor_y            = np.array([0,1])
    seq_dur          = first_in+stim_dur+mem_gap+var_delay_length+(out_gap-mem_gap)
    win              = signal.hann(10)
 
    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=np.int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1
    second_in = first_in + stim_dur + mem_gap
    
    out_t          = mem_gap+ first_in+stim_dur
    x              = np.arange(seq_dur)
    trial_types    = np.random.randint(2, size=sample_size)
    x_train_       = np.zeros((sample_size, seq_dur, 1))    
    x_train        = np.zeros((sample_size, seq_dur, 1))        
    y_train_       = 0.045* np.ones((sample_size, seq_dur, 1))
    y_train        = 0.045* np.ones((sample_size, seq_dur, 1))
    
    for ii in np.arange(sample_size): 
        x_train_[ii, first_in:first_in + stim_dur, 0]   = xor_seed_A[trial_types[ii], 0]
        x_train[ii, first_in:first_in + stim_dur, 0]    = signal.convolve(x_train_[ii, first_in:first_in + stim_dur, 0], win, mode='same') / sum(win) 
        #x_train[ii, first_in:first_in + stim_dur, 0]    = signal.convolve(x_train_[ii, first_in:first_in + stim_dur, 0],0.05*signal.gaussian(stim_dur, std=12), mode='same')
        y_train_[ii, out_t + var_delay[ii]:out_t+stim_dur, 0]  = xor_y[trial_types[ii]]
        #y_train[ii, out_t + var_delay[ii]:, 0]=signal.convolve(y_train_[ii, out_t + var_delay[ii]:, 0], np.sin(2*np.pi*(float(frec)/float(seq_dur))*x), mode='same')
        y_train[ii, out_t + var_delay[ii]:out_t+stim_dur, 0]=signal.convolve(y_train_[ii, out_t + var_delay[ii]:out_t+stim_dur, 0], 0.055*signal.gaussian(stim_dur, std=12), mode='same')

    mask = np.zeros((sample_size, seq_dur))
    for sample in np.arange(sample_size):
        mask[sample,:] = [1 for y in y_train[sample,:,:]]
       
    x_train = x_train + stim_noise * np.random.randn(sample_size, seq_dur, 1)
    y_train = y_train + stim_noise * np.random.randn(sample_size, seq_dur, 1)
   
    print("--- %s seconds to generate Pulsee learning dataset---" % (time.time() - start_time))
    return(x_train, y_train,mask,seq_dur)
   
#To test the rule that you want to teach

sample_size=10

x_train,y_train, mask,seq_dur = generate_trials(sample_size,20) 
'''
fig     = plt.figure(figsize=(6,8))
fig.suptitle("\"Time pulse\" Data Set Training Sample\n (amplitude in arb. units time in mSec)",fontsize = 20)
for ii in np.arange(10):
    plt.subplot(5, 2, ii + 1)    
    plt.plot(x_train[ii, :, 0],color='g',label="input")
    plt.plot(y_train[ii, :, 0],color='k',label="output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize= 5,loc=3)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    figname = "plots_pulse/data_set_pulse_sample.png" 
    plt.savefig(figname,dpi=200)
plt.show()
'''

