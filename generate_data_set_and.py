#########################################
#                                       #
#     An "And" data set generator       #
#            of samples                 #
#     with adjutable parameters         #
#                                       #
#   Mit License C. Jarne V. 1.0 2018    #
#########################################

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal
from numpy.random import seed
start_time = time.time()

#def generate_trials(size): #No time loop
def generate_trials(size, mem_gap):#time loop
    #seed(1)
    seed(2)
    #mem_gap          = 20 # output reaction time
    first_in         = 30 #time to start the first stimulus   #30 #60
    stim_dur         = 20 #stimulus duration #20 #30
    stim_noise       = 0.1 #noise
    var_delay_length = 0 #change for a variable length stimulus
    out_gap          = 130#130 #how much lenth add to the sequence duration    #140 #100
    sample_size      = size # sample size
    rec_noise        = 0
       
    and_seed_A = 1*np.array([[0],[1],[0],[1]])
    and_seed_B = 1*np.array([[0],[0],[1],[1]])
    and_y            =1* np.array([0,0,0,1])
    seq_dur          = first_in+stim_dur+mem_gap+var_delay_length+out_gap #Sequence duration

    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=np.int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1
    second_in = first_in + stim_dur + mem_gap

    win              = signal.hann(10)

    out_t       = mem_gap+ first_in+stim_dur
    trial_types = np.random.randint(4, size=sample_size)
    x_train     = np.zeros((sample_size, seq_dur, 2))
    x_train_       = np.zeros((sample_size, seq_dur, 2)) 
    y_train     =  np.zeros((sample_size, seq_dur, 1))

    for ii in np.arange(sample_size):
        x_train[ii, first_in:first_in + stim_dur, 0] = and_seed_A[trial_types[ii], 0]
        x_train[ii, first_in:first_in + stim_dur, 1] = and_seed_B[trial_types[ii], 0]
        #x_train[ii, first_in:first_in + stim_dur, 0]    = signal.convolve(x_train_[ii, first_in:first_in + stim_dur, 0],0.1*signal.gaussian(stim_dur*.25, std=12), mode='same')
        #x_train[ii, first_in:first_in + stim_dur, 1]    = signal.convolve(x_train_[ii, first_in:first_in + stim_dur, 1],0.1*signal.gaussian(stim_dur*.25, std=12), mode='same')
        y_train[ii, out_t + var_delay[ii]:, 0] = and_y[trial_types[ii]]

    mask = np.zeros((sample_size, seq_dur))
    for sample in np.arange(sample_size):
        mask[sample,:] = [1 for y in y_train[sample,:,:]]
       
    x_train = x_train#+stim_noise * np.random.randn(sample_size, seq_dur, 2)##+np.ones((sample_size, seq_dur, 2) )#
    print("--- %s seconds to generate And dataset---" % (time.time() - start_time))
    return (x_train, y_train, mask,seq_dur)


#To see how is the training data set uncoment these lines

sample_size=10
mem_gap=20
x_train,y_train, mask,seq_dur = generate_trials(sample_size,mem_gap) 

#print ("x",x_train)
#print ("y",y_train)

fig     = plt.figure(figsize=(6,8))
fig.suptitle("\"And\" Data Set Training Sample\n (amplitude in arb. units time in mSec)",fontsize = 20)
for ii in np.arange(10):
    plt.subplot(5, 2, ii + 1)
    
    plt.plot(x_train[ii, :, 0],color='g',label="input A")
    plt.plot(x_train[ii, :, 1],color='b',label="input B")
    plt.plot(y_train[ii, :, 0],color='k',label="output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize= 5,loc=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    figname = "plots_and/data_set_and_sample.png"
    #figname = base+"/data_set_and_sample.png" 
    plt.savefig(figname,dpi=200)
plt.show()


