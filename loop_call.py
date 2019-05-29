##########################################################
#                 Author C. Jarne                        #
#               call loop  (ver 1.0)                     #                       
# MIT LICENCE                                            #
##########################################################

import os
import time
#import numpy as np
#from binary_and_recurrent_main_to_loop import *
#from binary_and_recurrent_main_to_loop_inicial_state import *
#from binary_and_recurrent_main_to_loop_test_exi_ini import *
#from binary_or_recurrent_main_to_loop import *
#from binary_xor_recurrent_main_to_loop import *
#from binary_ff_recurrent_main_to_loop import *
from binary_pulse_recurrent_main_to_loop import *
#from binary_osc_recurrent_main_to_loop import *
#from binary_not_recurrent_main_to_loop import *

start_time = time.time()
time_vector=[50]#[100,105,110,120,130,140,150]

#[80,130,170,300,400,450] #[160,180,350]#np.arange(0,20,1)#(9,20,1)#[20]#[60,70,80,100,150,200,250,300,350,450,500,550,600,650,700]#[5,10,20,30,40,50,60,70,80,90,100,110]
#time_vector= [2,1]#[60,70,80,90,100,110,120,130,140,150,200,250,300,350]
#time_vector=[200,250,300,350]
#time_vector=[30,35,40,45,100,150,200,250,300,350,400,450,500,550]

#f          ='weights_and'
#f_plot     ='plots_and'

#f          ='weights_or'
#f_plot     ='plots_or'


#f          ='weights_xor'
#f_plot     ='plots_xor'

#f          ='weights_ff'
#f_plot     ='plots_ff'

f          ='weights_pulse'
f_plot     ='plots_pulse'

#f          ='weights_osc'
#f_plot     ='plots_osc'

#f          ='weights_not'
#f_plot     ='plots_not'

distancias = []

for t in time_vector:
    for i in np.arange(0,20,1):
        mem_gap = 20#150#150#20  #200
        N_rec   =t# 50#100#
        base= f+'/'+  os.path.basename(f+'_'+str(mem_gap)+'_N_'+str(N_rec)+'_gap_'+str(i))
        base_plot= f_plot+'/'+  os.path.basename(f_plot+'_'+str(t)+'_N_'+str(i))
        dir = str(base)
        if not os.path.exists(dir):
           os.mkdir(base)
        print(str(dir))

        dir = str(base_plot)
        if not os.path.exists(dir):
           os.mkdir(base_plot)        
        print(str(dir))
    
        pepe    =and_fun(mem_gap,N_rec,base,base_plot)
        distancias.append(pepe)
print('-------------------------')
print distancias
print("--- %s to train the network seconds ---" % (time.time() - start_time))
