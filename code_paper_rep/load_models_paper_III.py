import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)



fname      ="Networks_and_files/files_scale/time_delay.txt"
fname_2    ="Networks_and_files/files_scale/size_at_150_ms.txt"


pepe       = np.loadtxt(fname,delimiter=" ")
w       = pepe.T
print (pepe)
pepe_2  = np.loadtxt(fname_2,delimiter=" ")
w_2     =pepe_2.T
print (pepe_2)

x_axis=w[0]
y_axis=w[1]
y_axis_error=w[2]

x_axis_2=w_2[0]
y_axis_2=w_2[1]
y_axis_error_2=w_2[2]

fig     = plt.figure(figsize=cm2inch(18,8.5))

plt.axhline(y=min(y_axis), color='pink', linestyle='--')
plt.axhline(y=max(y_axis), color='pink', linestyle='--')

plt.axvline(x=  25, color='pink', linestyle='--')
plt.errorbar(x_axis,y_axis, xerr=0, yerr=y_axis_error,marker="o", markersize=5,fmt='o',color="deeppink",label="Distance target-output for task of pulse memorization")
plt.xlabel('Time Delay [mS]',fontsize = 12)
plt.ylabel('Distance',fontsize = 12)
plt.legend(fontsize= 10,loc=1)
plt.ylim([0.8,2.35])
plt.savefig("plots_paper/rate_delay_"+"0_.png",dpi=300,bbox_inches = 'tight')
plt.close()

fig     = plt.figure(figsize=cm2inch(17,8.5))

plt.axhline(y=min(y_axis_2), color='pink', linestyle='--')
plt.axhline(y=max(y_axis_2), color='pink', linestyle='--')
plt.errorbar(x_axis_2,y_axis_2, xerr=0, yerr=y_axis_error_2,marker="o", markersize=5,fmt='o',color="deeppink",label="Distance target-output for task of pulse memorization")
plt.xlabel('Network size [units]',fontsize = 12)
plt.ylabel('Distance',fontsize = 12)
plt.legend(fontsize= 10,loc=1)
plt.ylim([0.85,2.2])
plt.xticks(np.arange(0, max(x_axis_2)+50, 50.0))
plt.xlim([20,520])
plt.savefig("plots_paper/rate_units_"+"0.png",dpi=300,bbox_inches = 'tight')
plt.close()

