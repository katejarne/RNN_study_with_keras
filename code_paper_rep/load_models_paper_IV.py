import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

fname="Networks_and_files/files_damage/05_franjas_neg/aug_20_.txt"

pepe    = np.loadtxt(fname,delimiter=" ")
#pepe    = np.genfromtxt(fname,delimiter="\t")
w       = pepe.T
print (pepe)


fname_d="Networks_and_files/files_damage/04_franjas_pos/aug_20_.txt"

pepe_d    = np.loadtxt(fname_d,delimiter=" ")
#pepe    = np.genfromtxt(fname,delimiter="\t")
w_d       = pepe_d.T
print (pepe_d)

fname_c="Networks_and_files/files_damage/03_negativos/aug_20_.txt"

pepe_c    = np.loadtxt(fname_c,delimiter=" ")
#pepe    = np.genfromtxt(fname,delimiter="\t")
w_c       = pepe_c.T
print (pepe_c)


fname_b="Networks_and_files/files_damage/02_positivos/aug_20_.txt"

pepe_b    = np.loadtxt(fname_b,delimiter=" ")
#pepe    = np.genfromtxt(fname,delimiter="\t")
w_b       = pepe_b.T
print (pepe_b)


fname_a="Networks_and_files/files_damage/01_pos_y_neg/aug_20_.txt"

pepe_a    = np.loadtxt(fname_a,delimiter=" ")
#pepe    = np.genfromtxt(fname,delimiter="\t")
w_a       = pepe_a.T
print (pepe_a)

#lista_tot_porc__ = w[1]
#
#For partial cut
lista_tot_porc__ =np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36])*0.5

lista_tot_porc__a =np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36])

print(lista_tot_porc__)
acc_mean_0 =w[2]
acc_mean_1 =w[4]
acc_mean_2 =w[6]
acc_mean_3 =w[8]

acc_mean_0_d =w_d[2]
acc_mean_1_d =w_d[4]
acc_mean_2_d =w_d[6]
acc_mean_3_d =w_d[8]


acc_mean_0_c =w_c[2]
acc_mean_1_c =w_c[4]
acc_mean_2_c =w_c[6]
acc_mean_3_c =w_c[8]

acc_mean_0_b =w_b[2]
acc_mean_1_b =w_b[4]
acc_mean_2_b =w_b[6]
acc_mean_3_b =w_b[8]


acc_mean_0_a =w_a[2]
acc_mean_1_a =w_a[4]
acc_mean_2_a =w_a[6]
acc_mean_3_a =w_a[8]

'''
acc_mean_0 =w[1]/acc_mean_0[0]
acc_mean_1 =w[3]/acc_mean_1[0]
acc_mean_2 =w[5]/acc_mean_2[0]
acc_mean_3 =w[7]/acc_mean_3[0]
'''
acc_uncertanty_0= w[3]#/acc_mean_0[0]
acc_uncertanty_1= w[5]#/acc_mean_1[0]
acc_uncertanty_2= w[7]#/acc_mean_2[0]
acc_uncertanty_3= w[9]#/acc_mean_3[0]


acc_uncertanty_0_d= w_d[3]#/acc_mean_0[0]
acc_uncertanty_1_d= w_d[5]#/acc_mean_1[0]
acc_uncertanty_2_d= w_d[7]#/acc_mean_2[0]
acc_uncertanty_3_d= w_d[9]#/acc_mean_3[0]


acc_uncertanty_0_c= w_c[3]#/acc_mean_0[0]
acc_uncertanty_1_c= w_c[5]#/acc_mean_1[0]
acc_uncertanty_2_c= w_c[7]#/acc_mean_2[0]
acc_uncertanty_3_c= w_c[9]#/acc_mean_3[0]


acc_uncertanty_0_b= w_b[3]#/acc_mean_0[0]
acc_uncertanty_1_b= w_b[5]#/acc_mean_1[0]
acc_uncertanty_2_b= w_b[7]#/acc_mean_2[0]
acc_uncertanty_3_b= w_b[9]#/acc_mean_3[0]


acc_uncertanty_0_a= w_a[3]#/acc_mean_0[0]
acc_uncertanty_1_a= w_a[5]#/acc_mean_1[0]
acc_uncertanty_2_a= w_a[7]#/acc_mean_2[0]
acc_uncertanty_3_a= w_a[9]#/acc_mean_3[0]

########### Fig a) ############

fig     = plt.figure(figsize=cm2inch(12,6))

#porcen                 = np.arange(100,min(lista_tot_porc__),max(lista_tot_porc__))

plt.fill_between(lista_tot_porc__a,acc_mean_0_a-acc_uncertanty_0_a*1/float(11),acc_mean_0_a+acc_uncertanty_0_a*1/float(12), where=acc_mean_0_a+acc_uncertanty_0_a*1/float(11) >= acc_mean_0_a-acc_uncertanty_0_a*1/float(11), facecolor='lightblue', interpolate=True)

plt.fill_between(lista_tot_porc__a,acc_mean_1_a-acc_uncertanty_1_a*1/float(11),acc_mean_1_a+acc_uncertanty_1_a*1/float(12), where=acc_mean_1_a+acc_uncertanty_1_a*1/float(11) >= acc_mean_1_a-acc_uncertanty_1_a*1/float(11), facecolor='mistyrose', interpolate=True)

plt.fill_between(lista_tot_porc__a,acc_mean_2_a-acc_uncertanty_2_a*1/float(11),acc_mean_2_a+acc_uncertanty_2_a*1/float(12), where=acc_mean_2_a+acc_uncertanty_2_a*1/float(11) >= acc_mean_2_a-acc_uncertanty_2_a*1/float(11), facecolor='pink', interpolate=True)

plt.fill_between(lista_tot_porc__a,acc_mean_3_a-acc_uncertanty_3_a*1/float(11),acc_mean_3_a+acc_uncertanty_3_a*1/float(12), where=acc_mean_3_a+acc_uncertanty_3_a*1/float(11) >= acc_mean_3_a-acc_uncertanty_3_a*1/float(11), facecolor='lightgreen', interpolate=True)

plt.errorbar(lista_tot_porc__a,acc_mean_0_a, xerr=0, yerr=acc_uncertanty_0_a*1/float(11),marker="o", markersize=2,fmt='o',color="blue",label="00 Level Distance")
plt.errorbar(lista_tot_porc__a,acc_mean_1_a, xerr=0, yerr=acc_uncertanty_1_a*1/float(11),marker="o",markersize=2, fmt='o',color="red",label="11 Level Distance")
plt.errorbar(lista_tot_porc__a,acc_mean_2_a, xerr=0, yerr=acc_uncertanty_2_a*1/float(11),marker="o",markersize=2, fmt='o',color="deeppink",label="10 Level Distance")
plt.errorbar(lista_tot_porc__a,acc_mean_3_a, xerr=0, yerr=acc_uncertanty_3_a*1/float(11),marker="o",markersize=2, fmt='o',color="green",label="01 Level Distance")
plt.axhline(y=1, color='pink', linestyle='--')
plt.axhline(y=1.5, color='pink', linestyle='--')
plt.ylabel('Distance output-target',fontsize = 6)
plt.xlabel('% of connections removed',fontsize = 6)
plt.legend(fontsize=5,loc=2)
plt.ylim([-0.25,8.1])
# plt.xlim([-0,160])
#plt.ylim([-1,max(lista_distancia_all_3)+3])
plt.xticks(np.arange(0, max(lista_tot_porc__a)+2, 2.0),fontsize = 5)
plt.yticks(np.arange(0,8.1,2),fontsize = 5)
plt.savefig("plots_paper/Test_man_distance_a"+".png",dpi=300, bbox_inches = 'tight')
#plt.show()
plt.close()



########### Fig b) ############

fig     = plt.figure(figsize=cm2inch(12,6))

#porcen                 = np.arange(100,min(lista_tot_porc__),max(lista_tot_porc__))

plt.fill_between(lista_tot_porc__,acc_mean_0_b-acc_uncertanty_0_b*1/float(11),acc_mean_0_b+acc_uncertanty_0_b*1/float(12), where=acc_mean_0_b+acc_uncertanty_0_b*1/float(11) >= acc_mean_0_b-acc_uncertanty_0_b*1/float(11), facecolor='lightblue', interpolate=True)

plt.fill_between(lista_tot_porc__,acc_mean_1_b-acc_uncertanty_1_b*1/float(11),acc_mean_1_b+acc_uncertanty_1_b*1/float(12), where=acc_mean_1_b+acc_uncertanty_1_b*1/float(11) >= acc_mean_1_b-acc_uncertanty_1_b*1/float(11), facecolor='mistyrose', interpolate=True)

plt.fill_between(lista_tot_porc__,acc_mean_2_b-acc_uncertanty_2_b*1/float(11),acc_mean_2_b+acc_uncertanty_2_b*1/float(12), where=acc_mean_2_b+acc_uncertanty_2_b*1/float(11) >= acc_mean_2_b-acc_uncertanty_2_b*1/float(11), facecolor='pink', interpolate=True)

plt.fill_between(lista_tot_porc__,acc_mean_3_b-acc_uncertanty_3_b*1/float(11),acc_mean_3_b+acc_uncertanty_3_b*1/float(12), where=acc_mean_3_b+acc_uncertanty_3_b*1/float(11) >= acc_mean_3_b-acc_uncertanty_3_b*1/float(11), facecolor='lightgreen', interpolate=True)

plt.errorbar(lista_tot_porc__,acc_mean_0_b, xerr=0, yerr=acc_uncertanty_0_b*1/float(11),marker="o", markersize=2,fmt='o',color="blue",label="00 Level Distance")
plt.errorbar(lista_tot_porc__,acc_mean_1_b, xerr=0, yerr=acc_uncertanty_1_b*1/float(11),marker="o",markersize=2, fmt='o',color="red",label="11 Level Distance")
plt.errorbar(lista_tot_porc__,acc_mean_2_b, xerr=0, yerr=acc_uncertanty_2_b*1/float(11),marker="o",markersize=2, fmt='o',color="deeppink",label="10 Level Distance")
plt.errorbar(lista_tot_porc__,acc_mean_3_b, xerr=0, yerr=acc_uncertanty_3_b*1/float(11),marker="o",markersize=2, fmt='o',color="green",label="01 Level Distance")
plt.axhline(y=1, color='pink', linestyle='--')
plt.axhline(y=1.5, color='pink', linestyle='--')
plt.ylabel('Distance output-target',fontsize = 6)
plt.xlabel('% of positive connections removed',fontsize = 6)
plt.legend(fontsize=5,loc=2)
plt.ylim([-0.25,8.1])
# plt.xlim([-0,160])
#plt.ylim([-1,max(lista_distancia_all_3)+3])
plt.xticks(np.arange(0, max(lista_tot_porc__)+2, 2.0),fontsize = 5)
plt.yticks(np.arange(0,8.1,2),fontsize = 5)
plt.savefig("plots_paper/Test_man_distance_b"+".png",dpi=300, bbox_inches = 'tight')
#plt.show()
plt.close()


########### Fig c) ############

fig     = plt.figure(figsize=cm2inch(12,6))

#porcen                 = np.arange(100,min(lista_tot_porc__),max(lista_tot_porc__))

plt.fill_between(lista_tot_porc__,acc_mean_0_c-acc_uncertanty_0_c*1/float(11),acc_mean_0_c+acc_uncertanty_0_c*1/float(12), where=acc_mean_0_c+acc_uncertanty_0_c*1/float(11) >= acc_mean_0_c-acc_uncertanty_0_c*1/float(11), facecolor='lightblue', interpolate=True)

plt.fill_between(lista_tot_porc__,acc_mean_1_c-acc_uncertanty_1_c*1/float(11),acc_mean_1_c+acc_uncertanty_1_c*1/float(12), where=acc_mean_1_c+acc_uncertanty_1_c*1/float(11) >= acc_mean_1_c-acc_uncertanty_1_c*1/float(11), facecolor='mistyrose', interpolate=True)

plt.fill_between(lista_tot_porc__,acc_mean_2_c-acc_uncertanty_2_c*1/float(11),acc_mean_2_c+acc_uncertanty_2_c*1/float(12), where=acc_mean_2_c+acc_uncertanty_2_c*1/float(11) >= acc_mean_2_c-acc_uncertanty_2_c*1/float(11), facecolor='pink', interpolate=True)

plt.fill_between(lista_tot_porc__,acc_mean_3_c-acc_uncertanty_3_c*1/float(11),acc_mean_3_c+acc_uncertanty_3_c*1/float(12), where=acc_mean_3_c+acc_uncertanty_3_c*1/float(11) >= acc_mean_3_c-acc_uncertanty_3_c*1/float(11), facecolor='lightgreen', interpolate=True)

plt.errorbar(lista_tot_porc__,acc_mean_0_c, xerr=0, yerr=acc_uncertanty_0_c*1/float(11),marker="o", markersize=2,fmt='o',color="blue",label="00 Level Distance")
plt.errorbar(lista_tot_porc__,acc_mean_1_c, xerr=0, yerr=acc_uncertanty_1_c*1/float(11),marker="o",markersize=2, fmt='o',color="red",label="11 Level Distance")
plt.errorbar(lista_tot_porc__,acc_mean_2_c, xerr=0, yerr=acc_uncertanty_2_c*1/float(11),marker="o",markersize=2, fmt='o',color="deeppink",label="10 Level Distance")
plt.errorbar(lista_tot_porc__,acc_mean_3_c, xerr=0, yerr=acc_uncertanty_3_c*1/float(11),marker="o",markersize=2, fmt='o',color="green",label="01 Level Distance")
plt.axhline(y=1, color='pink', linestyle='--')
plt.axhline(y=1.5, color='pink', linestyle='--')
plt.ylabel('Distance output-target',fontsize = 6)
plt.xlabel('% of negative connections removed',fontsize = 6)
plt.legend(fontsize=5,loc=2)
plt.ylim([-0.25,8.1])
# plt.xlim([-0,160])
#plt.ylim([-1,max(lista_distancia_all_3)+3])
plt.xticks(np.arange(0, max(lista_tot_porc__)+2, 2.0),fontsize = 5)
plt.yticks(np.arange(0,8.1,2),fontsize = 5)
plt.savefig("plots_paper/Test_man_distance_c"+".png",dpi=300, bbox_inches = 'tight')
#plt.show()
plt.close()


########### Fig d) ############

fig     = plt.figure(figsize=cm2inch(12,6))

#porcen                 = np.arange(100,min(lista_tot_porc__),max(lista_tot_porc__))

plt.fill_between(lista_tot_porc__,acc_mean_0_d-acc_uncertanty_0_d*1/float(11),acc_mean_0_d+acc_uncertanty_0_d*1/float(12), where=acc_mean_0_d+acc_uncertanty_0_d*1/float(11) >= acc_mean_0_d-acc_uncertanty_0_d*1/float(11), facecolor='lightblue', interpolate=True)

plt.fill_between(lista_tot_porc__,acc_mean_1_d-acc_uncertanty_1_d*1/float(11),acc_mean_1_d+acc_uncertanty_1_d*1/float(12), where=acc_mean_1_d+acc_uncertanty_1_d*1/float(11) >= acc_mean_1_d-acc_uncertanty_1_d*1/float(11), facecolor='mistyrose', interpolate=True)

plt.fill_between(lista_tot_porc__,acc_mean_2_d-acc_uncertanty_2_d*1/float(11),acc_mean_2_d+acc_uncertanty_2_d*1/float(12), where=acc_mean_2_d+acc_uncertanty_2_d*1/float(11) >= acc_mean_2_d-acc_uncertanty_2_d*1/float(11), facecolor='pink', interpolate=True)

plt.fill_between(lista_tot_porc__,acc_mean_3_d-acc_uncertanty_3_d*1/float(11),acc_mean_3_d+acc_uncertanty_3_d*1/float(12), where=acc_mean_3_d+acc_uncertanty_3_d*1/float(11) >= acc_mean_3_d-acc_uncertanty_3_d*1/float(11), facecolor='lightgreen', interpolate=True)

plt.errorbar(lista_tot_porc__,acc_mean_0_d, xerr=0, yerr=acc_uncertanty_0_d*1/float(11),marker="o", markersize=2,fmt='o',color="blue",label="00 Level Distance")
plt.errorbar(lista_tot_porc__,acc_mean_1_d, xerr=0, yerr=acc_uncertanty_1_d*1/float(11),marker="o",markersize=2, fmt='o',color="red",label="11 Level Distance")
plt.errorbar(lista_tot_porc__,acc_mean_2_d, xerr=0, yerr=acc_uncertanty_2_d*1/float(11),marker="o",markersize=2, fmt='o',color="deeppink",label="10 Level Distance")
plt.errorbar(lista_tot_porc__,acc_mean_3_d, xerr=0, yerr=acc_uncertanty_3_d*1/float(11),marker="o",markersize=2, fmt='o',color="green",label="01 Level Distance")
plt.axhline(y=1, color='pink', linestyle='--')
plt.axhline(y=1.5, color='pink', linestyle='--')
plt.ylabel('Distance output-target',fontsize = 6)
plt.xlabel('% of a positive sclice of connections removed',fontsize = 6)
plt.legend(fontsize=5,loc=2)
plt.ylim([-0.25,8.1])
# plt.xlim([-0,160])
#plt.ylim([-1,max(lista_distancia_all_3)+3])
plt.xticks(np.arange(0, max(lista_tot_porc__)+2, 2.0),fontsize = 5)
plt.yticks(np.arange(0,8.1,2),fontsize = 5)
plt.savefig("plots_paper/Test_man_distance_d"+".png",dpi=300, bbox_inches = 'tight')
#plt.show()
plt.close()

########### Fig d) ############

fig     = plt.figure(figsize=cm2inch(12,6))

#porcen                 = np.arange(100,min(lista_tot_porc__),max(lista_tot_porc__))

plt.fill_between(lista_tot_porc__,acc_mean_0-acc_uncertanty_0*1/float(11),acc_mean_0+acc_uncertanty_0*1/float(12), where=acc_mean_0+acc_uncertanty_0*1/float(11) >= acc_mean_0-acc_uncertanty_0*1/float(11), facecolor='lightblue', interpolate=True)

plt.fill_between(lista_tot_porc__,acc_mean_1-acc_uncertanty_1*1/float(11),acc_mean_1+acc_uncertanty_1*1/float(12), where=acc_mean_1+acc_uncertanty_1*1/float(11) >= acc_mean_1-acc_uncertanty_1*1/float(11), facecolor='mistyrose', interpolate=True)

plt.fill_between(lista_tot_porc__,acc_mean_2-acc_uncertanty_2*1/float(11),acc_mean_2+acc_uncertanty_2*1/float(12), where=acc_mean_2+acc_uncertanty_2*1/float(11) >= acc_mean_2-acc_uncertanty_2*1/float(11), facecolor='pink', interpolate=True)

plt.fill_between(lista_tot_porc__,acc_mean_3-acc_uncertanty_3*1/float(11),acc_mean_3+acc_uncertanty_3*1/float(12), where=acc_mean_3+acc_uncertanty_3*1/float(11) >= acc_mean_3-acc_uncertanty_3*1/float(11), facecolor='lightgreen', interpolate=True)

plt.errorbar(lista_tot_porc__,acc_mean_0, xerr=0, yerr=acc_uncertanty_0*1/float(11),marker="o", markersize=2,fmt='o',color="blue",label="00 Level Distance")
plt.errorbar(lista_tot_porc__,acc_mean_1, xerr=0, yerr=acc_uncertanty_1*1/float(11),marker="o",markersize=2, fmt='o',color="red",label="11 Level Distance")
plt.errorbar(lista_tot_porc__,acc_mean_2, xerr=0, yerr=acc_uncertanty_2*1/float(11),marker="o",markersize=2, fmt='o',color="deeppink",label="10 Level Distance")
plt.errorbar(lista_tot_porc__,acc_mean_3, xerr=0, yerr=acc_uncertanty_3*1/float(11),marker="o",markersize=2, fmt='o',color="green",label="01 Level Distance")
plt.axhline(y=1, color='pink', linestyle='--')
plt.axhline(y=1.5, color='pink', linestyle='--')
plt.ylabel('Distance output-target',fontsize = 6)
#plt.xlabel('% of lowest connections removed',fontsize = 16)
#plt.xlabel('% of positive connections removed',fontsize = 16)
#plt.xlabel('% of negative connections removed',fontsize = 16)
#plt.xlabel('% of a positive sclice of connections removed',fontsize = 16)
plt.xlabel('% of a negative sclice of connections removed',fontsize = 6)
plt.legend(fontsize=5,loc=2)

plt.ylim([-0.25,8.1])
# plt.xlim([-0,160])
#plt.ylim([-1,max(lista_distancia_all_3)+3])
plt.xticks(np.arange(0, max(lista_tot_porc__)+2, 2.0),fontsize = 5)
plt.yticks(np.arange(0,8.1,2),fontsize = 5)
plt.savefig("plots_paper/Test_man_distance_e"+".png",dpi=300, bbox_inches = 'tight')
#plt.show()
plt.close()


