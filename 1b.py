from pylab import *
from numpy import *
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

matplotlib.rcParams.update({'font.size': 15})


#parameters
gm =0.1*0.001 #0.1 mS/cm2
cm =1.0*0.000001 # 1 muF/cm2
rm =1.0/gm 
Vm =-60.0 #mV
tm =cm/gm 
Vthresh=-50.0
Vreset =-55.0
Ve =0.0
 
#intialize vectors
I0 = arange(0.00001, 0.01, 0.00001)#current Amperes
vv = zeros(len(I0))#rate (Hz)
R = zeros(len(I0))#rate taylor approximation 

#1B: rate of firing for varying the current
for j in range(0, len(I0)):
	if I0[j]/gm > (Vthresh-Vm):
		vv[j] = 1.0/(tm*log((Vm + rm*I0[j] - Vreset)/(Vm+ rm*I0[j] - Vthresh))) 
	else:
		vv[j] = 0.0 
	
plot(I0,vv,linewidth=2)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("Current (A)")
ylabel(r'$\nu$ (Hz)')
title("f-I curve for the leaky integrate-and-fire neuron")
show()

#taylor linear approximation
for k in range(0, len(I0)):
    R[k]= (1.0/tm)*(Vm-Vthresh + (I0[k]*rm))/ ( Vthresh-Vreset);

plot(I0,vv,linewidth=2,label="Calculated rate" )
plot(I0,R,linewidth=2,label="Linear approximation of Taylor series")
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("Current (A)")
ylabel(r'$\nu$ (Hz)')
title("f-I curve & linear approximation of rate for the LIF neuron")
legend(loc = 'best',fontsize = 15)
show()
#synaptic activation.
taus = [0.0100,0.0500,0.1000] 
T = 3.0 # total time to simulate (msec)
dt = 0.0001
time = arange(0, T, dt)
V = zeros(len(time))
Io = zeros(len(time))
Io[10001:20000] = 0.0011
V[0] = Vreset
s =zeros((len(time),3))
spikes = zeros(len(time))
for n in range(0,len(taus)):
	for t in range(0,len(time)-1):
		s[t+1,n]=s[t,n] + -(s[t,n]*dt)/taus[n] + (spikes[t]) 
        #Vn= V[t] + ((Vm-V[t])*dt)/tm + ((Io[t]/cm)*dt)
		if V[t]>=Ve:
			V[t+1]=Vreset
		if V[t]<Vthresh:
			V[t+1]= V[t] + ((Vm-V[t])*dt)/tm + ((Io[t]/cm)*dt)
		if (V[t]>= Vthresh) and (V[t]<Ve):
			V[t+1]=Ve
			spikes[t+1]=1
			

#print V
subplot(221)
plot(time,V)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
ylabel("Voltage (mV)")
ylim(-70,10)
#xlabel('time (sec)')
title("Voltage")
subplot(222)
plot(time,s[:,0])
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
ylabel("Synaptic activation s")
title(r'$\tau_s=10 msec$')
ylim(0,7.5)
#xlabel('time (sec)')
subplot(223)
plot(time,s[:,1])
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
ylabel("Synaptic activation s")
ylim(0,7.5)
xlabel('time (sec)')
title(r'$\tau_s=50 msec$')
subplot(224)
plot(time,s[:,2])
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
ylabel("Synaptic activation s")
ylim(0,7.5)
title(r'$\tau_s=10 msec$')
xlabel('time (sec)')
show()

#rate based synaptic activation
rate = zeros(len(time))
rate[10001:20000] = 60.0
sr =zeros((len(time),3))
for j in range(0,len(taus)):
    for t in range(0,len(time)-1):
        sr[t+1,j] = sr[t,j]  - (sr[t,j]*dt)/taus[j] + dt*(rate[t])

subplot(221)
plot(time,V)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
ylabel("Voltage (mV)")
ylim(-70,10)
#xlabel('time (sec)')
title("Voltage")

subplot(222)
plot(time,s[:,0],label="Simulated synaptic activation")
plot(time,sr[:,0],linewidth=3.0,label="Rate-based synaptic activation")
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
ylabel("Synaptic activation s" )

title(r'$\tau_s=10 msec$')
legend(loc = 'best',fontsize = 15)
ylim(0,7.5)
#xlabel('time (sec)')
subplot(223)
plot(time,s[:,1])
plot(time,sr[:,1],linewidth=3.0)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
ylabel("Synaptic activation s" )
ylim(0,7.5)
xlabel('time (sec)')
title(r'$\tau_s=50 msec$')
subplot(224)
plot(time,s[:,2])
plot(time,sr[:,2],linewidth=3.0)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
ylabel("Synaptic activation s")
ylim(0,7.5)
title(r'$\tau_s=10 msec$')

xlabel('time (sec)')
show()


