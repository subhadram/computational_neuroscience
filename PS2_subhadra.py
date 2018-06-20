from pylab import *
from numpy import *
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from matplotlib.pyplot import cm
from scipy.optimize import fsolve

matplotlib.rcParams.update({'font.size': 15})

#initiate vectors
s = arange(0,1,0.001)
W = arange(3,6,1) 
f = zeros((len(s),len(W)))
ss= zeros(len(s))
 
#nullclines for different values of W

for j in range(len(W)):
	for i in range(len(s)):
		f[i,j] = exp(W[j]*(s[i] - 0.5))/(1.0+exp(W[j]*(s[i] - 0.5)))
	plot(s, f[:,j],label = "W = %s"%W[j])
	#idx = argwhere(diff(np.sign(f[i,j] - s[i])) != 0).reshape(-1) + 0
	#plot(s[idx], f[idx,j], 'ro')	
	
plot(s,s, label= "s = s",color ="black", linewidth = 2.0)
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("s")
ylabel("f(s)")
legend(loc = 'best',fontsize = 15 )
show()

#1b.for w<w* 
T = 1000.0
dt = 10
time = arange(0, T, dt) 
S = zeros(len(time))
SS = zeros(len(time))
S2 = zeros(len(time))
W = 3.0
tau = 10.0
S[0] = 0.61
SS[0] = 0.29
for i in range(len(time)-1):
	S[i+1] = S[i] + (-S[i] + exp(W*(S[i] - 0.5))/(1.0+exp(W*(S[i] - 0.5))))/tau *dt
	SS[i+1] = SS[i] + (-SS[i] + exp(W*(SS[i] - 0.5))/(1.0+exp(W*(SS[i] - 0.5))))/tau *dt
subplot(121)
plot(time,S,label= "S0=0.61")
plot(time,SS,label= "S0=0.29")
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("time msec")
ylabel("s")
title(r"$W < W^*$")
legend(loc = 'best',fontsize = 15 )
#show()
#1b.for w>w*
S = zeros(len(time))
SS = zeros(len(time))
S2 = zeros(len(time))
W = 5.0
tau = 10.0
S[0] = 0.61
SS[0] = 0.29

for i in range(len(time)-1):
	S[i+1] = S[i] + (-S[i] + exp(W*(S[i] - 0.5))/(1.0+exp(W*(S[i] - 0.5))))/tau *dt
	SS[i+1] = SS[i] + (-SS[i] + exp(W*(SS[i] - 0.5))/(1.0+exp(W*(SS[i] - 0.5))))/tau *dt
subplot(122)
plot(time,S,label= "S0=0.61")
plot(time,SS,label= "S0=0.29")
xlabel("time msec")
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
ylabel("s")
title(r"$W > W^*$")
legend(loc = 'best',fontsize = 15 )
show()

#1c. Shift in the curve for 3 db values
s = arange(0,1,0.001)
f= zeros(len(s))
fb= zeros(len(s))
ffb= zeros(len(s))
for i in range(len(s)):
	f[i] = exp(W*(s[i] - 0.5))/(1.0+exp(W*(s[i] - 0.5)))
	fb[i] = exp(W*(s[i] - 0.5)+1.0)/(1.0+exp(W*(s[i] - 0.5)+1.0))
	ffb[i] = exp(W*(s[i] - 0.5)-1.0)/(1.0+exp(W*(s[i] - 0.5)-1.0))

plot(s,s, label="s=s",color="black", linewidth = 2.0 )
plot(s,f, label="db = 0.0") 
plot(s,fb, label="db = 1.0") 
plot(s,ffb, label="db = -1.0") 
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("s")
ylabel("f(s)")
legend(loc = 'best',fontsize = 15 )
show()

#1c hystersis and plots.
W = 4.1
db = arange(-0.2,0.2,0.01)
S = zeros((len(time),len(db)))
SS = zeros((len(time),len(db)))
S[0,:] = 0.3
SS[0,:] = 0.9

color=cm.rainbow(np.linspace(0,1,len(db)))
subplot(121)
for j in range(len(db)):
	for i in range(len(time)-1):
		S[i+1,j] = S[i,j] + (-S[i,j] + exp(W*(S[i,j] - 0.5)+db[j])/(1.0+exp(W*(S[i,j] - 0.5)+db[j])))/tau *dt
	for j,c in zip(range(len(db)),color):
		plot(time,S[:,j],label = "db = %s"%j, c=c) 
title("S0 = lower state")	
xlabel("time msec")
ylabel("s")
#show()
subplot(122)
for k in range(len(db)):
	for i in range(len(time)-1):
		SS[i+1,k] = SS[i,k] + (-SS[i,k] + exp(W*(SS[i,k] - 0.5)+db[k])/(1.0+exp(W*(SS[i,k] - 0.5)+db[k])))/tau *dt
	for k,c in zip(range(len(db)),color):
		plot(time,SS[:,k],label = "db = %s"%j, c=c)
title("S0 = higher state")
xlabel("time msec")
ylabel("s")
#legend(loc = 'best',fontsize = 4 )
show()
SS1 = S[-1,:]
SS2 = SS[-1,:]


plot(db,SS1,label="S0 = lower state" )
plot(db,SS2,label="S0 = higher state")
xlabel(r"$\delta b$")
ylabel("Current state s")
title("Plot of Hysteresis")
legend(loc = 'best',fontsize = 15 )
show()

"""
dB = zeros(len(time))
dB[20] = 0.1
dB[40] = 0.1
Db = zeros(len(time))
dB[20] = 0.4
dB[40] = 0.4
S = zeros(len(time))
SS = zeros(len(time))
S[0] = 0.3
SS[0] = 0.9
for i in range(len(time)-1):
		S[i+1] = S[i] + (-S[i] + exp(W*(SS[i] - 0.5)+dB[i])/(1.0+exp(W*(S[i] - 0.5)+dB[i])))/tau *dt
		SS[i+1] = SS[i] + (-SS[i] + exp(W*(SS[i] - 0.5)+Db[i])/(1.0+exp(W*(SS[i] - 0.5)+Db[i])))/tau *dt
plot(time,SS)
plot(time,S)
show()
"""
#2

s = arange(0,1,0.001)
W = arange(3,6,1) 
f = zeros((len(s),len(W)))
ss= zeros(len(s))
 
#nullclines for different values of W

for j in range(len(W)):
	for i in range(len(s)):
		f[i,j] = exp(W[j]*(s[i] - 0.5))/(1.0+exp(W[j]*(s[i] - 0.5)))
		#if (f[i,j] - s[i]==0.0):
			#print W[j], s.all(i) 
	plot(s, f[:,j],label = "W = %s"%W[j])
	#idx = argwhere(diff(np.sign(f[i,j] - s[i])) != 0).reshape(-1) + 0
	#plot(s[idx], f[idx,j], 'ro')	

plot(s,s)
show()
def fff(W,s):
	fff = ((W*exp(W*(s - 0.5))*(1 +exp(W*(s - 0.5)))) - (W*exp(W*(s - 0.5))*exp(W*(s - 0.5))))/((1 +exp(W*(s - 0.5)))*(1 +exp(W*(s - 0.5))))
	print fff

fff(5.0,0.146)
fff(5.0,0.5)
fff(5.0,0.854)

#3.0
N1 = zeros(len(time))
N2 = zeros(len(time))
bb = zeros(len(time))
bb[10]=1.0
bb[20]=2.0
bb[30]=3.0
bb[60]=2.0
bb[80]=1.0

N1[0]=0.1
N2[0]=0.9
taus = 50.0
subplot(221)
ww = -1.0
for i in range(len(time)-1):
	N1[i+1] = (-N1[i] + (ww*N2[i] + bb[i]))/taus*dt + N1[i]
	N2[i+1] = (-N2[i] +(ww*N1[i] + bb[i]))/taus*dt + N2[i]

#plot(time,bb)
plot(time,N1,label="Neuron 1")
plot(time,N2,label="Neuron 2")
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("time (msec)")
ylabel("synaptic activation")
plot(time,bb,label="Input")
title("Integrator")
legend(loc = 'best',fontsize = 15 )
#show()
N1 = zeros(len(time))
N2 = zeros(len(time))
bb = zeros(len(time))
bb[10]=1.0
bb[20]=2.0
bb[30]=3.0
bb[60]=2.0
bb[80]=1.0

N1[0]=0.1
N2[0]=0.9
taus = 50.0
subplot(222)
ww = -0.990
for i in range(len(time)-1):
	N1[i+1] = (-N1[i] + (ww*N2[i] + bb[i]))/taus*dt + N1[i]
	N2[i+1] = (-N2[i] +(ww*N1[i] + bb[i]))/taus*dt + N2[i]

#plot(time,bb)
plot(time,N1,label="Neuron 1")
plot(time,N2,label="Neuron 2")
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("time (msec)")
ylabel("synaptic activation")
#plot(time,bb,label="Input")
title("1% mistune")


N1 = zeros(len(time))
N2 = zeros(len(time))
bb = zeros(len(time))
bb[10]=1.0
bb[20]=2.0
bb[30]=3.0
bb[60]=2.0
bb[80]=1.0

N1[0]=0.1
N2[0]=0.9
taus = 50.0
subplot(223)
ww = -0.980
for i in range(len(time)-1):
	N1[i+1] = (-N1[i] + (ww*N2[i] + bb[i]))/taus*dt + N1[i]
	N2[i+1] = (-N2[i] +(ww*N1[i] + bb[i]))/taus*dt + N2[i]

#plot(time,bb)
plot(time,N1,label="Neuron 1")
plot(time,N2,label="Neuron 2")
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("time (msec)")
ylabel("synaptic activation")
#plot(time,bb,label="Input")
title("2% mistune")

N1 = zeros(len(time))
N2 = zeros(len(time))
bb = zeros(len(time))
bb[10]=1.0
bb[20]=2.0
bb[30]=3.0
bb[60]=2.0
bb[80]=1.0

N1[0]=0.1
N2[0]=0.9
taus = 50.0
subplot(224)
ww = -0.950
for i in range(len(time)-1):
	N1[i+1] = (-N1[i] + (ww*N2[i] + bb[i]))/taus*dt + N1[i]
	N2[i+1] = (-N2[i] +(ww*N1[i] + bb[i]))/taus*dt + N2[i]

#plot(time,bb)
plot(time,N1,label="Neuron 1")
plot(time,N2,label="Neuron 2")
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("time (msec)")
ylabel("synaptic activation")
#plot(time,bb,label="Input")
title("5% mistune")
show()
############################################################################################################################
N1 = zeros(len(time))
N2 = zeros(len(time))
bb = zeros(len(time))
N1[0]=0.1
N2[0]=0.9
taus = 50.0
subplot(221)
ww = -1.0
for i in range(len(time)-1):
	N1[i+1] = (-N1[i] + (ww*N2[i] + bb[i]))/taus*dt + N1[i]
	N2[i+1] = (-N2[i] +(ww*N1[i] + bb[i]))/taus*dt + N2[i]

#plot(time,bb)
plot(time,N1,label="Neuron 1")
plot(time,N2,label="Neuron 2")
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("time (msec)")
ylabel("synaptic activation")
#plot(time,bb,label="Input")
title("Integrator")
legend(loc = 'best',fontsize = 15 )
#show()
N1 = zeros(len(time))
N2 = zeros(len(time))
bb = ones(len(time))

N1[0]=0.1
N2[0]=0.9
taus = 50.0
subplot(222)
ww = -0.990
for i in range(len(time)-1):
	N1[i+1] = (-N1[i] + (ww*N2[i] + bb[i]))/taus*dt + N1[i]
	N2[i+1] = (-N2[i] +(ww*N1[i] + bb[i]))/taus*dt + N2[i]

#plot(time,bb)
plot(time,N1,label="Neuron 1")
plot(time,N2,label="Neuron 2")
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("time (msec)")
ylabel("synaptic activation")
#plot(time,bb,label="Input")
title("1% mistune")


N1 = zeros(len(time))
N2 = zeros(len(time))
bb = ones(len(time))

N1[0]=0.1
N2[0]=0.9
taus = 50.0
subplot(223)
ww = -0.980
for i in range(len(time)-1):
	N1[i+1] = (-N1[i] + (ww*N2[i] + bb[i]))/taus*dt + N1[i]
	N2[i+1] = (-N2[i] +(ww*N1[i] + bb[i]))/taus*dt + N2[i]

#plot(time,bb)
plot(time,N1,label="Neuron 1")
plot(time,N2,label="Neuron 2")
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("time (msec)")
ylabel("synaptic activation")
#plot(time,bb,label="Input")
title("2% mistune")

N1 = zeros(len(time))
N2 = zeros(len(time))
bb = ones(len(time))

N1[0]=0.1
N2[0]=0.9
taus = 50.0
subplot(224)
ww = -0.950
for i in range(len(time)-1):
	N1[i+1] = (-N1[i] + (ww*N2[i] + bb[i]))/taus*dt + N1[i]
	N2[i+1] = (-N2[i] +(ww*N1[i] + bb[i]))/taus*dt + N2[i]

#plot(time,bb)
plot(time,N1,label="Neuron 1")
plot(time,N2,label="Neuron 2")
plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)
xlabel("time (msec)")
ylabel("synaptic activation")
#plot(time,bb,label="Input")
title("5% mistune")
show()

