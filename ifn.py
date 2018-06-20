from pylab import *
from numpy import *

#Parameters
T = 3000 # total time to simulate (msec)
dt = 0.01
time = arange(0, T, dt)
V = zeros(len(time))
Io = zeros(len(time))
#Io[100001:200000] = 1.0

Cm = 1.0
gm = 0.1
taum =Cm/gm
Vm = -60.0
V_reset = -55.0
V_thresh = -50.0
print len(time)
V[0] = Vm
for i in range(0, len(time)-1):
	#Vn = 0.0
	Vn = V[i]+ (Vm*dt/taum) - (V[i]*dt/taum)+(Io[i]*dt/Cm)
	#print I
	if Vn >= V_thresh:
		V[i+1] = V_reset
		print "Hi"
	else:
		V[i+1] = Vn


plot(time,V)
show()		 
	 
