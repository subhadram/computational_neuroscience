from pylab import *
from numpy import *
matplotlib.rcParams.update({'font.size': 15})

T = 1000.0
dt = 0.10
time = arange(0, T, dt)
"""
x1 = zeros(len(time))
x2 = zeros(len(time))
z1 = zeros(len(time))
z2 = zeros(len(time))
"""
x1 = zeros((len(time),2))
x2 = zeros((len(time),2))
z1 = zeros((len(time),2))
z2 = zeros((len(time),2))

b1 = 5.0
b2 = 1.25
g = [2.0,10]
b = 2.0
tau =100.0
for j in range(0,len(g)):
	for i in range(0,len(time)-1):
		xx1 = b1-b*x2[i,j] - g[j]*z1[i,j]
		if xx1 < 0.0:
			xx1 = 0.0
		xx2 = b2-b*x1[i,j] - g[j]*z2[i,j]
		if xx2 < 0.0:
			xx2 = 0.0
		x1[i+1,j] =  x1[i,j] + (-x1[i,j] + xx1)*dt
		x2[i+1,j] =  x2[i,j] + (-x2[i,j] + xx2)*dt
		z1[i+1,j] =  z1[i,j] + (x1[i,j] - z1[i,j])*dt/tau
		z2[i+1,j] =  z2[i,j] + (x2[i,j] - z2[i,j])*dt/tau

plot(time,x1[:,0],label=r'$x_1$'  )
plot(time,x2[:,0],label=r'$x_2$')
ylabel(r'$x$')
xlabel("time msec")
legend(loc = 'best',fontsize = 15 )
show()
plot(time,x1[:,1],label=r'$x_1$'  )
plot(time,x2[:,1],label=r'$x_2$')
ylabel(r'$x$')
xlabel("time msec")
legend(loc = 'best',fontsize = 15 )
show()
