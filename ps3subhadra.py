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



matplotlib.rcParams.update({'font.size': 15})
N = 1000
P = [100,140,200,500]
#crosstalk = zeros((N,len(P)))

fig,ax= subplots(2,2)
#axs = axs.ravel()
#E = zeros((N,P))
for i in range(len(P)):
	num_pat= P[i]
	memory = 2*around(random.uniform(0,1,(N,num_pat))) - 1
	observed_memory=memory[:,-1]
	tmemory= memory.transpose()
	wij = (dot(memory[:,0:num_pat-1],tmemory[0:num_pat-1,:]) -(num_pat-1)*identity(N))/N
	crosstalk=dot(wij,observed_memory)
	subplot(2,2,i+1)
	#xlim(-2,2)
	n,bins,patches= hist(crosstalk, 50,normed=0)
	axvline(1,color= "red")
	axvline(-1,color= "red")
	ylabel("count")
	xlabel(r"$C_i^\nu$")
	title('Patterns = %s'%P[i])

#print wij

show()


matplotlib.rcParams.update({'font.size': 15})
N = 1000
f=0.05;
P=[1000,3050,5000]
theta=[0.5]
crosstalk = zeros((N,len(P)))

fig,ax= subplots(1,3)
for i in range(len(P)):
	num_pat= P[i]
	memory = np.random.binomial(1, f,(N,num_pat) )
	#memory = (random.uniform(0,1,(N,num_pat)))
	#memory[:,:] = [0 if ele < f else 1  for ele in memory ]
	observed_memory=memory[:,-1]
	tmemory= memory.transpose()
	subplot(1,3,i+1)
	wij = dot((memory[:,0:num_pat-1]-f),(tmemory[0:num_pat-1,:]-f))/(N*f*(1-f))
	fill_diagonal(wij, 0)
	#print wij
	#for k in range(1,len(theta)):
	crosstalk[:,i]=dot(wij,observed_memory)-0.0
	n,bins,patches= hist(crosstalk[:,i], 50,normed=0)
	xlim(-2,2)
	axvline(1,color= "red")
	axvline(-1,color= "red")
	ylabel("count")
	xlabel(r"$C_i^\nu$")
	title('Patterns = %s'%P[i])
		

show()
	
		
P=arange(1000,5000,1000)
theta=arange(-1,1,0.5)
crosstalk = zeros((N,len(theta),len(P)))
crsstalk = zeros((N,len(theta),len(P)))
error = zeros((len(theta),len(P)))
errr = zeros((len(theta),len(P)))
#fig,ax= subplots(1,3)
for i in range(len(P)):
	num_pat= P[i]
	memory = np.random.binomial(1, f,(N,num_pat) )
	#memory = (random.uniform(0,1,(N,num_pat)))
	#memory[:,:] = [0 if ele < f else 1  for ele in memory ]
	observed_memory=memory[:,-1]
	tmemory= memory.transpose()
	#subplot(1,3,i+1)
	wij = dot((memory[:,0:num_pat-1]-f),(tmemory[0:num_pat-1,:]-f))/(N*f*(1-f))
	fill_diagonal(wij, 0)
	#print wij
	for k in range(1,len(theta)):
		crosstalk[:,k,i]=dot(wij,observed_memory)-theta[k]
		crsstalk[:,k,i]=dot(wij,observed_memory)-0
		cr1=crosstalk[where(1 <= abs(crosstalk[:,k,i]))]
		cr2=crsstalk[where(1 <= abs(crsstalk[:,k,i]))]
		error[k,i]=sum(cr1)/N
		errr[k,i]=sum(cr2)/N

theta_val=[]
zero_val=[]
theta_min=[]
for i in range(0,len(P)):
	valmin= min(error[:,i])
	omin= min(errr[:,i])
	print valmin
	index_min = argmin(error[:,i])
	theta_val.append(theta[index_min])
	print theta
	theta_min.append(valmin)

plot(P,theta_val)
xlabel("no of patterns")
ylabel(r"$\theta_{min}$")
show()
	
		
