from pylab import *
from numpy import *

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
theta=arange(-1,1,0.2)
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
		error[k,i]=sum(abs(cr1))/N
		errr[k,i]=sum(abs(cr2))/N

theta_val=[]
zero_val=[]
theta_min=[]
for i in range(0,len(P)):
	xx= error[:,i]
	valmin= min(error[:,i])
	omin= min(errr[:,i])
	#print valmin
	index_min = argmin(error[:,i])
	theta_val.append(theta[index_min])
	print theta[index_min]
	theta_min.append(valmin)
print xx
plot(P,theta_val)
xlabel("no of patterns")
ylabel(r"$\theta_{min}$")
show()
	
		
