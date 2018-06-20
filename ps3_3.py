from pylab import *
from numpy import *

matplotlib.rcParams.update({'font.size': 15})
N = 1000
P = [100,140,200,500]
#crosstalk = zeros((N,len(P)))

fig,ax= subplots(2,2)
#axs = axs.ravel()
#E = zeros((N,P))
for i in range(len(P)):
	num_pat= P[i]
	memory = around(random.uniform(0,1,(N,num_pat)))
	observed_memory=memory[:,-1]
	tmemory= memory.transpose()
	wij = (dot(memory[:,0:num_pat-1]-0.5,tmemory[0:num_pat-1,:]-0.5) -(num_pat-1)*identity(N))/N
	crosstalk=dot(wij,observed_memory)
	subplot(2,2,i+1)
	#xlim(-2,2)
	n,bins,patches= hist(crosstalk, 50,normed=0)
	axvline(1,color= "red")
	axvline(-1,color= "red")
	ylabel("count")
	xlabel(r"$C_i^\nu$")
	title('Patterns = %s'%P[i])

print memory




show()
	

