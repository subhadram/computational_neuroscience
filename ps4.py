from numpy import *
from pylab import *
matplotlib.rcParams.update({'font.size': 15})
x = np.arange(-7,7,0.1)
y1 = zeros(len(x))
y2 = zeros(len(x))
for i in range (0,len(x)):
	y1 = -x*sin(x)*sin(x) +	 x*x
	y2 = 2*x*sin(x)*sin(x) - 2*x*x

subplot(121)
plot(x,y1)
title(r"$-x sin^{2}(x) + x^2 > 0$")
xlabel(r"$x$")
ylabel(r"$f(x)$")
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
subplot(122)
plot(x,y2)
title(r"$2x sin^{2}(x) - 2x^2 < 0$")
xlabel(r"$x$")
ylabel(r"$f(x)$")

plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
show()

def fen(x,g,b): 
	f = x*log(x) + ((1-x)*log(1-x)) -1.0 -0.5*b*x*x + b*(1+g)*x/2.0
	return f
def dfen(x,g,b):
	f = (log(x)- log(1.0 - x) - b*x + (b*(1-g)*0.5))/(x*log(x) + ((1-x)*log(1-x)))
	return f
y = arange(0,1,0.01)
ffen = zeros(len(y))
dffen = zeros(len(y))
for i in range(0,len(y)):
	ffen[i]=fen(y[i],0.0,5.0)
	dffen[i]=dfen(y[i],.0,5.0)
subplot(121)
plot(y,ffen)
title(r"L(x) for $\gamma$ = 0")
xlabel("x")
ylabel("L(x)")  
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
subplot(122)
#print dffen	
plot(y,dffen,label= "dL(x)/dx" )
plot(y,0*y )
title(r"dL(x)/dx for $\gamma$ = 0")
xlabel("x")
ylabel("f(x)") 
legend(loc = 'best',fontsize = 15)  
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
show() 
ss  = []
gama = [] 
gamma = arange(-0.25,0.25,0.01)
for j in range(0,len(gamma)):
	for i in range(0,len(y)):
		ffen[i]=fen(y[i],gamma[j],5.0)
		dffen[i]=dfen(y[i],gamma[j],5.0)
	slope,intercept=polyfit(y, dffen, 1)
	print intercept
print ss
print gama
plot(gama,ss)
xlabel(r"$\gamma$")
ylabel("stable points")
show()
