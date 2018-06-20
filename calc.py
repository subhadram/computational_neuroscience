from pylab import *
from numpy import *
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from matplotlib.pyplot import cm
from scipy.optimize import fsolve
def fff(W,s):
	fff = ((W*exp(W*(s - 0.5))*(1 +exp(W*(s - 0.5)))) - (W*exp(W*(s - 0.5))*exp(W*(s - 0.5))))/((1 +exp(W*(s - 0.5)))*(1 +exp(W*(s - 0.5))))
	print fff

fff(5.0,0.146)
fff(5.0,0.5)
fff(5.0,0.854)
