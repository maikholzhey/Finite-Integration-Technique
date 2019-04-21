# --- mh ---
"""
Example to calculate steady state in rectangular box resonator using FIT for time domain simulations
"""
# guardian protected import
# -------------------------
try:
	import numpy as np
	import scipy.sparse as sp
	import numpy.matlib as npm
	import copy
	import scipy.sparse.linalg
	import scipy.fftpack  
	import scipy.signal  
except ImportError:
	raise ImportError("open cmd, type: pip install scipy - then try again")
# -------
try:	
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
except ImportError:
	raise ImportError("open cmd, type pip install matplotlib - then try a gain")
# -------	
import time
import FIT

# ==========================
# -- MAIN ------------------
# ==========================
# create instance
a = FIT.FIT()
# clock the execution
startTime = time.time()
# geometry
nx = 31;
ny = 31;
nz = 11;
LxMax =(nx-1)*0.5; # 1/2 = 0 !integer division! WELCOME TO PYTHON
LyMax =(ny-1)*0.5;
LzMax =(nz-1)*0.5;
# boundary conditions
boundaryConditions = [1, 1, 1, 1, 1, 1] # all to PEC, 0 = PMC
# create DefaultModel
a.defaultModel(nx , ny , nz, -LxMax, -LyMax, -LzMax, LxMax, LyMax, LzMax,boundaryConditions)
# calculate maximum stable time step
Acc = a.Mepsi * a.Cs * a.Mmuei * a.C
val, vec =  sp.linalg.eigs(Acc,k=1, which='LM')
dtmax = 2/np.sqrt(np.real(val[0]))
print dtmax
# excitation signal
Nt = 1*1400;
tdt = np.linspace(0,Nt*dtmax,Nt)
f1 = 0.5e7
f2 = 1e7 
sig, npuls = a.gaussin(tdt, f1, f2)
# canonical Index of exitation
cI = int(np.ceil(a.nx/2) + (np.ceil(a.ny/2)-1) * a.nx  + 2 *a.np)
# leapfrog init
e = sp.csc_matrix((3*a.np,1))
h = sp.csc_matrix((3*a.np,1))
# if modification in current vector become expensive use lil_matrix instead
js = sp.csc_matrix((3*a.np,1))
# container to store field energy
w = copy.deepcopy(tdt) # container equal time vector
# Fieldmonitor Containers as attributes
a.etd = copy.deepcopy(tdt*0)
efreq = sp.csc_matrix((3*a.np,1))
hfreq = sp.csc_matrix((3*a.np,1))
em1 = sp.csc_matrix((3*a.np,1))
# the leaping frog - keep this section short for efficiency
for i in range(Nt):
	js[cI] = sig[i]
	h = h - dtmax * a.Mmuei * a.C * e
	e = e + dtmax * a.Mepsi * (a.Cs *h - js)
	# fieldEnergy
	w[i] = a.fieldEnergy(em1,e,h)
	em1 = e
	# fieldMonitorTD
	a.fieldProbe(e,a.etd,cI,i)
	if i > Nt//2:
		# fieldMonitor
		efreq = a.fieldMonitor(e, efreq, 7.1e6, len(tdt), i, dtmax)
		hfreq = a.fieldMonitor(h, hfreq, 7.1e6, len(tdt), i, dtmax)
	
# ---------------
# dft
f = np.linspace(f1,f2,250)
fsig = np.zeros(f.shape,dtype=complex)
dftM = a.dftMatrix(f, len(sig), dtmax) # keep out of UQ repetition (expensive!)
window = scipy.signal.blackman(len(a.etd))
fsig = a.selectiveDFT(dftM,a.etd*window)
# fft
xf, yf = a.fft(dtmax,a.etd)

# ------------------
# visualize results
ez = efreq.todense()[np.array([i for i in range(a.nx*a.ny)],dtype=int)+2*a.np]
eField = np.reshape(ez,(a.nx,a.ny))
# plot visualization
X, Y = np.meshgrid(a.xmesh,a.ymesh)
figureNumber = 1
title = 'ez field component; cutting-plane XY'
ax = a.plotField(X,Y,eField,figureNumber, title)
#ax.set_zlim(-15, 15)
ax.view_init(azim=0, elev=90)
plt.figure(figureNumber+1)
plt.plot(tdt,np.divide(w,np.max(w)))
plt.title('energy in resonator')
plt.figure(figureNumber+2)
plt.plot(tdt,a.etd)
plt.title('time signal in resonator')

plt.figure(figureNumber+3)
plt.plot(f,np.abs(fsig))
plt.title('frequency selective dft of time signal')
# plt.figure(figureNumber+4)
# plt.plot(xf,np.abs(yf[0:Nt//2]))
# plt.title('fft of time signal')
plt.figure(figureNumber+5)
plt.plot(tdt,sig)
plt.title('excitation signal in time')
plt.figure(figureNumber+6)
plt.plot(f,np.abs(a.selectiveDFT(dftM,sig)))
plt.title('spectrum of excitation signal')

# ------------------
# Field Distribution along an axis
ez = efreq.todense()[np.array([i for i in range(a.nx*a.ny)],dtype=int)+2*a.np]
eField = np.reshape(ez,(a.nx,a.ny))
plt.figure(figureNumber+7)
plt.plot(a.xmesh,np.transpose(eField[a.nx//2,:]))
plt.title('field distribution ez')

print time.time() - startTime
plt.show()