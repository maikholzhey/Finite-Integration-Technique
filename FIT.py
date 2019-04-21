# --- mh ---
# FIT in python
"""
This module is a generic PYTHON implementation of the essential functions for doing computations using the Finite-Integration-Technique

.. warning::  	1/2 = 0 !integer division! 

Use instead:
	>>> 1./2 = 0.5
	
This module can be imported using
	>>> import FIT
	
The main function calculating an example does only execute when module runs in main frame of the execution stack, which is not the case for `import` statements.
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
#--------
try:	
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
except ImportError:
	raise ImportError("open cmd, type pip install matplotlib - then try a gain")
#--------	
# try:
	# from drawnow import *
# except ImportError:
	# raise ImportError("open cmd, type pip install drawnow - then try again")
	
import time

# ========================
class FIT():
	"""
	This class-object represents the basic framework for any FIT calculations
	"""
	def __init__(self):
		""" 
		This function initializes the basics of a FIT object 
		
		:param c0: vacuum light speed
		:type c0: integer
		
		:param mue0: magnetic permeability in vacuum
		:type mue0: float
		
		:param eps0: electric permeability in vacuum
		:type eps0: float
		
		"""
		# constants
		self.c0 = 299792458;
		self.mue0 =4e-7*np.pi;
		self.eps0 = 1./(self.c0**2*self.mue0);


	def setGeometry(self, nx , ny , nz, LxMin, LyMin, LzMin, LxMax, LyMax, LzMax):
		"""
		This functions sets the basic geometrical shape representation. 
		
		:param N_: number of points in _
		:type N_: integer
		
		:param L_Min: total length in _ (minimum)
		:type L_Min: float 
		
		:param L_Max: total length in _ (maximum)
		:type L_Max: float 
		
		All of the above are then accessible via attribute calling convention 
		
		Example usage
			>>> fitObject.nx
		"""
		self.nx = nx
		self.ny = ny
		self.nz = nz
		self.LxMin = LxMin
		self.LyMin = LyMin
		self.LzMin = LzMin
		self.LxMax = LxMax
		self.LyMax = LyMax
		self.LzMax = LzMax 
		# algebraic dimension
		self.np = nx * ny * nz
		
		
	def makeMesh(self):
		"""
		Generates meshgrid for cartesian grid
		
		Sets the object's attributes:
		
		:ivar xmesh: xmesh as linspace spanning L_Min-L_Max in n points
		:ivar ymesh: ymesh as linspace spanning L_Min-L_Max in n points
		:ivar zmesh: zmesh as linspace spanning L_Min-L_Max in n points
		"""
		self.xmesh = np.linspace(self.LxMin,self.LxMax,self.nx)
		self.ymesh = np.linspace(self.LyMin,self.LyMax,self.ny)
		self.zmesh = np.linspace(self.LzMin,self.LzMax,self.nz)
		
		
	def makeMeshBOR(self):
		"""
		Generates meshgrid for rz grid
		
		.. warning:: If called, will raise NotImplementedError
		"""
		raise NotImplementedError
		
	def createP(self):
		"""
		FIT elementary operators
		
		Initializes the attributes:
		
		:ivar Px: Px as FIT operator
		:ivar Py: Py as FIT operator
		:ivar Pz: Pz as FIT operator
		"""
		xp = np.ones(self.np,dtype = int)
		self.Px = sp.spdiags([xp*(-1), xp],[0, 1], self.np, self.np)
		self.Py = sp.spdiags([xp*(-1), xp],[0, self.nx], self.np, self.np)
		self.Pz = sp.spdiags([xp*(-1), xp],[0, self.nx*self.ny], self.np, self.np)
		

	def createPBOR(self):
		"""
		FIT elementary operators: rz code
		
		.. warning:: If called, will raise NotImplementedError
		"""	
		raise NotImplementedError
		
	def createOperators(self):
		"""
		FIT operators
		
		Intializes the attributes:
		
		:ivar G: gradient operator 
		:ivar S: divergence operator 
		:ivar C: curl operator 
		:ivar Cs: curl operator on dual grid 
		"""
		# create elementary operators
		self.createP()
		# Gradient G
		self.G = sp.vstack([self.Px,self.Py,self.Pz])
		# Divergence S 
		self.S = -1*np.transpose(self.G)
		# curl C
		zs = sp.spdiags(np.zeros(self.np),[0],self.np,self.np);
		C1 = sp.hstack([zs, (-1)*self.Pz, self.Py]) 
		C2 = sp.hstack([self.Pz, zs, (-1)*self.Px])
		C3 = sp.hstack([(-1)*self.Py ,self.Px, zs])
		self.C = sp.vstack([C1,C2,C3])
		# Cs dual
		self.Cs = np.transpose(self.C)
		
		
	def createOperatorsBOR(self):
		"""
		FIT operators: rz code
		
		.. warning:: If called, will raise NotImplementedError
		"""
		raise NotImplementedError	
		
	def geomMatrices(self):
		"""
		Geometrical data for FIT to build material matrices
		
		It initializes the attributes:
		
		:ivar ds: primary edges
		:ivar da: primary facets
		:ivar dsd: dual edges
		:ivar dad: dual facets
		"""
		# xmesh, ymesh, zmesh init locals
		xm = self.xmesh
		ym = self.ymesh
		zm = self.zmesh
		
		# matlab to pyhton indexing
		# [:-1] array to end-1 start=[0] end=[-1]
		# (1:end-1) = [:-1]
		# (2:end) = [1:]
		dx = np.concatenate((xm[1:]-xm[:-1], 0),axis=None)
		dy = np.concatenate((ym[1:]-ym[:-1], 0),axis=None)
		dz = np.concatenate((zm[1:]-zm[:-1], 0),axis=None)
		
		xd  = (xm[:-1]+xm[1:])/2;   # length: nx-1
		dxd = np.concatenate( (xd[0]-xm[0],\
				xd[1:]-xd[:-1], \
				xm[-1]-xd[-1]), axis=None)# length nx

		yd  = (ym[:-1]+ym[1:])/2
		dyd = np.concatenate((yd[0]-ym[0] ,\
				yd[1:]-yd[:-1],\
				ym[-1]-yd[-1]), axis=None)
				
		if self.nz != 1:
			zd  = (zm[:-1]+zm[1:])/2;
			if self.nz==2:
				dzd = np.concatenate(( zd[0]-zm[0],\
					zm[-1]-zd[-1]), axis=None)
			else:
				dzd = np.concatenate((zd[0]-zm[0],\
					zd[1:]-zd[:-1],\
					zm[-1]-zd[-1]), axis=None)
			
		else:
			# special for 2D-Problems: set dz = 1
			zd = zm;
			dz = [1];
			dzd = [1];
		
		# dimension conventions of the matlab origin
		dx = np.matrix(dx).T
		dy = np.matrix(dy).T
		dz = np.matrix(dz).T
		dxd = np.matrix(dxd).T
		dyd = np.matrix(dyd).T
		dzd = np.matrix(dzd).T
		
		# repmat matlab to python: dimensions n,m need (n,m) + Fortran like indexing order: , order='F')
		
		### primary grid		
		dsx = npm.repmat(dx,self.ny*self.nz,1 );
		dsy = np.reshape(npm.repmat(dy.T,self.nx,self.nz),(self.nx*self.ny*self.nz,1), order='F');
		dsz = np.reshape(npm.repmat(dz.T,self.nx*self.ny,1 ),(self.nx*self.ny*self.nz,1), order='F');
		
		self.ds = np.concatenate((dsx, dsy, dsz), axis=0)
		self.da = np.concatenate((np.multiply(dsy,dsz), np.multiply(dsx,dsz), np.multiply(dsx,dsy)), axis=0)
		### dual grid
		dsdx = npm.repmat(dxd,self.ny*self.nz,1);
		dsdy = np.reshape(npm.repmat(dyd.T,self.nx,self.nz),(self.nx*self.ny*self.nz,1), order='F');
		dsdz = np.reshape(npm.repmat(dzd.T,self.nx*self.ny,1),(self.nx*self.ny*self.nz,1), order='F');
		
		self.dsd = np.concatenate((dsdx, dsdy, dsdz), axis=0);
		self.dad = np.concatenate((np.multiply(dsdy,dsdz),np.multiply(dsdx,dsdz), np.multiply(dsdx,dsdy)), axis=0);
	
	def geomMatricesBOR(self):
		"""
		Geometrical data for FIT: rz code
		
		.. warning:: If called, will raise NotImplementedError
		"""
		raise NotImplementedError
		
	def nullinv(self, a):
		"""
		Matrix (vector) inversion by replacing diagonal entries with its reciproke value v = 1/v
		
		can handle some python sparse data, attribute a.shape is a necessity
		
		:param a: input to be inverted
		:type a: array like
		
		:return: pseudo inverse
		:rtype: array like
		"""
		b=np.zeros(a.shape)
		tmpI = np.nonzero(a)
		b[tmpI] = 1. / a[tmpI]
		return b

	def nulldiv(self, a, b):
		"""
		Calculates a/b elementwise if b contains 0
		
		Division doing multiplication with nullinv essentially
		
		can handle some python sparse data, attribute a.shape is a necessity
		
		:param a: numerator
		:type a: array like
		:param b: denominator
		:type b: array like
		
		:return: pseudo divison
		:rtype: array like
		"""
		return np.multiply(a,self.nullinv(b))
		
	def boundaryIndices(self, boundaryConditions):
		"""
		Boundary Indices:
			determine the indices of boundary components (transversal e and normal h)
			boundaryConditions = array of coded boundary info [xlow xhigh ylow yhigh zlow zhigh] 
			1 = boundaries to be considered (i.e. PEC-bounds)
			
		PMC (Neumann) Boundary conditions are given by default with the FIT operator stencil at boundary. So only if PEC = 1 this function makes an alteration to the given operators
		
		Initializes attributes
		
		:param boundaryConditions: array referencing [xlow xhigh ylow yhigh zlow zhigh]
		:type boundaryConditions: integer array
		:ivar epsBOUND: array of indices of e-components
		:ivar mueBOUND: array of indices of h-components
		"""
		# save time and rename
		bC = boundaryConditions
		# eps, mue container for indeces where boundary conditions are
		eps = np.array([],dtype = int)
		mue = np.array([],dtype = int)
		
		# do the finding using the canonical indexing
		if bC[0]==1:  #xlow
			for iy in np.array([i for i in range(self.ny)],dtype=int):
				ip = np.array([i for i in range(self.nz)],dtype=int)*self.nx*self.ny + (iy)*self.nx 
				eps = np.concatenate((eps, self.np+ip, 2*self.np+ip),axis = None) # y,z-comp.
				mue = np.concatenate((mue, ip),axis = None)           # x  -comp.
		
		if bC[1]==1:  # xhigh
			for iy in np.array([i for i in range(self.ny)],dtype=int):
				ip = np.array([i for i in range(self.nz)],dtype=int)*self.nx*self.ny + (iy)*self.nx + (self.nx-1)
				eps = np.concatenate((eps, self.np+ip, 2*self.np+ip),axis = None)
				mue = np.concatenate((mue, ip),axis = None)
		
		if bC[2]==1:  # ylow
			for ix in np.array([i for i in range(self.nx)],dtype=int):
				ip = np.array([i for i in range(self.nz)],dtype=int)*self.nx*self.ny + (0)*self.nx + ix
				eps = np.concatenate((eps, ip, 2*self.np+ip),axis = None)  # x,z-comp.
				mue = np.concatenate((mue, self.np+ip),axis = None)      # y  -comp. 
		 
		if bC[3]==1:  # yhigh
			for ix in np.array([i for i in range(self.nx)],dtype=int):
				ip = np.array([i for i in range(self.nz)],dtype=int)*self.nx*self.ny + (self.ny-1)*self.nx + ix
				eps = np.concatenate((eps, ip, 2*self.np+ip),axis = None)  # x,z-comp.
				mue = np.concatenate((mue, self.np+ip),axis = None)     # y  -comp.
		
		if bC[4]==1: # zlow
			for ix in np.array([i for i in range(self.nx)],dtype=int):
				ip = np.array([i for i in range(self.ny)],dtype=int)*self.nx + ix
				eps = np.concatenate((eps, ip,self.np+ip),axis = None)  # x,y-comp.
				mue = np.concatenate((mue, 2*self.np+ip),axis = None)   # z  -comp.
		  
		if bC[5]==1:  # zhigh
			for ix in np.array([i for i in range(self.nx)],dtype=int):
				ip = (self.nz-1)*self.nx*self.ny + np.array([i for i in range(self.ny)],dtype=int)*self.nx + ix
				eps = np.concatenate((eps, ip, self.np+ip),axis = None)  # x,y-comp.
				mue = np.concatenate((mue, 2*self.np+ip),axis = None) # z  -comp.
				
		self.epsBOUND = np.unique(eps)
		self.mueBOUND = np.unique(mue)
		
	def boundaryIndicesBOR(self, boundaryConditions):
		"""
		Boundary Indices:
			same as 'boundaryIndices' but for rotational body (rz-code)
		
		.. warning:: If called, will raise NotImplementedError
		"""
		# save time and rename
		bC = boundaryConditions
		raise NotImplementedError
		
	def buildMaterial(self):
		"""
		Builds material matrices and stores them as attributes:
		
		Also processes boundary information to represent the model geometry
		
		:ivar Meps: FIT epsilon matrix
		:ivar Mepsi: FIT epsilon matrix inverse
		:ivar Mmue: FIT mue matrix
		:ivar Mmuei: FIT mue matrix inverse
		"""
		# material
		nds = self.nullinv(self.ds)
		ndsd = self.nullinv(self.dsd)
		
		# M epsilon
		e0v = np.matrix(np.linspace(self.eps0,self.eps0,3*self.np)).T
		MeL = np.multiply(np.multiply(self.dad,e0v),nds)
		# boundary
		MeL[self.epsBOUND] = 0
		self.Meps = sp.spdiags(MeL.T[0],[0],3*self.np,3*self.np)
		self.Mepsi = sp.spdiags(self.nullinv(MeL.T[0]),[0],3*self.np,3*self.np)
		
		# M mue
		mue0v = np.matrix(np.linspace(self.mue0,self.mue0,3*self.np)).T
		MmueL = np.multiply(np.multiply(self.da,mue0v),ndsd)
		# boundary
		MmueL[self.mueBOUND] = 0
		self.Mmue = sp.spdiags(MmueL.T[0],[0],3*self.np,3*self.np)
		self.Mmuei = sp.spdiags(self.nullinv(MmueL.T[0]),[0],3*self.np,3*self.np)
	
	
	def fieldProbe(self, data, dataMonitor, spaceIndex, stepIndex):
		"""
		Field Monitor in time domain for subsequent frequency analysis
		
		dataMonitor container needs to be attribute of a FIT object. Function DOES NOT return anything.
		
		:param data: data each time step
		:type data: array like
		:param dataMonitor: data container for monitored field
		:type dataMonitor: array like
		:param spaceIndex: FIT canonical index
		:type spaceIndex: array like
		:param stepIndex: iterate time step
		:type stepIndex: number
		
		Example canonical index
			>>> spaceIndex = np.array([i for i in range(a.nx*a.ny)],dtype=int)+2*a.np
		"""
		# np.array([i for i in range(a.nx*a.ny)],dtype=int)+2*a.np
		dataMonitor[stepIndex] = data.todense()[spaceIndex]
	
	def fieldMonitor(self, data, dataMonitor, frequency, lenTimeSignal, stepIndex, dtmax):
		"""
		Field Monitor to assess field solution at given frequency point. Needs to be placed within for-loop of time domain simulation. Essentially performs DFT at a given frequency point. Take care to run as many time steps so that the monitored frequency can be resolved.
		
		dataMonitor is returned and needs to be stored. If it is an attribute of a FIT object this function would not automatically modify the content of the dataMonitor container
		
		:param data: data each time step
		:type data: array like (sparse vector, same as field type)
		:param dataMonitor: data container for monitored field
		:type dataMonitor: array like (sparse vector, same as field type)
		:param frequency: frequency to be monitored
		:type frequency: number 
		:param lenTimeSignal: length of timeSignal
		:param stepIndex: step index within for loop
		:param dtmax: time step ("sampling")
		
		:return dataMonitor: field monitor result
		:rtype dataMonitor: array like (sparse vector, same as field type)
		"""
		# guardian
		if not isinstance(1j,complex):
			raise Exception('1j is not the complex unit anymore')
		# dft term
		f = frequency
		t = stepIndex * dtmax
		fexp = np.exp(-1j * 2 * np.pi * f * t)
		# dataMonitor # H. "das physikalische Feld ist immer der Realteil"
		dataMonitor += np.real(np.multiply(1./lenTimeSignal,np.multiply(fexp,data)))
		return dataMonitor
		
	
	def gaussin(self,tdt, fmin, fmax):
		"""
		Standard Gaussian-spectrum modulated excitation signal
		
		:param tdt: time signal
		:type tdt: array like
		:param fmin: minimum frequency
		:type fmin: float
		:param fmax: maximum frequency
		:type fmax: float
		
		:return sig: gaussin modulated signal
		:rtype: array like
		:return npuls: normed puls end without trailing zeros
		:rtype npuls: array like
		"""
		lim1 = 1e-1  # spectrum factor at f1,f2
		lim2 = 1e-6  # error by signal step at it=1
		t0 = tdt[0]
		dt = tdt[1]-tdt[0]

		f0   = (fmin+fmax)*0.5
		df60 = np.abs(fmax-fmin)*0.5

		fg = df60 / np.sqrt(np.log(1/lim1))
		tpuls = np.sqrt(np.log(1/lim2)) /np.pi/fg

		# shift mid of signal to integer*dt
		tpuls = (np.floor(tpuls/ dt) + 1)*dt
		itpuls = int(np.floor(tpuls / dt) +1)

		# ======================================================================
		# symmetric modulated Gauss
		if np.abs(f0)<1e-12:
			sig = np.exp(-1*np.power(((tdt-t0-tpuls)*np.pi*fg),2))
		else:
			sig = np.multiply(np.exp( -1*np.power(((tdt-t0-tpuls)*np.pi*fg),2 )) , np.sin(2*np.pi*f0 *(tdt-t0-tpuls)))
		# ======================================================================
		
		# make symmetric signal end
		sig[2*itpuls:] = 0
		npuls = 2*itpuls-1

		# norm to max = 1
		sig = np.divide(sig,np.max(sig))
		
		return sig, npuls
		
	def fft(self, dtmax, timeSignal):
		"""
		Normal Fast Fourier Transform for time signals.
		
		Use for long time signals only, otherwise spectrum does not look very nice or use zero padding to increase frequency resolution. It is the fastest way to compute a frequency curve for an application. It uses a blackman window to mitigate wrong frequency  modulation due to finite length time signal.
		
		:param dtmax: time step ("sampling")
		:type dtmax: number
		:param timeSignal: time signal to be analysed
		:type timeSignal: numpy array
		"""
		# -----------------
		# Standard FFT
		w = scipy.signal.blackman(len(timeSignal))
		yf = scipy.fftpack.fft(timeSignal*w)#dft(len(y)).dot(y*w)
		xf = np.matrix(np.linspace(0.0, 1.0/(2.0*dtmax), len(timeSignal)//2)).T
		return xf, yf
		
	def dftMatrix(self,  frequency, timeSignalLength, dtmax):
		"""
		Constructs a linear operator, the DFT matrix, with a frequency selective sampling for use in `selectiveDFT()`
		
		:param frequency: array with frequencies to be sampled
		:type frequency: numpy array
		:param timeSignalLength: length of timeSignal to norm result
		:type timeSignalLength: number
		:param dtmax: time step (sampling)
		
		:return: dft matrix
		"""
		# guardian
		if not isinstance(1j,complex):
			raise Exception('1j is not the complex unit anymore')
		# ------------------
		# Frequency Selective DFT using linear operator DFT-Matrix
		f = frequency
		t = np.linspace(0,timeSignalLength*dtmax,timeSignalLength)
		fexp = np.matrix(np.zeros((len(f),timeSignalLength),dtype=complex))
		for iif in range(len(f)):
			fexp[iif,:] = np.matrix(np.exp(np.multiply(-1j * 2*np.pi * t, f[iif])))
		return fexp
		
	def selectiveDFT(self,dftMatrix, timeSignal):
		"""
		A specalized version of the discrete fourier transform, giving only frequency selective result for a given sampling vector in frequency domain. It is well suited for short time signals, that therefore have poor frequency resolution using a normal FFT, for long time signal the effort is excessive. It uses a linear operator, the DFT matrix, to optimize the calculation speed. This was made for UQ implementations where a frequency curve needs to be calculated for each of several runs with a time signal of equal length. Cutting the linear operator allows DFT of parts of the time signal.
		
		.. warning:: Pay attention to sampling. Time Signal needs to be long enough to give the correct frequency representation t_max > 1/f_min
		
		:param dftMatrix: DFT matrix generated for discrete, selective frequency points
		:type dftMatrix: numpy matrix
		:param timeSignal: time signal generated with `fieldMonitorTD()`
		:type timeSignal: numpy array
		
		:return: frequency representation of timeSignal
		:rtype: numpy matrix
		"""
		return dftMatrix * np.matrix(timeSignal).T
		
	def fieldEnergy(self,em1, e, h):
		"""
		Calculates the field energy of the discrete system
		
		:param e,h: e,h field column vector in sparse (FIT algebraic dimension)
		:type e,h: sparse array csc_matrix
		
		:param em1: leapfrog update n-1 e field vector (FIT algebraic dimension)
		:type em1: sparse array csc_matrix
		
		:return: scalar representing the momentary field energy
		"""		
		energy = 0.5 * (em1.T * self.Meps * e + h.T * self.Mmue * h)
		return energy[0,0]
		
	def plotField(self, X,Y,field,figureNumber, title):
		"""
		This function produces a 3D plot for a field quantity in one cutting plane X,Y
		
		:param X,Y: meshgrid representing a 2D cutting plane
		:type X,Y: array like
		:param field: field quantity scalar representation fitting mesh X,Y
		:type field: array like (dense matrix)
		:figureNumber: convenience to assigne dedicated plot window
		:type figureNumber: integer
		:param title: title of the plot figure
		:type title: string
		
		:return: axis handler of python plot axis3D object
		:rtype: matplotlib.axes._subplots.Axes3DSubplot
		"""
		fig = plt.figure(figureNumber)
		ax = fig.gca(projection='3d')
		ax.plot_surface(X,Y,field,cmap=cm.coolwarm, rstride=1, cstride=1, shade = True, alpha = 0.5, antialiased=True)
		plt.title(title)
		return ax
	
	
	def defaultModel(self, nx , ny , nz, LxMin, LyMin, LzMin, LxMax, LyMax, LzMax, boundaryConditions):
		"""
		A wrapper function to have everything as quickly as possible
		
		:param N_: number of points in _
		:type N_: integer
		
		:param L_Min: total length in _ (minimum)
		:type L_Min: float 
		
		:param L_Max: total length in _ (maximum)
		:type L_Max: float 
		
		Boundary Indices:
			determine the indices of boundary components (transversal e and normal h)
			boundaryConditions = array of coded boundary info [xlow xhigh ylow yhigh zlow zhigh] 
			1 = boundaries to be considered (i.e. PEC-bounds)
		
		PML (Neumann) Boundary conditions are given by default with the FIT operator stencil at boundary. So only if PEC = 1 this function makes an alteration to the given operators
		
		:param boundaryConditions: array referencing [xlow xhigh ylow yhigh zlow zhigh]
		:type boundaryConditions: integer array
		
		Initializes attributes
		
		:ivar epsBOUND: array of indices of e-components
		:ivar mueBOUND: array of indices of h-components
		
		All of the above are then accessible via attribute calling convention 
		
		Example usage
			>>> fitObject.nx
		"""
		# set parameters
		self.setGeometry(nx , ny , nz, -LxMax, -LyMax, -LzMax, LxMax, LyMax, LzMax)
		# make mesh
		self.makeMesh()
		# FIT operators
		self.createOperators()
		# build geomMatrices
		self.geomMatrices()
		# process boundary information
		self.boundaryIndices(boundaryConditions)
		# build material
		self.buildMaterial()
		
	def defaultModelBOR(self):
		"""
		A wrapper function to have everything as quickly as possible: rz code
		
		.. warning:: If called, will raise NotImplementedError
		"""
		return NotImplementedError

def main():
	"""
	Example to calculate steady state in rectangular box resonator
	
	main frame protected execution (means: only runs if directly executed. Does not run at import modul statements)
	"""
	# clock the execution
	startTime = time.time()
	# create instance
	a = FIT()
	# geometry
	nx = 71;
	ny = nx;
	nz = 2;
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
	Nt = 1400;
	tdt = np.linspace(0,Nt*dtmax,Nt)
	f1 = 1e2
	f2 = 6e6 
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
	a.etd = copy.deepcopy(tdt)
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
	ez = e.todense()[np.array([i for i in range(a.nx*a.ny)],dtype=int)+2*a.np]
	eField = np.reshape(ez,(a.nx,a.ny))	
	# plot visualization
	X, Y = np.meshgrid(a.xmesh,a.ymesh)
	figureNumber = 1
	title = 'ez field component; cutting-plane XY'
	ax = a.plotField(X,Y,eField,figureNumber, title)
	ax.set_zlim(-15, 15)
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
	# plt.figure(figureNumber+5)
	# plt.plot(tdt,sig)
	# plt.title('excitation signal in time')
	# plt.figure(figureNumber+6)
	# plt.plot(f,np.abs(a.selectiveDFT(dftM,sig)))
	# plt.title('spectrum of excitation signal')
	
	print time.time() - startTime
	plt.show()

# main frame protected execution
if __name__ == "__main__":	
	main()