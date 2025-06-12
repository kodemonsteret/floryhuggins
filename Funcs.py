import numpy as np
import scipy
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import label
import shutil
import os
import cupy as cp



def regularize_state_GPU(vec: np.ndarray, sum_max: float = 1, bounds: list[float] = [1e-8, 1]) -> np.ndarray:
    """ (taken and modified from Zwicker Group)
    regularize a state ensuring that variables stay within bounds

    The bounds for all variables are defined in the class attribute
    :attr:`variable_bounds`.

    Args:
        phi (:class:`~numpy.ndarray`):
            The state given as an array of local concentrations
        sum_max (float):
            The maximal value the sum of all concentrations may have. This can be
            used to limit the concentration of a variable that has been removed due
            to incompressibility. If this value is set to `np.inf`, the constraint
            is not applied
    """
    if not cp.all(cp.isfinite(vec)):
        raise RuntimeError("State is not finite")
    sum0A = cp.sum(vec[:,0])
    sum0B = cp.sum(vec[:,1])
    vec = cp.clip(vec, *bounds)
    # limit the sum of all variables
    if cp.isfinite(sum_max):
        phis = vec.sum(axis=1)
        loc = phis > sum_max
        if cp.any(loc):
            vec[loc,:] *= sum_max / phis[loc].reshape(-1,1)
    sum1A = cp.sum(vec[:,0])
    sum1B = cp.sum(vec[:,1])
    deltaA = sum0A - sum1A
    deltaB = sum0B - sum1B
    valids = cp.argwhere(vec.sum(axis=1)<0.9)
    ndeltaA = deltaA/len(valids)
    ndeltaB = deltaB/len(valids)
    
    vec[valids,0] += ndeltaA
    vec[valids,1] += ndeltaB
    vec = np.clip(vec, *bounds)
    # limit the sum of all variables
    return vec

def regularize_state(vec: np.ndarray, sum_max: float = 1, bounds: list[float] = [1e-8, 1]) -> np.ndarray:
    """ (taken and modified from Zwicker Group)
    regularize a state ensuring that variables stay within bounds

    The bounds for all variables are defined in the class attribute
    :attr:`variable_bounds`.

    Args:
        phi (:class:`~numpy.ndarray`):
            The state given as an array of local concentrations
        sum_max (float):
            The maximal value the sum of all concentrations may have. This can be
            used to limit the concentration of a variable that has been removed due
            to incompressibility. If this value is set to `np.inf`, the constraint
            is not applied
    """
    if not np.all(np.isfinite(vec)):
        raise RuntimeError("State is not finite")
    sum0A = np.sum(vec[:,0])
    sum0B = np.sum(vec[:,1])
    vec = np.clip(vec, *bounds)
    # limit the sum of all variables
    if np.isfinite(sum_max):
        phis = vec.sum(axis=1)
        loc = phis > sum_max
        if np.any(loc):
            vec[loc,:] *= sum_max / phis[loc].reshape(-1,1)
    sum1A = np.sum(vec[:,0])
    sum1B = np.sum(vec[:,1])
    deltaA = sum0A - sum1A
    deltaB = sum0B - sum1B
    valids = np.argwhere(vec.sum(axis=1)<0.9)
    ndeltaA = deltaA/len(valids)
    ndeltaB = deltaB/len(valids)
    
    vec[valids,0] += ndeltaA
    vec[valids,1] += ndeltaB
    vec = np.clip(vec, *bounds)
    # limit the sum of all variables

        
    return vec




def Deriv_scheme(order = 2, N_points=3, dx=1):
    '''
    This function returns the finite difference scheme for the derivative of order 'order' using 'N_points' points.
    The scheme is returned as a matrix of size N x N, where N is the number of points used in the scheme.
    The scheme is calculated using the method of undetermined coefficients.
    '''
    if order > N_points-1:
        raise ValueError('The order of differentiation must be less than the number of points used in the scheme.')

    # Initialize the scheme matrix
    scheme = np.zeros((N_points, N_points))

    # setting rightside vector
    B = np.roll(np.append(1, np.zeros(N_points-1)),order)*scipy.special.factorial(order)

    # Calculate the scheme
    for i in range(N_points):
        powers = np.linspace(0, N_points-1, N_points)
        Xs = powers-i
        A = np.meshgrid(Xs, powers)
        A = np.power(A[0], A[1])
        scheme[i] = np.linalg.solve(A, B)
    
    scheme = scheme/(dx**order)
    return scheme





def Deriv(axis = False, #Int, Along which axis should be differentiated
          order: int = 2, #Int, Order of differentiation
          L: list[float] = [100,100], # Float, Length of the domain
          Nx: list[int] = [100,100], # Int, Number of points in the domain
          N_diff = False, # Int, Number of points to use for differentiation
          closed_axis: list[int] = [] # list[int], which axes should use closed bondary conditions instead of periodic
          ):
    '''
    This function assigns the finite difference scheme for the derivative of order 'order' using 'N_points' points to the correct position in the scheme matrix
    '''
    if not N_diff:
        if order%2 == 0:
            N_diff = order+1
        else:
            N_diff = order+2

    Half = np.floor(N_diff/2).astype(int)
    
    schemes = []
    for i in range(len(Nx)):
        # first derive finite difference scheme
        scheme_local = Deriv_scheme(order, N_diff, L[i]/Nx[i])
        
        # then assign the scheme to the correct position in the scheme matrix
        scheme = scipy.sparse.diags(scheme_local[Half], range(-Half, Half+1), shape=(Nx[i], Nx[i]), format = 'csr')
        if order == 2 and i in closed_axis:
            scheme[0] = 0
            scheme[0,:2] = np.array([-1,1])/((L[i]/Nx[i])**2)
            scheme[-1] = 0
            scheme[-1,-2:] = np.array([1,-1])/((L[i]/Nx[i])**2)
        else:
            scheme[0,-Half:] = scheme_local[Half,-Half:]
            scheme[-1,:Half] = scheme_local[Half,:Half]

        
        scheme_copy = scheme.copy()
        if axis == False or axis == i:
            schemes.append(scheme_copy)
        else:
            schemes.append(scipy.sparse.csr_matrix((Nx[i],Nx[i])))
    
    scheme = scipy.sparse.csr_matrix((np.prod(Nx),np.prod(Nx)))

    for i in range(len(Nx)):
        for j in range(len(Nx)):
            if i == j:
                scheme_local = schemes[i]
            else:
                scheme_local = scipy.sparse.eye(Nx[j])
            if j != 0:
                scheme_local2 = scipy.sparse.kron(scheme_local2, scheme_local)
            else:
                scheme_local2 = scheme_local 
        scheme += scheme_local2

    return scheme


def count_droplets(fields, threshold=0, closed_axis = []):
    if fields.shape[0] == 2:
        threshold=0.3
    radss = []
    stdss = []
    bubbless = []
    for field in fields:
        if field.ndim == 2:
            bubbles, rad, std = count_droplets2d(field, threshold = threshold, closed_axis = closed_axis)
        elif field.ndim == 3:
            bubbles, rad, std = count_droplets3d(field, threshold = threshold, closed_axis = closed_axis)
        else:
            binary_mask = field > threshold
            labeled_array, bubbles = label(binary_mask)
            rads = []
            for i in range(1,bubbles):
                Vol = np.sum(labeled_array == i)
                n = field.ndim
                radius = ((3.14*n)**(1/(2*n)))*np.sqrt(n/(17.08))*(Vol**(1/n))
                rads.append(radius)

            if len(rads) != 0:
                rad = np.array(rads).mean()
                std = np.array(rads).std()
            else:
                rad=0
                std=0
        radss.append(rad)
        stdss.append(std)
        bubbless.append(bubbles)
        
    return np.array(bubbless), np.array(radss), np.array(stdss)

def count_droplets2d(field, threshold=0, closed_axis = []):
    """Count the number of droplets in the final state by labeling connected regions. For 2D grids"""
    binary_mask = field > threshold  # Identify droplets (positive phase)
    labeled_array, num_features = label(binary_mask)  # Count connected components  

    if 0 not in closed_axis:
        for i in range(labeled_array.shape[1]):
            x = labeled_array[0,i]
            if x != 0:
                x_back = labeled_array[-1,i]
                if x != x_back and x_back != 0:
                    index = np.argwhere(labeled_array==x_back).T
                    labeled_array[index[0],index[1]] = x
    if 1 not in closed_axis:
        for i in range(labeled_array.shape[0]):
            y = labeled_array[i,0]
            if y != 0:
                y_back = labeled_array[i,-1]
                if y_back!=0 and y!= y_back:
                    index = np.argwhere(labeled_array==y_back).T
                    labeled_array[index[0],index[1]] = y

    bubbles = np.unique(labeled_array)[1:]
    rads = []
    for i in bubbles:
        Area = np.sum(labeled_array == i)
        radius = np.sqrt(Area/3.14)
        rads.append(radius)
    if len(rads) != 0:
        stds = np.array(rads).std()
        rads = np.array(rads).mean()
    else:
        rads=0
        stds = 0
    return len(bubbles), rads , stds

def count_droplets3d(field, threshold=0.5, closed_axis = []):
    '''Count the number of droplets in the final state by labeling connected regions. For 3D grids'''
    binary_mask = field > threshold
    labeled_array, num = label(binary_mask)
    Nx = binary_mask.shape
    for i in range(3):
         if i not in closed_axis:
             grid = np.meshgrid(range(Nx[i-2]),range(Nx[i-1]))
             grid = np.vstack((grid[0].flatten(),grid[1].flatten())).T
             for j, k in grid:
                x_index = np.roll(np.array([0,j,k]),i)
                y_index = np.roll(np.array([-1,j,k]),i)
                x = labeled_array[*x_index]
                y = labeled_array[*y_index]
                if y != 0 and x!=0 and y != x:
                    indeces = np.argwhere(labeled_array==y).T
                    labeled_array[indeces[0],indeces[1],indeces[2]] = x
    bubbles = np.unique(labeled_array)[1:]
    rads = []
    for i in bubbles:
        Vol = np.sum(labeled_array == i)
        radius = ((3*Vol)/(4*3.14))**(1/3)
        rads.append(radius)

    if len(rads) != 0:
        stds = np.array(rads).std()
        rads = np.array(rads).mean()
    else:
        rads=0
        stds=0
    return len(bubbles), rads, stds



    


class TimePartial:
    def __init__(self, 
                 L = [100,100], # list[float], length of grid along each axis
                 Nx = [100,100], # list[int], number of gridpoints along each axis
                 closed_axis = [], # list[int], which axes should use closed bondary conditions instead of periodic
                 useGPU = False
                 ):
        self.L = L
        self.Nx = Nx
        self.closed_axis = closed_axis
        self.useGPU = useGPU
        self.parrams = ''
    
    def load_vec(self, vec):
        self.vec = vec
        return self
    
    def setup_CH(self, 
                 D, # Float, Diffusion constant
                 gamma, # Float, Coefficient of the fourth order term
                 precision = 'normal' # str , Precision of the solver (normal or high) how many points to use for differentiation
                 ):
        '''class method to setup the CH equation'''
        self.params = '\n'.join((f'D: {D}', f'gamma: {gamma}'))
        if precision == 'normal':
            second_deriv = Deriv(axis = False, order = 2, L = self.L, Nx = self.Nx, closed_axis=self.closed_axis)
        elif precision == 'high':
            second_deriv = Deriv(axis = False, order = 2, L = self.L, Nx = self.Nx, N_diff = 5, closed_axis=self.closed_axis)
        else:
            raise ValueError('Precision must be either normal or high.')
        
        if self.useGPU:
            second_deriv = cp.sparse.csr_matrix(second_deriv)

        def action(vec):
            dvec_dt = D*second_deriv @ (vec**3 - vec - gamma*second_deriv@vec)
            return dvec_dt
        
        self.action = action
        return self
    

    def setup_CH2(self,
                a,  # Float, coefficient for the cubic term
                b,  # Float, coefficient for the linear term
                kappa,  # Float, coefficient for the fourth-order term
                Lambda,  # Float, constant mobility
                c_c, #critical concentration
                precision='normal'  # str, precision level
                ):
        self.params = '\n'.join((f'a: {a}', f'b: {b}', f'kappa: {kappa}', f'Lambda: {Lambda}', f'c_c: {c_c}'))
        if precision == 'normal':
            laplacian = Deriv(axis=False, order=2, L=self.L, Nx=self.Nx, closed_axis=self.closed_axis)
        elif precision == 'high':
            laplacian = Deriv(axis=False, order=2, L=self.L, Nx=self.Nx, N_diff=5, closed_axis=self.closed_axis)
        else:
            raise ValueError('Precision must be either normal or high.')
        if self.useGPU:
            laplacian = cp.sparse.csr_matrix(laplacian)
        def action(vec): 
            dvec_dt = Lambda * laplacian @ (a * (vec - c_c)**3 - b * (vec - c_c) - kappa * laplacian @ vec)
            return dvec_dt
        self.action = action
        return self
    

    def setup_FH(self,
                 chi: list[float] = [1,0,0],   # AB SA SB interaction parameters
                 v: float = 1., # molecular volume
                 kBT: float = 1., # boltzman constant times temperature 
                 precision: str = 'normal'
                 ):
        '''setup timepartial according to the Flory huggins potential assuming constant temperature'''
        self.params = '\n'.join(('chi: AB, AS, BS',f'chi: {chi}', f'v: {v}', f'kBT: {kBT}'))
        if precision == 'normal':
            laplacian = Deriv(axis=False, order=2, L=self.L, Nx=self.Nx, closed_axis=self.closed_axis)
        elif precision == 'high':
            laplacian = Deriv(axis=False, order=2, L=self.L, Nx=self.Nx, N_diff=5, closed_axis=self.closed_axis)
        if self.useGPU:
            laplacian = cp.sparse.csr_matrix(laplacian)
            kappaAB = (v**(2/3)/2)*cp.array([chi[0]-chi[1]-chi[2],chi[0]-chi[1]-chi[2]])
            kappas = (v**(2/3))*cp.array([chi[1],chi[2]])
            chiAB = chi[0]
            chi = cp.array([chi[1:]])
            
            def action(vec):
                Diff = laplacian @ vec
                LAP = Diff[:,::-1]*kappaAB
                LAPR = Diff*kappas 
                return laplacian@(kBT*(cp.log(vec)-cp.log((1-vec-vec[:,::-1]))+(chiAB-chi[::-1])*vec[:,::-1] + chi*(1-2*vec-vec[:,::-1])+LAP-LAPR))
            self.action = action 
        else:
            kappaAB = (v**(2/3)/2)*np.array([chi[0]-chi[1]-chi[2],chi[0]-chi[1]-chi[2]])
            kappas = (v**(2/3))*np.array([chi[1],chi[2]])
            chiAB = chi[0]
            chi = np.array([chi[1:]])
            
            def action(vec):
                Diff = laplacian @ vec
                LAP = Diff[:,::-1]*kappaAB
                LAPR = Diff*kappas 
                return laplacian@(kBT*(np.log(vec)-np.log((1-vec-vec[:,::-1]))+(chiAB-chi[::-1])*vec[:,::-1] + chi*(1-2*vec-vec[:,::-1])+LAP-LAPR))
            self.action = action 

    def setup_FH_old(self,
                 chi: list[float] = [1,0,0],   # AB SA SB interaction parameters
                 v: float = 1., # molecular volume
                 kBT: float = 1., # boltzman constant times temperature 
                 kappaA: float = 1.,
                 kappaB: float =1.,
                 kappaS: float = 1.,
                 precision: str = 'normal'
                 ):
        '''setup timepartial according to the Flory huggins potential assuming constant temperature'''
        self.params = '\n'.join(('chi: AB, AS, BS',f'chi: {chi}', f'v: {v}', f'kBT: {kBT}', f'kappaA: {kappaA}', f'kappaB: {kappaB}'))
        if precision == 'normal':
            laplacian = Deriv(axis=False, order=2, L=self.L, Nx=self.Nx, closed_axis=self.closed_axis)
        elif precision == 'high':
            laplacian = Deriv(axis=False, order=2, L=self.L, Nx=self.Nx, N_diff=5, closed_axis=self.closed_axis)
        if self.useGPU:
            laplacian = cp.sparse.csr_matrix(laplacian)
            chiAB = chi[0]
            chi = cp.array([chi[1:]])
            kappamatrix = cp.array([[kappaA,kappaB]])
            def action(vec):
                LAP = (laplacian @ vec)*kappamatrix
                LAPR = kappaS*laplacian @ (vec[:,::-1]+vec)
                return laplacian@(kBT*(cp.log(vec)-cp.log((1-vec-vec[:,::-1]))+(chiAB-chi[::-1])*vec[:,::-1] + chi*(1-2*vec-vec[:,::-1]))-LAP-LAPR)
            self.action = action 
        else:
            chiAB = chi[0]
            chi = np.array([chi[1:]])
            kappamatrix = np.array([[kappaA,kappaB]])
            def action(vec):
                LAP = (laplacian @ vec)*kappamatrix
                LAPR = kappaS*laplacian @ (vec[:,::-1]+vec)
                return laplacian@(kBT*(np.log(vec)-np.log((1-vec-vec[:,::-1]))+(chiAB-chi[::-1])*vec[:,::-1] + chi*(1-2*vec-vec[:,::-1]))-LAP-LAPR)
            self.action = action 

    


    def __call__(self,vec):
        self.load_vec(vec)
        return self.action(self.vec)



class integrator:
    def __init__(self, 
                 Grid0, # Array, Initial condition 
                 partial, # callable, Partial differential equation
                 dt = 0.01, # Float, Time step
                 t_end = 1, # Float, End time
                 scheme = 'explicit', # Str, Integration scheme 'implicit' or 'explicit' or 'rk4' implicit and explicit are euler methods
                 N_frames = 50, # Int, Number of frames to show in the animation 
                 regularize: bool = False, #whether to regularize or not
                 klist: list[float] = [0.1,0.1,0.1,0.1], # Array, constants for chemical reactions
                 chemtype: str = "pass",
                 threshold = 0.5
                ) -> object:
        '''
        Class to integrate a partial differential equation
        '''
        if Grid0.ndim != len(partial.Nx) and Grid0.shape[0] != np.prod(np.array(partial.Nx)) and Grid0.ndim != (len(partial.Nx)+1) and Grid0.shape[1] != np.prod(np.array(partial.Nx)):
            raise ValueError('The dimension of the initial condition must be the same as the number of dimensions in the partial differential equation.')
        if Grid0.shape[-1] == 2:
            self.Grid = Grid0.reshape(-1,2)
        elif Grid0.shape[0] == 2:
            self.Grid = Grid0.reshape(2,-1).T
        else:
            self.Grid = Grid0.flatten()
        
        self.threshold = threshold
        self.frames = [self.Grid.copy()]
        self.partial = partial
        self.regularize = regularize
        self.dt = dt
        self.t = 0
        self.scheme = scheme
        self.t_end = t_end
        self.N_frames = N_frames
        dropletcount, rad, std = count_droplets(self.Grid.T.reshape(-1,*self.partial.Nx), self.threshold)
        self.droplets = [dropletcount]
        self.rads = [rad]
        self.stds = [std]
        self.dts = [dt]
        self.dts = [dt]
        self.klist = klist
        self.chemtype = chemtype
        self.tlistframes = [0]
        if self.partial.useGPU:
            self.Grid = cp.array(self.Grid)
        
    def check_dt(self):
        Grid1 = self.Grid.copy()
        Grid2 = self.Grid.copy()
        Grid1 = self.step(Grid1, self.dt)
        Grid2 = self.step(Grid2, self.dt/2)
        Grid2 = self.step(Grid2, self.dt/2)
        if np.all(np.abs((Grid1-Grid2)) < 1e-5) and np.all(np.abs(np.sum((Grid1-Grid2),axis=1)) < 1e-5):
            self.Grid = Grid2
            try:
                self.Grid = self.boundary(self.Grid)
            except:
                pass
            if self.regularize and self.partial.useGPU:
                self.Grid = regularize_state_GPU(self.Grid)
            elif self.regularize:
                self.Grid = regularize_state(self.Grid)
            self.Grid = Chemical_reactions(self.Grid,self.klist, method = self.chemtype, dt = self.dt)  
            self.dts.append(self.dt)
            self.t += self.dt
            self.dt/=0.8
            self.dt/=0.8
        else:
            self.dt *= 0.8
            self.dt *= 0.8
        return self


    def step(self, Grid, dt):
        if self.scheme == 'implicit':
            Grid *= 1/(1-dt*self.partial(Grid))
        elif self.scheme == 'explicit':
            Grid += dt*self.partial(Grid)
        elif self.scheme == 'rk4':
            k1 = dt*self.partial(Grid)
            k2 = dt*self.partial(Grid + k1/2)
            k3 = dt*self.partial(Grid + k2/2)
            k4 = dt*self.partial(Grid + k3)
            Grid += (k1 + 2*k2 + 2*k3 + k4)/6
        else:
            raise ValueError('The scheme must be either implicit, explicit or rk4.')
        return Grid

    def SetConcentrationGrad(self, axis = 0, sinkVal = 1, sourceVal = 0):
        source_idx = []
        sink_idx = []
        for i in range(len(self.partial.Nx)):
            if i == axis:
                source_idx.append(-1)
                sink_idx.append(0)
            else:
                source_idx.append(slice(None))
                sink_idx.append(slice(None))

        Gridspace = np.zeros(self.partial.Nx)
        Gridspace[*sink_idx] = 1
        Gridspace = Gridspace.flatten()
        sink_idx = np.argwhere(Gridspace)
        
        Gridspace = np.zeros(self.partial.Nx)
        Gridspace[*source_idx] = 1
        Gridspace = Gridspace.flatten()
        source_idx = np.argwhere(Gridspace)
        if self.partial.useGPU:
            source_idx = cp.array(source_idx)
            sink_idx = cp.array(sink_idx)
        
        def boundary(Grid):
            Grid[source_idx] = sourceVal
            Grid[sink_idx] = sinkVal
            return Grid
        self.boundary = boundary



    def integrate(self):
        print('Integrating...')
        TimesForSaving = np.linspace(0,self.t_end, self.N_frames)
        TimesForSaving = TimesForSaving[1:]
        while self.t < self.t_end:
            self.check_dt()
            if self.t >= TimesForSaving[0]:
                TimesForSaving = TimesForSaving[1:]
                Grid_numpy = cp.asnumpy(self.Grid)
                self.frames.append(Grid_numpy.copy())
                dropletcount, rad,std = count_droplets(Grid_numpy.T.reshape(-1,*self.partial.Nx),self.threshold, self.partial.closed_axis)
                self.droplets.append(dropletcount)
                self.rads.append(rad)
                self.stds.append(std)
            print('Progress: '+str(np.round(100*self.t/self.t_end)), '%', end = '\r')
        return self

    def droplet_graph(self, count = True):
        self.droplets = np.array(self.droplets)
        self.rads = np.array(self.rads)
        self.stds = np.array(self.stds)
        if count:
            x = np.linspace(0, self.t_end, len(self.droplets))
            fig, ax = plt.subplots(2,3)
            #print(self.droplets.shape)
            ax[0,0].plot(x, self.droplets[:,0], label = 'A')
            ax[0,0].set_xlabel('Time')
            ax[0,0].set_ylabel('Number of droplets')

            ax[0,1].plot(x,self.rads[:,0], label = 'A')
            ax[0,1].set_xlabel('Time')
            ax[0,1].set_ylabel('mean dropletsize')
            
            ax[0,1].fill_between(x, np.array(self.rads[:,0])-np.array(self.stds[:,0]), np.array(self.rads[:,0])+np.array(self.stds[:,0]), alpha = 0.5)

            ax[0,2].plot(x,self.stds[:,0])
            ax[0,2].set_xlabel('Time')
            ax[0,2].set_ylabel('std dropletsize')

            ax[1,0].plot(x, self.droplets[:,1], label = 'B')
            ax[1,0].set_xlabel('Time')
            ax[1,0].set_ylabel('Number of droplets')

            ax[1,1].plot(x,self.rads[:,1], label = 'B')
            ax[1,1].set_xlabel('Time')
            ax[1,1].set_ylabel('mean dropletsize')

            ax[1,1].fill_between(x, np.array(self.rads[:,1])-np.array(self.stds[:,1]), np.array(self.rads[:,1])+np.array(self.stds[:,1]), alpha = 0.5)

            ax[1,2].plot(x,self.stds[:,1])
            ax[1,2].set_xlabel('Time')
            ax[1,2].set_ylabel('std dropletsize')

            
            plt.show()
        return self
    def show(self, animate = True # Bool, Whether to animate the solution
             ):
        print('Animating...')
        if self.Grid.shape[-1] == 2:
            if animate:
                DT = self.t_end/(self.N_frames-1)
                fig, ax = plt.subplots()
                image = self.frames[-1].reshape(*self.partial.Nx,-1)
                image = np.append(image,np.zeros((*self.partial.Nx,1)),axis = -1)
                image = image[*((np.array(self.partial.Nx[:-2])//2).astype(int))]
                # fig.colorbar(ax.imshow(image))
                #fig.colorbar(ax.imshow(np.append((self.frames[-1].reshape(*self.partial.Nx)[:,*np.array(self.partial.Nx[:-2])//2],np.zeros((*self.partial.Nx[-2:]))))))
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                def update(frame):
                    ax.clear()
                    ax.text(0.05, 0.95, self.partial.params, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
                    plt.title('Time: %.2f' % (DT*frame))
                    image = self.frames[frame].reshape(*self.partial.Nx,-1)
                    image = np.append(image,np.zeros((*self.partial.Nx,1)),axis = -1)
                    image = image[*((np.array(self.partial.Nx[:-2])//2).astype(int))]
                    #print the mean of A and B in the latest frame
                    return ax.imshow(image)

                    #return ax.imshow(np.vstack((self.frames[frame].reshape(2,*self.partial.Nx)[:,*np.array(self.partial.Nx[:-2])//2],np.zeros((1,*self.partial.Nx[-2:])))).T)
                ani = animation.FuncAnimation(fig, update, frames = range(len(self.frames)), repeat = False)
                ani.save('animation.gif')
                plt.show()
            else:
                fig, ax = plt.subplots(3,4)
        else:
            if animate:
                DT = self.t_end/self.N_frames
                fig, ax = plt.subplots()
                fig.colorbar(ax.imshow(self.frames[-1].reshape((self.partial.Nx))[*np.array(self.partial.Nx[:-2])//2],cmap = 'plasma',vmin = -1, vmax = 1))
                def update(frame):
                    ax.clear()
                    plt.title('Time: %.2f' % (frame*DT))
                    return ax.imshow(self.frames[frame].reshape(self.partial.Nx)[*np.array(self.partial.Nx[:-2])//2], cmap = 'plasma',vmin=-1, vmax=1)
                ani = animation.FuncAnimation(fig, update, frames = range(len(self.frames)), repeat = False)
                
                ani.save('animation.gif')
                plt.show()
            else:
                fig, ax = plt.subplots(3,4)
    


    def save(self,
             runname, # str, name of folder to save data
             save_frames = False # bool, should the grids be saved?
             ):
        
        new_folder = os.path.join(os.getcwd(), runname)
        FilesSaved = ["mean_field.png", "animation.gif"]
        os.makedirs(new_folder, exist_ok=True)
        if save_frames:
            frames = np.array(self.frames)
            np.savetxt(runname+'frames.txt',frames)
        T = np.linspace(0,self.t_end, len(self.rads))
        counts = np.array(self.droplets)
        Rads = np.array(self.rads)
        stds = np.array(self.stds)
        A = np.append(T.reshape(-1,1),counts, axis = 1)
        A = np.append(A,Rads, axis = 1)
        A = np.append(A,stds, axis = 1)
        np.savetxt(runname+'\\'+'countsAndRads.txt',A)
        B = []
        for i in range(len(self.frames)):
            B.append([np.mean(self.frames[i][:,0]), np.mean(self.frames[i][:,1])])
        B = np.array(B)
        np.savetxt(runname+'\\'+'mean_field.txt',B)
        for file in FilesSaved:
            if os.path.exists(file):
                new_filename = f"{runname}_{file}"
                shutil.copy(file, os.path.join(new_folder, new_filename))
    

    def mean_field_graph(self):
        c1 =[]
        c2 = []
        for i in self.frames:
            c1.append(np.mean(i[:,0]))
            c2.append(np.mean(i[:,1]))
        t = np.linspace(0,self.t_end,len(c1))
        fig, ax = plt.subplots(1,3)
        #set the size of the figure to 20x10
        fig.set_size_inches(15, 5)
        ax[0].plot(t,c1, label = 'A')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Mean concentration')
        ax[0].legend()
        ax[0].set_title('Mean field concentration of A')
        ax[1].plot(t,c2, label = 'B')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Mean concentration')
        ax[1].legend()
        ax[1].set_title('Mean field concentration of B')
        ax[2].plot(t,c1, label = 'A')
        ax[2].plot(t,c2, label = 'B')
        ax[2].set_xlabel('Time')
        ax[2].set_ylabel('Mean concentration')
        ax[2].legend()
        ax[2].set_title('Mean field concentration of A and B')
        plt.savefig('mean_field.png')
        plt.show()

            



    def __call__(self, *args, **kwds):
        self.__dict__.update(kwds)
        self.integrate()
        # if self.Grid.shape[-1] != 2:
        #     return self.show(), self.droplet_graph()
        # else:
        #return self.show(), self.mean_field_graph()
        return self.show(), self.droplet_graph(),self.mean_field_graph()




def gmm(mu1=0.4,sigma1=0.02,mu2=0.3,sigma2=0.02,dim = [60,60]):
    """Function to generate a grid with two Gaussian distributions
    Args:
    dim: list[int], dimensions of the grid
    mu1: float, mean of the first Gaussian distribution
    sigma1: float, standard deviation of the first Gaussian distribution
    mu2: float, mean of the second Gaussian distribution
    sigma2: float, standard deviation of the second Gaussian distribution
    Returns:
    np.ndarray, grid with two Gaussian distributions"""
    grid1 = np.random.normal(mu1,sigma1,[1,np.prod(dim)])
    grid2 = np.random.normal(mu2,sigma2,[1,np.prod(dim)])
    grid = np.append(grid1,grid2,axis=0)
    grid = regularize_state(grid.T).T
    return grid





def Chemical_reactions(Grid, 
                       k: list[float]=[1,1,1,1],  
                       method: str = 'pass',
                       dt: float = 0.001 ,) -> np.ndarray:
    '''Function to simulate chemical reactions outputs 
    Args:
    Grid: np.ndarray, initial condition
    k: list[float], rate constants
    method: str, type of reaction to simulate (pass, inhibit or destroy)
    dt: float, time step
    
    Returns:
    np.ndarray, final state of the system after one iteration of chemical reactions
    '''
    if method == 'pass':
        return Grid
    elif method == "inhibit":
        cs = 1-Grid.sum(axis=1)
        dca = k[0]*Grid[:,1]*cs - k[1]*Grid[:,0]
        dcb = cs/(1+k[2]*Grid[:,0]) - k[3]*Grid[:,1]
        Grid[:,0] += dca*dt
        Grid[:,1] += dcb*dt
    elif method == "destroy":
        cs = 1-Grid.sum(axis=1)
        dca = k[0]*Grid[:,1]*cs - k[1]*Grid[:,0]
        dcb = -k[2]*Grid[:,0]*Grid[:,1]+k[3]*cs
        Grid[:,0] += dca*dt
        Grid[:,1] += dcb*dt
    elif method == "normaldestroy":
        cs = 1-Grid.sum(axis=1)
        dca = k[0]*Grid[:,1] - k[1]*Grid[:,0]
        dcb = -k[2]*Grid[:,0]*Grid[:,1]+k[3]
        Grid[:,0] += dca*dt
        Grid[:,1] += dcb*dt
    elif method == 'cuberoot':
        cs = 1-Grid.sum(axis=1)
        dca = k[0]*Grid[:,1]*cs - k[1]*Grid[:,0]
        dcb = -k[2]*Grid[:,0]*Grid[:,1]**(1/3)+k[3]*cs
        Grid[:,0] += dca*dt
        Grid[:,1] += dcb*dt

    return Grid


