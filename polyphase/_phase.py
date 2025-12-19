import pdb
import numpy as np
import time
import pandas as pd
import os
from collections import Counter
    
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist, euclidean, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from numpy.linalg import norm

import warnings
from itertools import combinations
from math import pi
from collections import defaultdict 
import ray

MIN_POINT_PRECISION = 1e-8
            
""" Main functions serial """
def makegridnd(meshsize, dimension):
    """
    Given mesh size and a dimensions, creates a n-dimensional grid for the volume fraction.
    Note that the grid would be a hyper plane in the n-dimensions.
    """
    x = np.meshgrid(*[np.linspace(MIN_POINT_PRECISION, 1,meshsize) for d in range(dimension)])
    mesh = np.asarray(x)
    total = np.sum(mesh,axis=0)
    plane_mesh = mesh[:,np.isclose(total,1.0,atol=1e-2)]

    return plane_mesh

def label_simplex(grid, simplex, thresh):
    """ given a simplex, labels it to be a n-phase region by computing number of connected components """
    coords = [grid[:,x] for x in simplex]
    dist = squareform(pdist(coords,'euclidean'))
    adjacency = dist<thresh
    adjacency =  adjacency.astype(int)  
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    return n_components

def is_purecomp_hull(grid, simplex):
    """ 
    return True if a simplex connects only the pure components
    The assumption when using this function as a simplex refining method is that, 
    the lower conex hull only consists the simplex coming from the pure component connections.
    """
    dim = grid.shape[0]
    points = grid[:,simplex]
    flags = np.zeros(dim)
    for ind,pt in enumerate(points):
        flags[ind] = is_nzero_comp(dim-1,pt)
        
    if np.sum(flags)==dim:
        return True
    else:
        return False

def is_upper_hull(grid, simplex):
    """ 
    return True if a simplex connects anything on the edge.
    
    The assumption is that everything that connects to the edge belongs to upper convex hull.
    We would want to compute only the lower convex hull.
    """
    point = grid[:,simplex]
    if np.isclose(point, MIN_POINT_PRECISION).any():
        return True
    else:
        return False


def lift_label(grid,lift_grid, simplex, label):
    """ Lifting the labels from simplices to points """
    try:
        v = np.asarray([grid[:-1,x] for x in simplex])
        tri = Delaunay(v)
        inside = Delaunay.find_simplex(tri,lift_grid[:-1,:].T)
        inside =~(inside<0)
        iscoplanar = False
    except Exception as err:
        inside = None
        iscoplanar = True
        
    return inside, iscoplanar
   
def is_boundary_point(point, zero_value = MIN_POINT_PRECISION):
    if np.isclose(point, MIN_POINT_PRECISION).any():
        return True
    else:
        return False

def is_pure_component(point, zero_value = MIN_POINT_PRECISION):
    counts = Counter(point)
    if counts[MIN_POINT_PRECISION]>1:
        return True
    else:
        return False

def get_max_delaunay_edge_length(grid):
    delaunay = Delaunay(np.asarray(grid[:-1,:].T))
    max_delaunay_edge = 0.0
    for sx in delaunay.simplices:
        vertex_sx = [grid[:,x] for x in sx]
        edges = combinations(vertex_sx, 2)
        edge_lengths = np.array([norm(e[0]-e[1]) for e in edges])
        current_max = np.max(edge_lengths)
        if max_delaunay_edge<current_max:
            max_delaunay_edge = current_max
    
    return max_delaunay_edge       


def is_nzero_comp(n,point, zero_value = MIN_POINT_PRECISION):
    n_out = np.sum(np.isclose(point, MIN_POINT_PRECISION))
    
    return n_out>=n

def point_at_inifinity_convexhull(points):
    inf_ind = np.shape(points)[0]
    base_points = points[:,:-1].mean(axis=0)
    inf_height = 1e10*abs(max(points[:,-1]))
    p_inf = np.hstack((base_points,inf_height))
    points_inf = np.vstack((points,p_inf))
    hull = ConvexHull(points_inf)
    lower = ~(hull.simplices==inf_ind).any(axis=1)
    lower_hull = hull.simplices[lower]
    
    return lower_hull, hull,~lower

def negative_znorm_convexhull(points):
    hull = ConvexHull(points)
    zlower = hull.equations[:,-2]<=0
    lower_hull = hull.simplices[zlower]
    
    return lower_hull, hull, ~zlower

""" Main comoutation function """
def _serialcompute(f, dimension, meshsize,**kwargs):
    """
    Main python function to obtain a phase diagram for n-component polymer mixture system.   
    """
    verbose = kwargs.get('verbose', False)
    lower_hull_method = kwargs.get('lower_hull_method', None)
    flag_lift_label = kwargs.get('flag_lift_label',False)
    lift_grid_size = kwargs.get('lift_grid_size', meshsize)    
    since = time.time()
  
    outdict = defaultdict(list)
    
    """ Perform a parallel computation of phase diagram """
    # 1. generate grid
    grid = makegridnd(meshsize, dimension)
    outdict['grid'] = grid
    
    lap = time.time()
    if verbose:
        print('{}-dimensional grid generated at {:.2f}s'.format(dimension,lap-since))

    energy = np.asarray([f(x) for x in grid.T])

    lap = time.time()
    if verbose:
        print('Energy computed at {:.2f}s'.format(lap-since))
    
    max_energy = np.max(energy)
    
    if lower_hull_method is None:
        pad_energy = kwargs.get('pad_energy',2)
        doctor_points = np.asarray([is_boundary_point(x) for x in grid.T])
        energy[doctor_points] = pad_energy*max_energy
        
        if verbose:
            print('Aplpying {:d}x padding of {:.2f} maximum energy'.format(pad_energy, max_energy))
    
    outdict['energy'] = energy
    
    lap = time.time()
    if verbose:
        print('Energy is corrected at {:.2f}s'.format(lap-since))
    points = np.concatenate((grid[:-1,:].T,energy.reshape(-1,1)),axis=1) 
    
    if lower_hull_method is None:    
        hull = ConvexHull(points)
        upper_hull = np.asarray([is_upper_hull(grid,simplex) for simplex in hull.simplices])
        simplices = hull.simplices[~upper_hull]
    elif lower_hull_method=='point_at_infinity':
        simplices, hull,upper_hull = point_at_inifinity_convexhull(points)
    elif lower_hull_method=='negative_znorm':
        simplices, hull,upper_hull = negative_znorm_convexhull(points)
            
    outdict['upper_hull']=upper_hull
    outdict['hull'] = hull
    
    lap = time.time()
    if verbose:
        print('Simplices are computed and refined at {:.2f}s'.format(lap-since))
        
    outdict['simplices'] = simplices
    if verbose:
        print('Total of {} simplices in the convex hull'.format(len(simplices)))

    thresh_scale = kwargs.get('thresh_scale',1.25)
    thresh = thresh_scale*euclidean(grid[:,0],grid[:,1])
    
    if verbose:
        print('Using {:.2E} as a threshold for Laplacian of a simplex'.format(thresh)) 
        
    outdict['thresh'] = thresh
    
    # 4. for each simplex in the hull compute number of connected components (parallel)
    num_comps = [label_simplex(grid, simplex, thresh) for simplex in simplices]
    lap = time.time()
    if verbose:
        print('Simplices are labelled at {:.2f}s'.format(lap-since))
    outdict['num_comps'] = num_comps
    outdict['coplanar'] = None
    
    if flag_lift_label:
        if lift_grid_size == meshsize:
            lift_grid = grid
        else:
            lift_grid = makegridnd(lift_grid_size, dimensions) # we lift labels to a constant mesh 
        
        inside = [lift_label(grid, lift_grid, simplex, label) for simplex, label in zip(simplices, num_comps)]
        
        coplanar = [item[1] for item in inside]
        outdict['coplanar']=np.asarray(coplanar)
        lap = time.time()
        if verbose:
            print('Labels are lifted at {:.2f}s'.format(lap-since))

            print('Total {}/{} coplanar simplices'.format(np.sum(coplanar),len(simplices)))

        phase = np.zeros(lift_grid.shape[1])
        for i,label in zip(inside,num_comps):
            if not i[1]:
                phase[i[0]] = label
        phase = phase.reshape(1,-1)
        output = np.vstack((lift_grid,phase))
        index = ['Phi_'+str(i) for i in range(1, output.shape[0])]
        index.append('label')
        output = pd.DataFrame(data = output,index=index)
                
    else:
        output = []
        
    outdict['output'] = output 
    
    lap = time.time()
    if verbose:
        print('Computation took {:.2f}s'.format(lap-since))
    
    return outdict

@ray.remote
def ray_is_boundary_point(point, zero_value = MIN_POINT_PRECISION):
    if np.isclose(point, MIN_POINT_PRECISION).any():
        return True
    else:
        return False

@ray.remote
def ray_is_pure_component(point, zero_value = MIN_POINT_PRECISION):
    counts = Counter(point)
    if counts[MIN_POINT_PRECISION]>1:
        return True
    else:
        return False

@ray.remote
def ray_label_simplex(grid, simplex, thresh):
    """ given a simplex, labels it to be a n-phase region by computing number of connected components """
    coords = [grid[:,x] for x in simplex]
    dist = squareform(pdist(coords,'euclidean'))
    adjacency = dist<thresh
    adjacency =  adjacency.astype(int)  
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    return n_components

@ray.remote
def ray_is_upper_hull(grid, simplex):
    """ 
    return True if a simplex connects anything on the edge.
    
    The assumption is that everything that connects to the edge belongs to upper convex hull.
    We would want to compute only the lower convex hull.
    """
    point = grid[:,simplex]
    if np.isclose(point, MIN_POINT_PRECISION).any():
        return True
    else:
        return False

@ray.remote    
def ray_lift_label(grid,lift_grid, simplex, label):
    """ Lifting the labels from simplices to points """
    try:
        v = np.asarray([grid[:-1,x] for x in simplex])
        #inside = inpolyhedron(v, grid[:-1,:].T)
        tri = Delaunay(v)
        inside = Delaunay.find_simplex(tri,lift_grid[:-1,:].T)
        inside =~(inside<0)
        flag = 1
    except:
        inside = None
        flag = 0
        
    return inside, flag

def _parcompute(f, dimension, meshsize,**kwargs):
    """Compute phase diagram using parallel computaion
    parallel version of serialcompute
    """
    verbose = kwargs.get('verbose', False)
    flag_lift_label = kwargs.get('flag_lift_label',False)
    use_weighted_delaunay = kwargs.get('use_weighted_delaunay', False)
    lift_grid_size = kwargs.get('lift_grid_size', 200)
        
    # Initialize ray for parallel computation
    ray.init(ignore_reinit_error=True)

    since = time.time()
    
    outdict = {}
    thresh_epsilon = 5e-3
    
    """ Perform a parallel computation of phase diagram """
    # 1. generate grid
    grid = makegridnd(meshsize, dimension)
    outdict['grid'] = grid
    grid_ray = ray.put(grid)
    lap = time.time()
    if verbose:
        print('{}-dimensional grid generated at {:.2f}s'.format(dimension,lap-since))
       
    energy = np.asarray([f(x) for x in grid.T])   
    
    lap = time.time()
    if verbose:
        print('Energy computed at {:.2f}s'.format(lap-since))

    outdict['energy'] = energy
    
    lap = time.time()
    if verbose:
        print('Energy is corrected at {:.2f}s'.format(lap-since))
    
    # 3. Compute convex hull
    points = np.concatenate((grid[:-1,:].T,energy.reshape(-1,1)),axis=1) 
    simplices, hull,upper_hull = point_at_inifinity_convexhull(points)
    outdict['upper_hull']=upper_hull
    outdict['hull'] = hull    
    outdict['simplices'] = simplices
        
    if verbose:
        print('Total of {} simplices in the convex hull'.format(len(simplices)))
        
    lap = time.time()
#    if verbose:
#        print('{} is computed at {:.2f}s'.format(_method,lap-since))

    if verbose:
        method_name = getattr(f, "__name__", str(f))
        print(f"{method_name} is computed at {lap - since:.2f}s")


    thresh_scale = kwargs.get('thresh_scale',1.25)
    thresh = thresh_scale*euclidean(grid[:,0],grid[:,1])
    
    if verbose:
        print('Using {:.2E} as a threshold for Laplacian of a simplex'.format(thresh)) 
        
    outdict['thresh'] = thresh

    lap = time.time()
    if verbose:
        print('Simplices are refined at {:.2f}s'.format(lap-since))
    # 4. for each simplex in the hull compute number of connected components (parallel)
    num_comps_ray = [ray_label_simplex.remote(grid_ray, simplex, thresh) for simplex in simplices]
    num_comps = ray.get(num_comps_ray) 
    lap = time.time()
    if verbose:
        print('Simplices are labelled at {:.2f}s'.format(lap-since))
        
    outdict['num_comps'] = num_comps
    
    del num_comps_ray
    outdict['coplanar'] = None
    if flag_lift_label:
        
        # 5. lift the labels from simplices to points (parallel)
        if lift_grid_size == meshsize:
            lift_grid_ray = grid_ray
            lift_grid = grid
        else:
            lift_grid = makegridnd(lift_grid_size, dimensions) # we lift labels to a constant mesh 
            lift_grid_ray = ray.put(lift_grid)
            
        inside_ray = [ray_lift_label.remote(grid_ray, lift_grid_ray,
                                            simplex, label) for simplex, label in zip(simplices, num_comps)]
        inside = ray.get(inside_ray)
        
        coplanar = [item[1] for item in inside]
        outdict['coplanar'] = coplanar
        lap = time.time()
        
        if verbose:
            print('Labels are lifted at {:.2f}s'.format(lap-since))

            print('Total {}/{} coplanar simplices'.format(Counter(coplanar)[0],len(simplices)))

        phase = np.zeros(lift_grid.shape[1])
        for i,label in zip(inside,num_comps):
            if i[1]==1:
                phase[i[0]] = label
        phase = phase.reshape(1,-1)
        output = np.vstack((lift_grid,phase))
        index = ['Phi_'+str(i) for i in range(1, output.shape[0])]
        index.append('label')
        output = pd.DataFrame(data = output,index=index)
        
        del lift_grid_ray, inside_ray, inside
        
    else:
        output = []
        
    outdict['output'] = output    
    lap = time.time()
    print('Computation took {:.2f}s'.format(lap-since))
    
    # we remove everything we don't need
    del grid_ray  
    
    # finish computation and exit ray
    ray.shutdown()

    return outdict
