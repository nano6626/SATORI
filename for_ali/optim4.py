import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import itertools
import multiprocess as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from PIL import Image, ImageOps
import itertools
import time
import os

#code that runs in parallel

class Mesh():

    def __init__(self, img_size, index, mesh_size, k_internal, k_external, momentum):

        self.k_internal = k_internal
        self.k_external = k_external
        #self.step_size = 0.01
        #self.step_scaling = 1.1
        self.momentum = momentum
        self.mesh_size = mesh_size
        self.index = index

        self.mesh_nodes, self.mesh_simplices, self.neighbours, self.tri = self.generate_mesh(img_size)
        self.resting_spring_lengths = self.find_resting_spring_lengths()
        self.previous_step = np.zeros((len(self.mesh_nodes), 2))
        self.prev_mesh_nodes = None

    def generate_mesh(self, img_size):
        '''eats an image, spits out the mesh nodes and simplices'''

        row, col = img_size

        step_col = col // self.mesh_size
        step_row = row // self.mesh_size

        col_ind = np.linspace(0, col, step_col)
        row_ind = np.linspace(0, row, step_row)

        #nodes = (list(itertools.product(col_ind, row_ind)))
        #nodes += (list(itertools.product(row_ind, col_ind)))

        nodes = []
        for i in col_ind:
            for j in row_ind:
                nodes.append([i,j])

        nodes = np.array(nodes)

        tri = Delaunay(nodes)
        simplices = tri.simplices
        indptr, indices = tri.vertex_neighbor_vertices
        neighbours = [indices[indptr[k]:indptr[k+1]] for k in range(len(nodes))]
        neighbours = np.array(neighbours)

        return nodes, simplices, neighbours, tri

    def point_to_barycentric(self, points):
        '''when given a point, finds the triangle which it belongs to, as well as barycentric weights, taken from adi peleg's pipeline'''

        p = points.copy()
        p[p < 0] = 0.01
        simplex_indices = self.tri.find_simplex(p)
        assert not np.any(simplex_indices == -1)

        #http://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
        X = self.tri.transform[simplex_indices, :2]
        Y = points - self.tri.transform[simplex_indices, 2]
        b = np.einsum('ijk,ik->ij', X, Y)
        pt_indices = self.tri.simplices[simplex_indices].astype(np.uint32)
        barys = np.c_[b, 1 - b.sum(axis=1)]

        return self.tri.simplices[simplex_indices].astype(np.uint32), barys

    def barycentric_to_point(self, simplex, weights):
        '''given a simplex index and barycentric weights, returns the coordinate'''

        coords = self.mesh_nodes[simplex]

        return np.sum(np.array([coords[i] * weights[i] for i in range(3)]), axis = 0)

    def find_distances(self, node_index):
        '''eats the mesh node positions and triangles, returns sum of normed distances'''

        #look at node, look at its neighbours
        #find coordinate of node and neighbours
        #subtract coordinates (neighbour - node)
        #this gives a signed distance

        node = self.mesh_nodes[node_index]
        neighbours = self.mesh_nodes[self.neighbours[node_index]]
        
        distances = np.array([neighbours[i] - node for i in range(len(neighbours))])

        return distances #distances at that particular node, used to find forces

    def find_resting_spring_lengths(self):

        #store resting lengths as a list of length values corresponding to neighbours at node

        resting_lengths = [self.find_distances(i) for i in range(len(self.mesh_nodes))]

        return np.array(resting_lengths)
    
    def find_internal_node_force_and_energy(self, node_index):
        '''calculates net force on a single node'''
        
        distances = self.find_distances(node_index)
        resting_lengths = self.resting_spring_lengths[node_index]

        forces = []
        energies = []
        for i in range(len(distances)):
            magnitude = np.sqrt(distances[i][0]**2 + distances[i][1]**2)
            resting_magnitude = np.sqrt(resting_lengths[i][0]**2 + resting_lengths[i][1]**2)
            force_x = distances[i][0] * (1 - ((resting_magnitude) / (magnitude))) * self.k_internal
            force_y = distances[i][1] * (1 - ((resting_magnitude) / (magnitude))) * self.k_internal
            forces.append([force_x, force_y])

            energy = (force_x**2 + force_y**2) / (2*self.k_internal)
            energies.append(energy)
        
        return np.sum(np.array(forces), axis = 0), np.sum(np.array(energies), axis = 0)

    def energy_minimization(self, external_force, step_size):

        def find_force_and_energy():

            internal_force = []
            internal_energy = []

            for i in range(len(self.mesh_nodes)):

                internal = self.find_internal_node_force_and_energy(i)
                internal_force.append(internal[0])
                internal_energy.append(internal[1])

            internal_force = np.array(internal_force)
            internal_energy = np.array(internal_energy)
 
            external_energy = np.array([(i[0]**2 + i[1]**2) / (2 * self.k_external) for i in external_force])

            nodal_force = internal_force + external_force
            nodal_energy = internal_energy + external_energy
            total_energy = np.sum(nodal_energy)
            max_energy = np.max(nodal_energy)

            return nodal_force, total_energy, max_energy

        nodal_force, total_energy, max_energy = find_force_and_energy()
        normalized_nodal_force = nodal_force / np.max(np.abs(nodal_force))
        mesh_config = self.mesh_nodes.copy()
        self.prev_mesh_nodes = self.mesh_nodes.copy()

        for i in range(len(self.mesh_nodes)):

            mesh_config[i] += normalized_nodal_force[i] * step_size + self.momentum * self.previous_step[i]
            self.previous_step[i] = normalized_nodal_force[i] * step_size + self.momentum * self.previous_step[i]

        self.mesh_nodes = mesh_config

        return total_energy, self.mesh_nodes
    
    def undo_prev_step(self):
        
        self.mesh_nodes = self.prev_mesh_nodes.copy()

class Optimizer():

    def __init__(self, indices, mpts):

        self.img_size = [20000, 20000]
        self.mesh_size = 100
        self.k_internal = 0.1
        self.k_external = 0.05
        self.momentum = 0.5
        self.indices = indices
        self.mpts = mpts
        self.prev_energy = None
        
        self.global_forces, self.barycentric_representation, self.meshes = self.generate_mesh()

    def generate_mesh(self):

        #create a mesh for each image
        
        global_forces = {}
        meshes = []
        for index in self.indices:
            mesh_index = Mesh(self.img_size, index, self.mesh_size, self.k_internal, self.k_external, self.momentum)
            meshes.append(mesh_index)
            num_of_nodes = len(mesh_index.mesh_nodes)
            global_forces[index] = np.full((num_of_nodes, 2), np.array([0,0], dtype = np.float64))

        #calculate forces and energies

        barycentric_representation = dict.fromkeys(self.mpts.keys(),[])

        for pair in list(self.mpts.keys()):
            
            bary_coords = []

            Mesh0 = meshes[pair[0]-0]
            Mesh1 = meshes[pair[1]-0]

            matches = self.mpts[pair]

            for match in matches:

                force = self.k_external * (match[0] - match[1])

                simplex0, weights0 = Mesh0.point_to_barycentric(np.array([match[0]]))
                simplex1, weights1 = Mesh1.point_to_barycentric(np.array([match[1]]))

                global_forces[pair[0]][simplex0[0]] += [w * force*-1 for w in weights0[0]]
                global_forces[pair[1]][simplex1[0]] += [w * force for w in weights1[0]]

                bary_coords.append([simplex0, weights0, simplex1, weights1])
                
            barycentric_representation[pair] = bary_coords

        return global_forces, barycentric_representation, meshes

    def recalculate_global_forces(self):

        global_forces = {}

        for i in self.global_forces.keys():
            global_forces[i] = np.zeros_like(self.global_forces[i])

        for pair in list(self.mpts.keys()):

            Mesh0 = self.meshes[pair[0]-0]
            Mesh1 = self.meshes[pair[1]-0]

            bary_matches = self.barycentric_representation[pair]

            for bary_match in bary_matches:

                simplex0 = bary_match[0][0]
                weights0 = bary_match[1][0]
                simplex1 = bary_match[2][0]
                weights1 = bary_match[3][0]

                match0 = Mesh0.barycentric_to_point(simplex0, weights0)
                match1 = Mesh1.barycentric_to_point(simplex1, weights1)

                force = (match0 - match1) * self.k_external

                global_forces[pair[0]][simplex0] += [force*weights0[i]*-1 for i in range(3)]
                global_forces[pair[1]][simplex1] += [force*weights1[i] for i in range(3)]

        return global_forces

    def optimize_mesh(self, step_size):

        #on init, optimizer creates meshes on each image, converts forces to global force
        #and finds barycentric representation of matching points
        #to iterate
        #on first iteration, use the global forces, optimize the meshes one by one using external force
        #update the global forces using barycentric representation and new mesh node positions
        #iterate

        energies = []
        mesh_debug = []

        for mesh in self.meshes:

            ind = mesh.index
            energy, new_mesh = mesh.energy_minimization(self.global_forces[ind], step_size)
            energies.append(energy)
            mesh_debug.append(new_mesh)
            self.global_forces = self.recalculate_global_forces()

        #self.global_forces = self.recalculate_global_forces()
        #print(self.global_forces)

        return energies, mesh_debug
    
    def undo_step(self):
        
        for mesh in self.meshes:
            
            mesh.undo_prev_step()
            
    def find_bary(self, dense_mpts):
        
        barycentric_representation = dict.fromkeys(self.mpts.keys(),[])

        for pair in list(self.mpts.keys()):
            
            bary_coords = []

            Mesh0 = self.meshes[pair[0]-0]
            Mesh1 = self.meshes[pair[1]-0]

            matches = self.mpts[pair]

            for match in matches:

                force = self.k_external * (match[0] - match[1])

                simplex0, weights0 = Mesh0.point_to_barycentric(np.array([match[0]]))
                simplex1, weights1 = Mesh1.point_to_barycentric(np.array([match[1]]))

                bary_coords.append([simplex0, weights0, simplex1, weights1])
                
            barycentric_representation[pair] = bary_coords

        return barycentric_representation
    
    def bary_to_coords(self, barycentric_rep):
        
        coordinate_representation = {}
        
        for pair in list(self.mpts.keys()):
                
            source_target_pairs = []

            Mesh0 = self.meshes[pair[0]-0]
            Mesh1 = self.meshes[pair[1]-0]

            bary_matches = barycentric_rep[pair]

            for bary_match in bary_matches:

                simplex0 = bary_match[0][0]
                weights0 = bary_match[1][0]
                simplex1 = bary_match[2][0]
                weights1 = bary_match[3][0]

                match0 = Mesh0.barycentric_to_point(simplex0, weights0)
                match1 = Mesh1.barycentric_to_point(simplex1, weights1)
                
                source_target_pair = np.array([[match0[0], match0[1]],[match1[0], match1[1]]])
                source_target_pairs.append(source_target_pair)
                
            source_target_pairs = np.array(source_target_pairs)
                
            coordinate_representation[pair[0], pair[1]] = source_target_pairs
        
        return coordinate_representation

    def optimize(self):

        num_of_iters = 250
        step_size = 0.01

        for i in range(num_of_iters):
            
            step_size *= 1.1
            if step_size > 1:
                step_size = 1
                
            if self.prev_energy == None or i < 10:

                energies, mesh_debug = self.optimize_mesh(step_size)
                print(i, np.sum(energies))
                self.prev_energy = np.sum(energies)
                plt.scatter(i, np.sum(energies))
                
            else:
                
                energies, mesh_debug = self.optimize_mesh(step_size)
                
                while np.sum(energies) > self.prev_energy:
                    
                    step_size *= 0.1
                    energies, mesh_debug = self.optimize_mesh(step_size)
                    if np.sum(energies) > self.prev_energy:
                        self.undo_step()
                    print(i, np.sum(energies), 'too high')
                    #print(self.prev_energy)
                    
                print(i, np.sum(energies))
                self.prev_energy = np.sum(energies)
                plt.scatter(i, np.sum(energies))
                
            #energies, mesh_debug = self.optimize_mesh(step_size)
            #print(np.sum(energies))
            #print(self.prev_energy)
            #plt.scatter(i, np.sum(energies))

            #if i%10 == 0:

                #plt.plot(mesh_debug[0][:,0], mesh_debug[0][:,1], 'o')
                #plt.show()

                #plt.plot(mesh_debug[1][:,0], mesh_debug[1][:,1], 'o')
                #plt.show()

        plt.show()
        
    def reoptimize(self, mpts_new):

        self.barycentric_representation = mpts_new
        self.global_forces = self.recalculate_global_forces()
        self.momentum = np.zeros_like(self.momentum)

        num_of_iters = 250
        step_size = 0.01

        for i in range(num_of_iters):

            step_size *= 1.1
            if step_size > 1:
                step_size = 1

            if self.prev_energy == None or i < 50:

                energies, mesh_debug = self.optimize_mesh(step_size)
                print(i, np.sum(energies))
                self.prev_energy = np.sum(energies)
                plt.scatter(i, np.sum(energies))

            else:
                
                energies, mesh_debug = self.optimize_mesh(step_size)
                
                while np.sum(energies) > self.prev_energy:
                    
                    step_size *= 0.1
                    energies, mesh_debug = self.optimize_mesh(step_size)
                    if np.sum(energies) > self.prev_energy:
                        self.undo_step()
                    print(i, np.sum(energies), 'too high')
                    #print(self.prev_energy)
                    
                print(i, np.sum(energies))
                self.prev_energy = np.sum(energies)
                plt.scatter(i, np.sum(energies))

        plt.show()

#testing with parallelism (50 iters)
def opti_mesh(x):
    
    mesh, global_forces, step_size = x

    ind = mesh.index
    energy, new_mesh = mesh.energy_minimization(global_forces[ind], step_size)

    return mesh, energy

if __name__ == '__main__':

    print(mp.cpu_count(), 'number of cpu cores available')
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    print(ncpus)

    start = 0
    end = 24
    num_of_nodes = 200
    nbhd_size = 25
    mesh_thresh = 100
    pr = 1.7
    
    mpts = np.load('/home/chandoki/projects/def-mzhen/chandoki/rough_alignment/mpts_fullres_25.npy', allow_pickle = True)[()]
    arr = [i for i in range(start, end + 1)]
    opt = Optimizer(arr, mpts)
    bary1 = opt.find_bary(mpts)
    
    num_of_iters = 250
    step_size = 0.01
    processes = int(ncpus)
    prev_output = None
    min_step = 0.01

    begin = time.time()
    
    for i in range(num_of_iters):
        
        if step_size < min_step:
            
            break
        
        step_size *= 1.1
        if step_size > 1:
            step_size = 1
            
        if opt.prev_energy == None or i < 50:
            
            p = mp.Pool(processes)
            output = p.map(opti_mesh, [(mesh, opt.global_forces, step_size) for mesh in opt.meshes])
            output = np.array(output)
            opt.meshes = output[:,0]
            energies = output[:,1]
            opt.global_forces = opt.recalculate_global_forces()
            print(i, np.sum(energies))
            opt.prev_energy = np.sum(energies)
            prev_output = output.copy()
            plt.scatter(i, np.sum(energies))
             
        else:
            
            p = mp.Pool(processes)
            output = p.map(opti_mesh, [(mesh, opt.global_forces, step_size) for mesh in opt.meshes])
            output = np.array(output)
            
            if np.sum(output[:,1]) < opt.prev_energy:
                
                opt.meshes = output[:,0]
                energies = output[:,1]
                opt.global_forces = opt.recalculate_global_forces()
                print(i, np.sum(energies))
                opt.prev_energy = np.sum(energies)
                prev_output = output.copy()
                plt.scatter(i, np.sum(energies))
                
            else:
                
                while np.sum(output[:,1]) >= opt.prev_energy:
                    
                    if step_size < min_step:
                        
                        output = prev_output.copy()
                        
                        break
                    
                    step_size *= 0.1
                    p = mp.Pool(processes)
                    output = p.map(opti_mesh, [(mesh, opt.global_forces, step_size) for mesh in opt.meshes])
                    output = np.array(output)
                    print(i, np.sum(output[:,1]), 'too high')
                        
                opt.meshes = output[:,0]
                energies = output[:,1]
                opt.global_forces = opt.recalculate_global_forces()
                print(i, np.sum(energies))
                opt.prev_energy = np.sum(energies)
                prev_output = output.copy()
                plt.scatter(i, np.sum(energies))
      
    plt.show()
    
    ending = time.time()
    print(ending-begin, 'runtime in s')
    
    post_mpts = opt.bary_to_coords(bary1) #returns matching points post relaxation
    
    relaxed_mesh = {} #saving mesh information into dictionary
    
    for i in range(len(opt.meshes)):
    
        nodes = opt.meshes[i].mesh_nodes
        relaxed_mesh[i] = np.array([opt.meshes[i].mesh_nodes, opt.meshes[i].tri.points])
        
    #saving into npy files
    
    np.save('/home/chandoki/projects/def-mzhen/chandoki/rough_alignment/post_mpts_fullres_25.npy', post_mpts)
    np.save('/home/chandoki/projects/def-mzhen/chandoki/rough_alignment/relaxed_mesh_fullres_25.npy', relaxed_mesh)