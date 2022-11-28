import cv2 as cv
import find_matching_points
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from PIL import Image, ImageOps
import itertools

start = 50
end = 52
num_of_nodes = 100
nbhd_size = 100
mesh_thresh = 100
pr = 1.7

masks = find_matching_points.get_masks(start, end)
mpts = find_matching_points.find_matching_points(start, end, masks, num_of_nodes, nbhd_size, mesh_thresh, pr)

#IMPORTANT: There was a problem with barycentric coord calculation which has now been fixed

#Mesh class
#On init
#Eats image, image index and matching points
#Contains structure
#Image, matching points specific to image, internal k, external k, momentum
#Mesh info (nodes, simplices, etc)

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

        self.img_size = [5000, 5000]
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

            Mesh0 = meshes[pair[0]-50]
            Mesh1 = meshes[pair[1]-50]

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

            Mesh0 = self.meshes[pair[0]-50]
            Mesh1 = self.meshes[pair[1]-50]

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

            Mesh0 = self.meshes[pair[0]-50]
            Mesh1 = self.meshes[pair[1]-50]

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

            Mesh0 = self.meshes[pair[0]-50]
            Mesh1 = self.meshes[pair[1]-50]

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

'''
mpts = {(110, 111): np.array([[[ 306.06122449,  408.08163265],
        [ 350.06121826,  400.0816345]],
       [[ 306.06122449,  510.10204082],
        [ 270.06121826,  509.10205078]],
       [[ 306.06122449,  612.12244898],
        [ 270.06121826,  607.12243652]]])}
'''

#check force lengths too

opt = Optimizer([50,51,52], mpts)
#bary1 = opt.find_bary(mpts)
opt.optimize()

from scipy.interpolate import LinearNDInterpolator

def matches_to_flow(nodes, offsets):
    
    mesh_els = 50
    
    x = nodes[:,0]
    y = nodes[:,1]
    
    x_off = offsets[:,0]
    y_off = offsets[:,1]

    #finding x flow field

    X = np.linspace(0, 5000, 5000)
    Y = np.linspace(0, 5000, 5000)

    X, Y = np.meshgrid(X, Y)

    interp1 = LinearNDInterpolator(list(zip(x, y)), x_off)
    X_OFF = interp1(X, Y)
    print('test1')
    
    #finding y flow field

    interp2 = LinearNDInterpolator(list(zip(x, y)), y_off)
    Y_OFF = interp2(X, Y)
    print('test2')
    
    #plt.pcolormesh(X, Y, X_OFF, shading='auto')
    #plt.plot(x, y, "ok", label="input point")
    #plt.colorbar()
    #plt.axis("equal")
    #plt.show()

    #plt.quiver(x, y, x_off, y_off)
    #plt.show()

    #plt.pcolormesh(X, Y, Y_OFF, shading='auto')
    #plt.plot(x, y, "ok", label="input point")
    #plt.colorbar()
    #plt.axis("equal")
    #plt.show()

    return X, Y, X_OFF, Y_OFF

#applying map to images

X_maps = []
Y_maps = []

for i in range(len(opt.meshes)):

    offsets = opt.meshes[i].mesh_nodes - opt.meshes[i].tri.points
    X, Y, X_OFF, Y_OFF = matches_to_flow(opt.meshes[i].tri.points, offsets)

    map_x = np.add(X_OFF.T, X)
    print('moge')
    map_y = np.add(Y_OFF.T, Y)
    print('my beloved')
            
    MAP_X = map_x.astype('float32')
    MAP_Y = map_y.astype('float32')
    
    X_maps.append(MAP_X)
    Y_maps.append(MAP_Y)

load_dir = '/mnt/onetb/yuuka/aligned_sections/'

def load_img(index):
    
    img = Image.open(load_dir + f'Fixed_Layer{index}.tif')
    
    return img

img_50 = np.array(load_img(50).resize((5000,5000)))
img_51 = np.array(load_img(51).resize((5000,5000)))
img_52 = np.array(load_img(52).resize((5000,5000)))

mapped_img_50 = cv.remap(img_50, X_maps[0], Y_maps[0], cv.INTER_LINEAR)
mapped_img_51 = cv.remap(img_51, X_maps[1], Y_maps[1], cv.INTER_LINEAR)
mapped_img_52 = cv.remap(img_52, X_maps[2], Y_maps[2], cv.INTER_LINEAR)

plt.figure(figsize=(15,10))
plt.imshow(mapped_img_50[1000:2000,1000:2000], cmap = plt.cm.Greys_r)
plt.figure(figsize=(15,10))
plt.imshow(mapped_img_51[1000:2000,1000:2000], cmap = plt.cm.Greys_r)
plt.figure(figsize=(15,10))
plt.imshow(mapped_img_52[1000:2000,1000:2000], cmap = plt.cm.Greys_r)

plt.figure(figsize=(15,10))
plt.imshow(mapped_img_50, cmap = plt.cm.Greys_r)
plt.figure(figsize=(15,10))
plt.imshow(mapped_img_51, cmap = plt.cm.Greys_r)
plt.figure(figsize=(15,10))
plt.imshow(mapped_img_52, cmap = plt.cm.Greys_r)