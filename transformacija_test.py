import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBList import PDBList
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import cm
import numpy as np


def average_pos(poss):
    avg_X, avg_Y, avg_Z = zip(*poss)
    avg_X = np.mean(avg_X)
    avg_Y = np.mean(avg_Y)
    avg_Z = np.mean(avg_Z)
    return avg_X, avg_Y, avg_Z


def dot_product(vec1, vec2):
    return sum([float(v1) * float(v2) for v1, v2 in zip(vec1, vec2)])


def vec_len(vec):
    return math.sqrt(float(sum([v * v for v in vec])))


def vec_angle(vec1, vec2):
    if vec1 == vec2:
        return 0.0
    angle = dot_product(vec1, vec2) / (vec_len(vec1) * vec_len(vec2))
    angle = math.degrees(math.acos(angle))
    assert 0.0 <= angle <= 360.0
    return angle


def calc_side_chain_vector(protres_name, protres_atoms, suppress_warnings=False):
    # composed of two vectors
    # 1. from average position of 'N', 'C' and 'O' atoms to 'CA'
    # 2. average vector from 'CA' to all other atoms except ('N', 'C' and 'O')

    # 1.
    avg_backbone = []
    CA_atom_coords = []
    avg_side_chain = []
    for a_id, coords in protres_atoms.items():
        if a_id in ['N', 'C', 'O']:
            if a_id != 'O':
                avg_backbone.append(coords)
        elif a_id == 'CA':
            CA_atom_coords.append(coords)
        else:
            if a_id == 'CB':
                avg_side_chain.append(coords)

    if len(CA_atom_coords) != 1:
        if not suppress_warnings:
            print("Warning, no CA atom in:", protres_name)
        return None, None

    assert len(CA_atom_coords) == 1
    CA_atom_coords = CA_atom_coords[0]
    CA_pos_X, CA_pos_Y, CA_pos_Z = CA_atom_coords
    backbone_X, backbone_Y, backbone_Z = average_pos(avg_backbone)

    vec_1 = (
        CA_pos_X - backbone_X, CA_pos_Y - backbone_Y, CA_pos_Z - backbone_Z)

    if protres_name == 'G':
        assert len(avg_side_chain) == 0
        return vec_1, CA_atom_coords
    else:
        assert len(avg_side_chain) > 0

    # 2.   
    side_X, side_Y, side_Z = average_pos(avg_side_chain)
    vec_2 = (side_X - CA_pos_X, side_Y - CA_pos_Y, side_Z - CA_pos_Z)

    # angle between the two vectors has to be less than 90
    # A . B = |A| * |B| * cos(angle)
    # cos(angle) = A.B / (|A|*|B|)
    # angle = arccos(A.B / (|A|*|B|))
##    angle_deg = vec_angle(vec_1, vec_2)
##
##    if angle_deg > 180.0:
##        angle_deg = angle_deg - 360.0
##    if angle_deg >= 70:
##        if not suppress_warnings:
##            print("Warning: high angle (%s) between CA-CB and bacbone-CA vectors in %s" % (angle_deg, protres_name))
##        return None, None
##
##    assert angle_deg < 70.0

    # average of the two vectors
    vec = tuple(v1 + v2 for v1, v2 in zip(vec_1, vec_2))

    return vec, CA_atom_coords


def pravokotna_proj(n, tocka):
    d = tocka.dot(n)/np.linalg.norm(n)
    return tocka - d * (n/np.linalg.norm(n))

structure_id = "5w21"

# read it
filename = "fa/5w21.pdb"
parser = PDBParser()
structure = parser.get_structure(structure_id, filename)


# print basic info
resolution = structure.header['resolution']
keywords = structure.header['keywords']

atoms = {}
s = 0
for model in structure:
    for chain in model:
        for residue in chain:
            s += 1
            for atom in residue:
                atoms[atom.id] = atom.get_coord()

            a = atoms['CA']
            b = np.array(calc_side_chain_vector('', atoms, suppress_warnings=False)[0])
            c1 = atoms['C']
            c2 = atoms['N']
            if (b-a).dot(c2-c1) == 0:
                c = a + (c2-c1)
            else:
                c1_p = pravokotna_proj(b-a, c1)
                c2_p =pravokotna_proj(b-a, c2)
                c = a + (c2_p - c1_p)
            d = a + np.cross((b-a), (c-a))


            u = ((d-a)/(np.linalg.norm(d-a)))
            v = ((c-a)/np.linalg.norm(c-a))
            w =((b-a)/np.linalg.norm(b-a))
            u_m = np.append(u,0)
            v_m = np.append(v,0)
            w_m = np.append(w,0)

            m = np.vstack((u_m, v_m, w_m, np.array([0,0,0,1])))
            a_t = m.dot(np.append(a,1))
            m[:,3] = -a_t

            t = np.empty((0,3))
            tn = np.empty((0,4))
            for id_a, coord in atoms.items():
                t = np.vstack((t, coord))
                new_a = m.dot(np.append(coord,1))
                tn = np.vstack((tn, new_a))

            fig = plt.figure(figsize=plt.figaspect(0.5))

            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.plot(t[:,0], t[:,1], t[:,2], 'ko')
            b_ = a + (b-a)*0.1
            c_ = a + (c-a)*2
            ax.plot([a[0], b_[0]], [a[1], b_[1]], [a[2], b_[2]], 'r')
            ax.plot([a[0], c_[0]], [a[1], c_[1]], [a[2], c_[2]], 'b')
            #ax.quiver([a[0]], [a[1]], [a[2]], [b[0]], [b[1]], [b[2]])
 

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.plot(tn[:,0], tn[:,1], tn[:,2], 'ko')
            an = m.dot(np.append(a,1))
            bn = m.dot(np.append(b,1))
            itmax = np.array([max(point[i] for point in tn) for i in range(3)])
            itmin = np.array([min(point[i] for point in tn) for i in range(3)])

            size_x = np.array([itmax[0] - itmin[0], 0, 0])
            size_y = np.array([0,itmax[1] - itmin[1], 0])
            size_z = np.array([0,0,itmax[2] - itmin[2]])

            tocke = np.array([itmin])
            tocke = np.vstack((tocke, itmin + size_x))
            tocke = np.vstack((tocke, itmax - size_z))
            tocke = np.vstack((tocke, itmin + size_y))
            tocke = np.vstack((tocke, itmin + size_z))
            tocke = np.vstack((tocke, itmax - size_y))
            tocke = np.vstack((tocke, itmax))
            tocke = np.vstack((tocke, itmax - size_x))

            #ax.plot([el[0] for el in tn], [el[1] for el in tn], [el[2] for el in tn], 'ko')

            # plot vertices
            ax.scatter3D(tocke[:, 0], tocke[:, 1], tocke[:, 2])

            # list of sides' polygons of figure
            verts = [[tocke[0],tocke[1],tocke[2],tocke[3]],
             [tocke[4],tocke[5],tocke[6],tocke[7]], 
             [tocke[0],tocke[1],tocke[5],tocke[4]], 
             [tocke[2],tocke[3],tocke[7],tocke[6]], 
             [tocke[1],tocke[2],tocke[6],tocke[5]],
             [tocke[4],tocke[7],tocke[3],tocke[0]], 
             [tocke[2],tocke[3],tocke[7],tocke[6]]]

            # plot sides
            ax.add_collection3d(Poly3DCollection(verts, facecolors='lightcyan', linewidths=1, edgecolors='dodgerblue', alpha=.25))

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')


            plt.show()
