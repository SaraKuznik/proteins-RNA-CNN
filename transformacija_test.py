import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBList import PDBList
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')

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
    angle_deg = vec_angle(vec_1, vec_2)

    if angle_deg > 180.0:
        angle_deg = angle_deg - 360.0
    if angle_deg >= 70:
        if not suppress_warnings:
            print("Warning: high angle (%s) between CA-CB and bacbone-CA vectors in %s" % (angle_deg, protres_name))
        return None, None

    assert angle_deg < 70.0

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
for model in structure:
    for chain in model:
        for residue in chain:
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




ax.plot(t[:,0], t[:,1], t[:,2], 'ro')
ax.plot(tn[:,0], tn[:,1], tn[:,2], 'o')
ax.grid(True)
fig.savefig('trans_test.png')
