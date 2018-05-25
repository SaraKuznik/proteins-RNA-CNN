import random
import shutil
import os

from Bio.PDB.PDBParser import PDBParser

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import numpy as np
from scipy.spatial import distance, cKDTree
import math





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

    if protres_name == 'GLY':
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


def get_protein_and_rna(structure):
    protein = []
    protein_names = []
    rna = []
    rna_names = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname().strip()
                atoms = {}
                for atom in residue:
                    atoms[atom.id] = atom.get_coord()

                if res_name in ['A', 'C', 'G', 'U']:
                    rna.append(atoms)
                    rna_names.append(res_name)
                if ((res_name in [ 'ALA', 'ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','ILE','LEU','LYS','MET',
                                 'PHE','PRO','SER','THR','TRP','TYR', 'VAL'])
                        and ({'CA', 'CB', 'C', 'N'}.issubset(set(atoms.keys())))):

                    protein.append(atoms)
                    protein_names.append(res_name)
    return protein, rna, protein_names, rna_names

def pravokotna_proj(n, tocka_r, tocka):
    d = (tocka-tocka_r).dot(n/np.linalg.norm(n))
    return tocka - d * (n/np.linalg.norm(n))

def plot_transformation(protein, rna, protein_names, structure_id, add_min_dist=False):
    for atoms_prot, protein_name in zip(protein, protein_names):
        min_dist = 1000
        closest = -1
        for i, atoms_rna in enumerate(rna):
            dist = np.min(distance.cdist(np.array(list(atoms_prot.values())), np.array(list(atoms_rna.values()))))
            if dist < min_dist:
                min_dist = dist
                closest = i

        a = atoms_prot['CA']
        b = np.array(calc_side_chain_vector(protein_name, atoms_prot, suppress_warnings=False)[0])
        c1 = atoms_prot['C']
        c2 = atoms_prot['N']
        if (b-a).dot(c2-c1) == 0:
            c = a + (c2-c1)
        else:
            c1_p = pravokotna_proj(b-a, a, c1)
            c2_p =pravokotna_proj(b-a, a, c2)
            print((c1_p-c2_p).dot(b-a))
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

        m2 = np.column_stack((u_m, v_m, w_m, np.array([0,0,0,1])))
        m2[:,3] = np.append(a,1)

        t = np.empty((0,3))
        tn = np.empty((0,4))
        for id_a, coord in atoms_prot.items():
            t = np.vstack((t, coord))
            new_a = m.dot(np.append(coord,1))
            tn = np.vstack((tn, new_a))

        t_rna = np.empty((0,3))
        tn_rna = np.empty((0,4))
        for id_a, coord in rna[closest].items():
            t_rna = np.vstack((t_rna, coord))
            new_a = m.dot(np.append(coord,1))
            tn_rna = np.vstack((tn_rna, new_a))


        fig = plt.figure(figsize=plt.figaspect(0.5))


        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot(tn[:,0], tn[:,1], tn[:,2], 'ko')

        ax.plot(tn_rna[:,0], tn_rna[:,1], tn_rna[:,2], 'mo')

        itmax = np.amax(tn, axis=0)[:3]
        itmin = np.amin(tn, axis=0)[:3]

        if add_min_dist:
            itmax +=(min_dist)
            itmin -=(min_dist)

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


        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot(t[:,0], t[:,1], t[:,2], 'ko')
        # b_ = a + (b-a)*0.1
        # c_ = a + (c-a)*2
        # ax.plot([a[0], b_[0]], [a[1], b_[1]], [a[2], b_[2]], 'r')
        # ax.plot([a[0], c_[0]], [a[1], c_[1]], [a[2], c_[2]], 'b')
        #ax.quiver([a[0]], [a[1]], [a[2]], [b[0]], [b[1]], [b[2]])

        ax.plot(t_rna[:,0], t_rna[:,1], t_rna[:,2], 'mo')


        tocke_ = np.empty((0,3))
        for coord in tocke:
                new_t = m2.dot(np.append(coord,1))[:3]
                tocke_ = np.vstack((tocke_, new_t))

        # list of sides' polygons of figure
        verts_ = [[tocke_[0],tocke_[1],tocke_[2],tocke_[3]],
         [tocke_[4],tocke_[5],tocke_[6],tocke_[7]],
         [tocke_[0],tocke_[1],tocke_[5],tocke_[4]],
         [tocke_[2],tocke_[3],tocke_[7],tocke_[6]],
         [tocke_[1],tocke_[2],tocke_[6],tocke_[5]],
         [tocke_[4],tocke_[7],tocke_[3],tocke_[0]],
         [tocke_[2],tocke_[3],tocke_[7],tocke_[6]]]


        print(np.linalg.norm(tocke[0] - tocke[1]), np.linalg.norm(tocke_[0] - tocke_[1]))

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts_, facecolors='lightcyan', linewidths=1, edgecolors='dodgerblue', alpha=.25))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        fig.suptitle(structure_id)

        plt.show()


def plot_structure(protein, rna, protein_names):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.gca(projection='3d')


    for i, (atoms_prot, protein_name) in enumerate(zip(protein, protein_names)):
        a = atoms_prot['CA']
        b = np.array(calc_side_chain_vector(protein_name, atoms_prot, suppress_warnings=False)[0])
        c1 = atoms_prot['C']
        c2 = atoms_prot['N']
        if (b - a).dot(c2 - c1) == 0:
            c = a + (c2 - c1)
        else:
            c1_p = pravokotna_proj(b - a, a, c1)
            c2_p = pravokotna_proj(b - a, a, c2)
            #print((c1_p - c2_p).dot(b - a))
            c = a + (c2_p - c1_p)

        d = a + np.cross((b - a), (c - a))

        u = ((d - a) / (np.linalg.norm(d - a)))
        v = ((c - a) / np.linalg.norm(c - a))
        w = ((b - a) / np.linalg.norm(b - a))
        u_m = np.append(u, 0)
        v_m = np.append(v, 0)
        w_m = np.append(w, 0)

        m = np.vstack((u_m, v_m, w_m, np.array([0, 0, 0, 1])))
        a_t = m.dot(np.append(a, 1))
        m[:, 3] = -a_t

        m2 = np.column_stack((u_m, v_m, w_m, np.array([0, 0, 0, 1])))
        m2[:, 3] = np.append(a, 1)

        t = np.empty((0, 3))
        tn = np.empty((0, 4))
        for id_a, coord in atoms_prot.items():
            t = np.vstack((t, coord))
            new_a = m.dot(np.append(coord, 1))
            tn = np.vstack((tn, new_a))

        itmax = np.array([max(point[i] for point in tn) for i in range(3)])
        itmin = np.array([min(point[i] for point in tn) for i in range(3)])

        size_x = np.array([itmax[0] - itmin[0], 0, 0])
        size_y = np.array([0, itmax[1] - itmin[1], 0])
        size_z = np.array([0, 0, itmax[2] - itmin[2]])

        tocke = np.array([itmin])
        tocke = np.vstack((tocke, itmin + size_x))
        tocke = np.vstack((tocke, itmax - size_z))
        tocke = np.vstack((tocke, itmin + size_y))
        tocke = np.vstack((tocke, itmin + size_z))
        tocke = np.vstack((tocke, itmax - size_y))
        tocke = np.vstack((tocke, itmax))
        tocke = np.vstack((tocke, itmax - size_x))

        tocke_ = np.empty((0, 3))
        for coord in tocke:
            new_t = m2.dot(np.append(coord, 1))[:3]
            tocke_ = np.vstack((tocke_, new_t))



        ax.plot(t[:, 0], t[:, 1], t[:, 2], marker='o', linestyle='None')

        #ax.plot([a[0]], [a[1]], [a[2]], marker='o', linestyle='solid')
        #b_ = a + (b - a) * 0.1
        #c_ = a + (c - a) * 2
        #ax.plot([a[0], b_[0]], [a[1], b_[1]], [a[2], b_[2]], 'r')
        #ax.plot([a[0], c_[0]], [a[1], c_[1]], [a[2], c_[2]], 'b')
        # ax.quiver([a[0]], [a[1]], [a[2]], [b[0]], [b[1]], [b[2]])

        #ax.plot(t_rna[:, 0], t_rna[:, 1], t_rna[:, 2], 'mo')

        # list of sides' polygons of figure
        verts_ = [[tocke_[0], tocke_[1], tocke_[2], tocke_[3]],
                  [tocke_[4], tocke_[5], tocke_[6], tocke_[7]],
                  [tocke_[0], tocke_[1], tocke_[5], tocke_[4]],
                  [tocke_[2], tocke_[3], tocke_[7], tocke_[6]],
                  [tocke_[1], tocke_[2], tocke_[6], tocke_[5]],
                  [tocke_[4], tocke_[7], tocke_[3], tocke_[0]],
                  [tocke_[2], tocke_[3], tocke_[7], tocke_[6]]]

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts_, facecolors='lightcyan', linewidths=1, edgecolors='dodgerblue', alpha=.25))

    print(i)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def bb_sizes_histogram(pdb_structures):
    x_sizes = []
    y_sizes = []
    z_sizes = []

    for j, structure in enumerate(pdb_structures):
        protein, rna, protein_names, rna_names = get_protein_and_rna(structure)
        for atoms_prot, protein_name in zip(protein, protein_names):
            min_dist = 100
            for i, atoms_rna in enumerate(rna):
                dist = np.min(distance.cdist(np.array(list(atoms_prot.values())), np.array(list(atoms_rna.values()))))
                if dist < min_dist:
                    min_dist = dist

            a = atoms_prot['CA']
            b = np.array(calc_side_chain_vector(protein_name, atoms_prot, suppress_warnings=False)[0])
            c1 = atoms_prot['C']
            c2 = atoms_prot['N']

            if (b - a).dot(c2 - c1) == 0:
                c = a + (c2 - c1)
            else:
                c1_p = pravokotna_proj(b - a, a, c1)
                c2_p = pravokotna_proj(b - a, a, c2)

                c = a + (c2_p - c1_p)

            d = a + np.cross((b - a), (c - a))

            u = ((d - a) / (np.linalg.norm(d - a)))
            v = ((c - a) / np.linalg.norm(c - a))
            w = ((b - a) / np.linalg.norm(b - a))
            u_m = np.append(u, 0)
            v_m = np.append(v, 0)
            w_m = np.append(w, 0)

            m = np.vstack((u_m, v_m, w_m, np.array([0, 0, 0, 1])))
            a_t = m.dot(np.append(a, 1))
            m[:, 3] = -a_t

            tn = np.empty((0, 4))
            for id_a, coord in atoms_prot.items():
                new_a = m.dot(np.append(coord, 1))
                tn = np.vstack((tn, new_a))

            itmax = np.amax(tn, axis=0)[:3]
            itmin = np.amin(tn, axis=0)[:3]

            itmax += (min_dist / 2)
            itmin -= (min_dist / 2)

            x_sizes.append(itmax[0] - itmin[0])
            y_sizes.append(itmax[1] - itmin[1])
            z_sizes.append(itmax[2] - itmin[2])


    plt.hist(x_sizes, bins=100)
    plt.xticks(range(0, 101, 5))
    plt.xlabel('Bounding box x edge length')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(y_sizes, bins=100)
    plt.xticks(range(0, 101, 5))
    plt.xlabel('Bounding box y edge length')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(z_sizes, bins=100)
    plt.xticks(range(0, 101, 5))
    plt.xlabel('Bounding box z edge length')
    plt.ylabel('Frequency')
    plt.show()


def histogram(pdb_structures):
    distances = []
    str_len = len(pdb_structures)
    for j, structure in enumerate(pdb_structures):
        protein, rna, protein_names, rna_names = get_protein_and_rna(structure)
        for atoms_prot in protein:
            min_dist = 100
            for i, atoms_rna in enumerate(rna):
                dist = np.min(distance.cdist(np.array(list(atoms_prot.values())), np.array(list(atoms_rna.values()))))
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)
        print(str(j+1) + '/' + str(str_len))

    plt.hist(distances, bins=100)
    plt.xticks(range(0,101,5))
    plt.xlabel('Distance from aminoacide to closest nucleotid')
    plt.ylabel('Frequency')
    plt.show()


def clean_data():
    parser = PDBParser()
    path = "./pdb structures clean"
    os.makedirs(path)
    dat = open('pdb_structures_clean.txt', 'w')
    for (dirpath, dirnames, filenames) in os.walk('./pdb structures/'):
        for file in filenames:
            structure_id = file.split('.')[0][3:]
            structure = parser.get_structure(structure_id, 'pdb structures/' + file)
            protein, rna, protein_names, rna_names = get_protein_and_rna(structure)
            if 1 < len(protein) <= 500 and len(rna) > 0:
                shutil.copy('pdb structures/' + file, path)
                print(structure_id, file=dat)
    dat.close()



def get_structures():
    parser = PDBParser()
    pdb_structures = {}
    for (dirpath, dirnames, filenames) in os.walk('./pdb structures clean/'):
        for file in filenames:
            structure_id = file.split('.')[0][3:]
            structure = parser.get_structure(structure_id, 'pdb structures clean/' + file)
            pdb_structures[structure_id] = structure


    return pdb_structures




if __name__ == "__main__":

    pdb_structures = get_structures()
    print('PDB structures Done')

    # bb_sizes_histogram(pdb_structures.values())
    # histogram(pdb_structures.values())


    structure_id = random.choice(list(pdb_structures.keys()))
    #structure_id = '2ab4'
    structure = pdb_structures[structure_id]
    protein, rna, protein_names, rna_names = get_protein_and_rna(structure)
    plot_transformation(protein, rna, protein_names, structure_id)


