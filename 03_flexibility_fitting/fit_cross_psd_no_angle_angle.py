#!/usr/bin/python3

"""
Description:
This code performs flexibility parameters calculations for all structures. 
Since the procedure for a structure containing at least one rotatable dihedral 
is different from a structure containing all non-rotatable or hindered 
dihedrals, we first check if the structure has any rotatable dihedral and then
follow the related procedure. In this code we use average equilibrium values
for each bond, angle, and dihedral types. Also, the bond-bond cross term is 
included among calculated potentials. At the first part of the code we 
extract lists of bonds, angles, and dihedrals and corresponding lists of  
bonds, angles, and dihedral types. Then we start with dihedral mode 
selection and after that we do least-squares fitting to extract the 
flexibility parameters of the selected modes for rotatable dihedrals. Next, we 
compute R-squared and RMSE for rotatable dihedrals. Then, we do least-squares 
fitting to extract training flexibility parameters of bonds, angles, and 
non-rotatable/hindered dihedrals and compute the corresponding R-squared and 
RMSE values. Next, we reinsert rotatable dihedral flexibility parameters and 
compute R-squared and RMSE for complete training set. Finally, we use 
flexibility parameters obtained from the training data set on the geometries 
from the validation dataset to compute R-squared and RMSE for the validation 
dataset.

Input files:
For each structure, we have a folder containing all input files which are text
files and csv files. The text files are lists of bonds, bond types, angles,
angle types, dihedrals, dihedral types, atom types, and optimized structure 
(XYZ file) .For structures having rotatable dihedrals we also have a text file 
having the structure refcode, rotatable dihedral type and selected dihedral 
instance which were used in creating displaced geometries required for single 
point DFT calculations.The csv files contain training and validation 
geometries and forces (training:one csv file having information from AIMD and 
Hessian calculations, two csv files that will be added if there is any 
rotatable dihedral in the structure having the displaced geometries and 
the energy of each displaced geometry from SPDFT calculations respectively, 
validation: one csv file having information from AIMD calculations)

Output files:
The code will print out some information regarding the structure, flexibility
parameters and statistics results (e.g., R-squared, RMSE)
instance.

"""

import numpy as np
import math 
import copy
import glmnet_python
from glmnetCoef import glmnetCoef
import scipy
from glmnetPrint import glmnetPrint;
import time
import sys

def calc_torsion_potential(dphi,m,sign,count_duplicates,dihedral_angles_A_params,dihedral_angles_B_params,dihedral_angles_A_eq_params,dihedral_angles_B_eq_params):
    
    torsion_potential_tot = 0
    angle_threshold = 130*np.pi/180
    K = 2.815891616117388
    
    for i in range(len(dphi)):
        theta_1 = dihedral_angles_A_params[i]
        theta_2 = dihedral_angles_B_params[i]
        theta_eq_1 = dihedral_angles_A_eq_params[i]
        theta_eq_2 = dihedral_angles_B_eq_params[i]
        if theta_eq_1 < angle_threshold and theta_eq_2 < angle_threshold:
            J_ADDT_1 = 1; J_ADDT_2 = 1; J_ADDT_3 = 1; J_ADDT_4 = 1; H_ADDT_1 = 1; H_ADDT_2 = 1; H_ADDT_3 = 1; H_ADDT_4 = 1;
        else:
            cht_1 = math.cos(theta_1/2)
            cht_2 = math.cos(theta_2/2)
            cht_eq_1 = math.cos(theta_eq_1/2)
            cht_eq_2 = math.cos(theta_eq_2/2)
            
            P1_1 = (cht_1 + 3*pow(cht_1,3))/4
            P1_2 = (cht_2 + 3*pow(cht_2,3))/4
            P2_1 = (3*pow(cht_1,2) + pow(cht_1,4))/4
            P2_2 = (3*pow(cht_2,2) + pow(cht_2,4))/4 
            P3_1 = (6*pow(cht_1,3) - 3*pow(cht_1,5) + pow(cht_1,7))/4
            P3_2 = (6*pow(cht_2,3) - 3*pow(cht_2,5) + pow(cht_2,7))/4     
            P4_1 = (10*pow(cht_1,4) - 9*pow(cht_1,6) + 3*pow(cht_1,8))/4
            P4_2 = (10*pow(cht_2,4) - 9*pow(cht_2,6) + 3*pow(cht_2,8))/4                                    
            
            P1_eq_1 = (cht_eq_1 + 3*pow(cht_eq_1,3))/4
            P1_eq_2 = (cht_eq_2 + 3*pow(cht_eq_2,3))/4
            P2_eq_1 = (3*pow(cht_eq_1,2) + pow(cht_eq_1,4))/4
            P2_eq_2 = (3*pow(cht_eq_2,2) + pow(cht_eq_2,4))/4 
            P3_eq_1 = (6*pow(cht_eq_1,3) - 3*pow(cht_eq_1,5) + pow(cht_eq_1,7))/4
            P3_eq_2 = (6*pow(cht_eq_2,3) - 3*pow(cht_eq_2,5) + pow(cht_eq_2,7))/4     
            P4_eq_1 = (10*pow(cht_eq_1,4) - 9*pow(cht_eq_1,6) + 3*pow(cht_eq_1,8))/4
            P4_eq_2 = (10*pow(cht_eq_2,4) - 9*pow(cht_eq_2,6) + 3*pow(cht_eq_2,8))/4
            
            f1_1_ratio = math.tanh(K*P1_1)/math.tanh(K*P1_eq_1)
            f1_2_ratio = math.tanh(K*P1_2)/math.tanh(K*P1_eq_2)
            f2_1_ratio = math.tanh(K*P2_1)/math.tanh(K*P2_eq_1)
            f2_2_ratio = math.tanh(K*P2_2)/math.tanh(K*P2_eq_2)
            f3_1_ratio = math.tanh(K*P3_1)/math.tanh(K*P3_eq_1)
            f3_2_ratio = math.tanh(K*P3_2)/math.tanh(K*P3_eq_2)
            f4_1_ratio = math.tanh(K*P4_1)/math.tanh(K*P4_eq_1)
            f4_2_ratio = math.tanh(K*P4_2)/math.tanh(K*P4_eq_2)
            
            f1_1_ratio_pow2 = f1_1_ratio**2
            f1_2_ratio_pow2 = f1_2_ratio**2
            
            H_ADDT_1 = f1_1_ratio*f1_2_ratio
            H_ADDT_2 = f2_1_ratio*f2_2_ratio
            H_ADDT_3 = f3_1_ratio*f3_2_ratio
            H_ADDT_4 = f4_1_ratio*f4_2_ratio
            
            f2_1_ratio_over_f1_1_ratio = 0; f3_1_ratio_over_f1_1_ratio = 0; f4_1_ratio_over_f2_1_ratio = 0;
            f2_2_ratio_over_f1_2_ratio = 0; f3_2_ratio_over_f1_2_ratio = 0; f4_2_ratio_over_f2_2_ratio = 0;
            
            if abs(f1_1_ratio) > 1e-9:
                f2_1_ratio_over_f1_1_ratio = f2_1_ratio/f1_1_ratio; f3_1_ratio_over_f1_1_ratio = f3_1_ratio/f1_1_ratio;
            if abs(f2_1_ratio) > 1e-9:
                f4_1_ratio_over_f2_1_ratio = f4_1_ratio/f2_1_ratio;
            if abs(f1_2_ratio) > 1e-9:
                f2_2_ratio_over_f1_2_ratio = f2_2_ratio/f1_2_ratio; f3_2_ratio_over_f1_2_ratio = f3_2_ratio/f1_2_ratio;
            if abs(f2_2_ratio) > 1e-9:
                f4_2_ratio_over_f2_2_ratio = f4_2_ratio/f2_2_ratio;
            
            J_ADDT_1 = (1/4)*(f1_1_ratio_pow2 + 1)*(f1_2_ratio_pow2 + 1)
            J_ADDT_2 = (1/4)*(f2_1_ratio_over_f1_1_ratio**2 + f1_1_ratio_pow2)*(f2_2_ratio_over_f1_2_ratio**2 + f1_2_ratio_pow2)
            J_ADDT_3 = (1/4)*(f3_1_ratio_over_f1_1_ratio**2 + f1_1_ratio_pow2)*(f3_2_ratio_over_f1_2_ratio**2 + f1_2_ratio_pow2)
            J_ADDT_4 = (1/4)*(f4_1_ratio_over_f2_1_ratio**2 + f2_1_ratio**2)*(f4_2_ratio_over_f2_2_ratio**2 + f2_2_ratio**2)
         
        if m == 1:
            torsion_potential_tot += (1/count_duplicates[i])*(J_ADDT_1 - H_ADDT_1*math.cos(m*dphi[i]))
        elif m == 2:
            torsion_potential_tot += (1/count_duplicates[i])*(J_ADDT_2 - H_ADDT_2*math.cos(m*dphi[i]))
        elif m == 3:
            torsion_potential_tot += (1/count_duplicates[i])*(J_ADDT_3 - H_ADDT_3*math.cos(m*dphi[i]))
        elif m == 4:
            torsion_potential_tot += (1/count_duplicates[i])*(J_ADDT_4 - H_ADDT_4*math.cos(m*dphi[i]))
        elif m == 5:
            torsion_potential_tot += (1/count_duplicates[i])*sign[i]*(3*math.sin(dphi[i])*H_ADDT_1 - math.sin(3*dphi[i])*H_ADDT_3)/np.sqrt(10)
        elif m == 6:
            torsion_potential_tot += (1/count_duplicates[i])*sign[i]*(2*math.sin(2*dphi[i])*H_ADDT_2 - math.sin(4*dphi[i])*H_ADDT_4)/np.sqrt(5)
        elif m  == 7:
            torsion_potential_tot += (1/count_duplicates[i])*sign[i]*( (math.sin(dphi[i])*H_ADDT_1 + 3*math.sin(3*dphi[i])*H_ADDT_3) - (math.sin(2*dphi[i])*H_ADDT_2 + 2*math.sin(4*dphi[i])*H_ADDT_4))/np.sqrt(15)
        
    return torsion_potential_tot

def calc_torsion_potential_CO(phi,phi_eq,adjusted_c_i,count_duplicates,dihedral_angles_A_params,dihedral_angles_B_params,dihedral_angles_A_eq_params,dihedral_angles_B_eq_params):
    
    torsion_potential_tot = 0
    angle_threshold = 130*np.pi/180
    K = 2.815891616117388
    
    for i in range(len(phi)):
        theta_1 = dihedral_angles_A_params[i]
        theta_2 = dihedral_angles_B_params[i]
        theta_eq_1 = dihedral_angles_A_eq_params[i]
        theta_eq_2 = dihedral_angles_B_eq_params[i]
        if theta_eq_1 < angle_threshold and theta_eq_2 < angle_threshold:
            J_ADDT_1 = 1; J_ADDT_2 = 1; J_ADDT_3 = 1; J_ADDT_4 = 1; H_ADDT_1 = 1; H_ADDT_2 = 1; H_ADDT_3 = 1; H_ADDT_4 = 1;
        else:
            cht_1 = math.cos(theta_1/2)
            cht_2 = math.cos(theta_2/2)
            cht_eq_1 = math.cos(theta_eq_1/2)
            cht_eq_2 = math.cos(theta_eq_2/2)
            
            P1_1 = (cht_1 + 3*pow(cht_1,3))/4
            P1_2 = (cht_2 + 3*pow(cht_2,3))/4
            P2_1 = (3*pow(cht_1,2) + pow(cht_1,4))/4
            P2_2 = (3*pow(cht_2,2) + pow(cht_2,4))/4 
            P3_1 = (6*pow(cht_1,3) - 3*pow(cht_1,5) + pow(cht_1,7))/4
            P3_2 = (6*pow(cht_2,3) - 3*pow(cht_2,5) + pow(cht_2,7))/4     
            P4_1 = (10*pow(cht_1,4) - 9*pow(cht_1,6) + 3*pow(cht_1,8))/4
            P4_2 = (10*pow(cht_2,4) - 9*pow(cht_2,6) + 3*pow(cht_2,8))/4                                    
            
            P1_eq_1 = (cht_eq_1 + 3*pow(cht_eq_1,3))/4
            P1_eq_2 = (cht_eq_2 + 3*pow(cht_eq_2,3))/4
            P2_eq_1 = (3*pow(cht_eq_1,2) + pow(cht_eq_1,4))/4
            P2_eq_2 = (3*pow(cht_eq_2,2) + pow(cht_eq_2,4))/4 
            P3_eq_1 = (6*pow(cht_eq_1,3) - 3*pow(cht_eq_1,5) + pow(cht_eq_1,7))/4
            P3_eq_2 = (6*pow(cht_eq_2,3) - 3*pow(cht_eq_2,5) + pow(cht_eq_2,7))/4     
            P4_eq_1 = (10*pow(cht_eq_1,4) - 9*pow(cht_eq_1,6) + 3*pow(cht_eq_1,8))/4
            P4_eq_2 = (10*pow(cht_eq_2,4) - 9*pow(cht_eq_2,6) + 3*pow(cht_eq_2,8))/4
            
            f1_1_ratio = math.tanh(K*P1_1)/math.tanh(K*P1_eq_1)
            f1_2_ratio = math.tanh(K*P1_2)/math.tanh(K*P1_eq_2)
            f2_1_ratio = math.tanh(K*P2_1)/math.tanh(K*P2_eq_1)
            f2_2_ratio = math.tanh(K*P2_2)/math.tanh(K*P2_eq_2)
            f3_1_ratio = math.tanh(K*P3_1)/math.tanh(K*P3_eq_1)
            f3_2_ratio = math.tanh(K*P3_2)/math.tanh(K*P3_eq_2)
            f4_1_ratio = math.tanh(K*P4_1)/math.tanh(K*P4_eq_1)
            f4_2_ratio = math.tanh(K*P4_2)/math.tanh(K*P4_eq_2)
            
            f1_1_ratio_pow2 = f1_1_ratio**2
            f1_2_ratio_pow2 = f1_2_ratio**2
            
            H_ADDT_1 = f1_1_ratio*f1_2_ratio
            H_ADDT_2 = f2_1_ratio*f2_2_ratio
            H_ADDT_3 = f3_1_ratio*f3_2_ratio
            H_ADDT_4 = f4_1_ratio*f4_2_ratio
            
            f2_1_ratio_over_f1_1_ratio = 0; f3_1_ratio_over_f1_1_ratio = 0; f4_1_ratio_over_f2_1_ratio = 0;
            f2_2_ratio_over_f1_2_ratio = 0; f3_2_ratio_over_f1_2_ratio = 0; f4_2_ratio_over_f2_2_ratio = 0;
            
            if abs(f1_1_ratio) > 1e-9:
                f2_1_ratio_over_f1_1_ratio = f2_1_ratio/f1_1_ratio; f3_1_ratio_over_f1_1_ratio = f3_1_ratio/f1_1_ratio;
            if abs(f2_1_ratio) > 1e-9:
                f4_1_ratio_over_f2_1_ratio = f4_1_ratio/f2_1_ratio;
            if abs(f1_2_ratio) > 1e-9:
                f2_2_ratio_over_f1_2_ratio = f2_2_ratio/f1_2_ratio; f3_2_ratio_over_f1_2_ratio = f3_2_ratio/f1_2_ratio;
            if abs(f2_2_ratio) > 1e-9:
                f4_2_ratio_over_f2_2_ratio = f4_2_ratio/f2_2_ratio;
            
            J_ADDT_1 = (1/4)*(f1_1_ratio_pow2 + 1)*(f1_2_ratio_pow2 + 1)
            J_ADDT_2 = (1/4)*(f2_1_ratio_over_f1_1_ratio**2 + f1_1_ratio_pow2)*(f2_2_ratio_over_f1_2_ratio**2 + f1_2_ratio_pow2)
            J_ADDT_3 = (1/4)*(f3_1_ratio_over_f1_1_ratio**2 + f1_1_ratio_pow2)*(f3_2_ratio_over_f1_2_ratio**2 + f1_2_ratio_pow2)
            J_ADDT_4 = (1/4)*(f4_1_ratio_over_f2_1_ratio**2 + f2_1_ratio**2)*(f4_2_ratio_over_f2_2_ratio**2 + f2_2_ratio**2)
            
        
        for j in range(4):
            m_i = j+1
            if m_i == 1:
                torsion_potential_tot += (1/count_duplicates[i])*adjusted_c_i[j]*(H_ADDT_1*math.cos(m_i*phi[i]) - J_ADDT_1*math.cos(m_i*phi_eq[i]))
            if m_i == 2:
                torsion_potential_tot += (1/count_duplicates[i])*adjusted_c_i[j]*(H_ADDT_2*math.cos(m_i*phi[i]) - J_ADDT_2*math.cos(m_i*phi_eq[i]))
            if m_i == 3:
                torsion_potential_tot += (1/count_duplicates[i])*adjusted_c_i[j]*(H_ADDT_3*math.cos(m_i*phi[i]) - J_ADDT_3*math.cos(m_i*phi_eq[i]))
            if m_i == 4:
                torsion_potential_tot += (1/count_duplicates[i])*adjusted_c_i[j]*(H_ADDT_4*math.cos(m_i*phi[i]) - J_ADDT_4*math.cos(m_i*phi_eq[i]))

    return torsion_potential_tot

def calc_displaced_array(single_geom, atom, displacement_vect):
    displaced_array = np.zeros(single_geom.shape,dtype=float)
    displaced_array = displaced_array + single_geom
    displaced_array[atom-1, :] = single_geom[atom-1, :] + displacement_vect
    return displaced_array
    
def convert_to_base_unit_cell(single_geom,M):
    M_inverse = np.linalg.inv(M)
    B_cartesian_array = np.empty(shape = (0,3),dtype=float)
    for i in range(len(single_geom)):
        row = single_geom[i]
        E_array = row @ M_inverse
        F1 = ((E_array[0]) % 1.0)
        F2 = ((E_array[1]) % 1.0)
        F3 = ((E_array[2]) % 1.0)
        F = np.array([F1, F2, F3])
        B_cartesian = F @ M
        B_cartesian_array = np.append(B_cartesian_array, [B_cartesian],axis= 0)
    return B_cartesian_array

def calc_closest_image_pos(R_A,R_B,M):
    R_AB = R_B - R_A
    M_inverse = np.linalg.inv(M)
    E_array = R_AB @ M_inverse
    F1 = ((E_array[0]+0.5) % 1.0) - 0.5
    F2 = ((E_array[1]+0.5) % 1.0) - 0.5
    F3 = ((E_array[2]+0.5) % 1.0) - 0.5
    F = np.array([F1, F2, F3])
    B_closest_pos = R_A + (F @ M)
    B_direct = B_closest_pos @ M_inverse
    H1 = B_direct[0] - (B_direct[0] % 1.0)
    H2 = B_direct[1] - (B_direct[1] % 1.0)
    H3 = B_direct[2] - (B_direct[2] % 1.0)
    B_image_indice = np.array([H1,H2,H3])
    B_pos_indice = np.array([B_closest_pos,B_image_indice])
    return B_pos_indice

def calc_angle(bond_1_vector,bond_2_vector):
    b2_dot_b1 = np.dot(bond_2_vector,bond_1_vector)
    bond_1_length = np.linalg.norm(bond_1_vector)
    bond_2_length = np.linalg.norm(bond_2_vector)
    cos_theta = b2_dot_b1/(bond_1_length*bond_2_length)
    cos_theta = max(min(cos_theta, 1), -1)
    theta = math.acos(cos_theta) 
    return theta

def calc_dihedral_phi(bond_1_vect,bond_2_vect,bond_3_vect):
            bond_3_cross_bond_2 = np.cross(bond_3_vect,-bond_2_vect)
            bond_2_cross_bond_1 = np.cross(bond_2_vect,-bond_1_vect)
            bond_3_cross_bond_2_length = np.linalg.norm(bond_3_cross_bond_2)
            bond_2_cross_bond_1_length = np.linalg.norm(bond_2_cross_bond_1)
            bond_3_cross_bond_2_dot_bond_2_cross_bond_1 = np.dot(bond_3_cross_bond_2,bond_2_cross_bond_1)
            m_cross_n = np.cross(bond_2_cross_bond_1,bond_3_cross_bond_2)
            bond_2_vect_dot_m_cross_n = np.dot(m_cross_n,bond_2_vect)
            
            if bond_2_vect_dot_m_cross_n == 0:
                sign_phi = 1                
            else:
                sign_phi = bond_2_vect_dot_m_cross_n / abs(bond_2_vect_dot_m_cross_n)
                    
            temp = bond_3_cross_bond_2_dot_bond_2_cross_bond_1 / (bond_3_cross_bond_2_length*bond_2_cross_bond_1_length)
            temp = min(max(-1,temp),1)  
            phi = sign_phi*math.acos(temp)
            return phi

def calc_atoms_and_indices(bonds,angles,dihedrals):
    bond_atoms = []
    for i in range(len(bonds)):
        bond_atoms.append([])
    for i in range(len(bonds)):
        for j in range(len(bonds[i])):
            if any(bonds[i][j][1]):
                A_atom_number = bonds[i][j][2]
                B_atom_number = bonds[i][j][0]
            else:
                A_atom_number = bonds[i][j][0]
                B_atom_number = bonds[i][j][2]
            bond_atoms[i].append([A_atom_number,B_atom_number])
            
    angle_atoms = []
    bond_vector_indices_angle = []
    for i in range(len(angles)):
        angle_atoms.append([])
        bond_vector_indices_angle.append([])
    for i in range(len(angles)):
        for j in range(len(angles[i])):
            angle_j = angles[i][j]
            angle_atoms[i].append([angle_j[0],angle_j[1],angle_j[3]])
            atom_a = angle_j[1]
            atom_b = angle_j[0]
            atom_c = angle_j[3]
            b_vect_1 = []
            for index1, value1 in enumerate(bond_atoms):
                Flag = True
                for index2, value2 in enumerate(value1):
                    if value2 == [atom_b, atom_a] or value2 == [atom_a, atom_b]:
                        b_vect_1.append((index1, index2))
                        Flag = False
                        break
                if Flag == False:
                    break
            
            b_vect_2 = []
            for index1, value1 in enumerate(bond_atoms):
                Flag = True
                for index2, value2 in enumerate(value1):
                    if value2 == [atom_b, atom_c] or value2 == [atom_c, atom_b]:
                        b_vect_2.append((index1, index2))
                        Flag = False
                        break
                if Flag == False:
                    break
            
            bond_vector_indices_angle[i].append(([int(x2) for ind1, x1 in enumerate(b_vect_1) for ind2, x2 in enumerate(x1)] + [int(x2) for ind1, x1 in enumerate(b_vect_2) for ind2, x2 in enumerate(x1)]))  
                 
    dihedral_atoms = []   
    bond_vector_indices_dihedral = []
    for i in range(len(dihedrals)):
        dihedral_atoms.append([])
        bond_vector_indices_dihedral.append([])
    for i in range(len(dihedrals)):
        for j in range(len(dihedrals[i])):
            dihedral_j = dihedrals[i][j]
            dihedral_atoms[i].append([dihedral_j[0],dihedral_j[2],dihedral_j[4],dihedral_j[6]])
            atom_a = dihedral_j[0]
            atom_b = dihedral_j[2]
            atom_c = dihedral_j[4]
            atom_d = dihedral_j[6]
            
            b_vect_1 = []
            for index1, value1 in enumerate(bond_atoms):
                Flag = True
                for index2, value2 in enumerate(value1):
                    if value2 == [atom_b, atom_a] or value2 == [atom_a, atom_b]:
                        b_vect_1.append((index1, index2))
                        Flag = False
                        break
                if Flag == False:
                    break
            
            b_vect_2 = []
            for index1, value1 in enumerate(bond_atoms):
                Flag = True
                for index2, value2 in enumerate(value1):
                    if value2 == [atom_b, atom_c] or value2 == [atom_c, atom_b]:
                        b_vect_2.append((index1, index2))
                        Flag = False
                        break
                if Flag == False:
                    break
            
            b_vect_3 = []
            for index1, value1 in enumerate(bond_atoms):
                Flag = True
                for index2, value2 in enumerate(value1):
                    if value2 == [atom_c, atom_d] or value2 == [atom_d, atom_c]:
                        b_vect_3.append((index1, index2))
                        Flag = False
                        break
                if Flag == False:
                    break
            
            bond_vector_indices_dihedral[i].append(([int(x2) for ind1, x1 in enumerate(b_vect_1) for ind2, x2 in enumerate(x1)] + [int(x2) for ind1, x1 in enumerate(b_vect_2) for ind2, x2 in enumerate(x1)] + [int(x2) for ind1, x1 in enumerate(b_vect_3) for ind2, x2 in enumerate(x1)]))           

    atoms = [bond_atoms,angle_atoms,dihedral_atoms, bond_vector_indices_angle, bond_vector_indices_dihedral]
    return atoms
 

def calc_params_vect(single_geom,bonds,angles,dihedrals, atoms_and_indices, M):
    bonds_vectors = []
    bonds_lengths = []   
    for i in range(len(bonds)):
        bonds_vectors.append([])
        bonds_lengths.append([])
    for i in range(len(bonds)):
        for j in range(len(bonds[i])):
            if any(bonds[i][j][1]):
                A_atom_number = bonds[i][j][2]
                B_atom_number = bonds[i][j][0]
            else:
                A_atom_number = bonds[i][j][0]
                B_atom_number = bonds[i][j][2]
            R_A = single_geom[A_atom_number-1, :]
            R_B = single_geom[B_atom_number-1, :]
            R_B_pos_indice = calc_closest_image_pos(R_A, R_B, M)
            R_B_close = R_B_pos_indice[0]
            bond_length = np.linalg.norm(R_B_close-R_A)
            bonds_vectors[i].append(R_B_close-R_A)
            bonds_lengths[i].append(bond_length)
      
    angles_theta = []
    angle_bonds_length = []
    for i in range(len(angles)):
        angles_theta.append([])
        angle_bonds_length.append([])
    for i in range(len(angles)):       
        for j in range(len(angles[i])):  
            angle_j = angles[i][j]
            k1 = atoms_and_indices[3][i][j][0]
            m1 = atoms_and_indices[3][i][j][1]
            k2 = atoms_and_indices[3][i][j][2]
            m2 = atoms_and_indices[3][i][j][3]
            if (angle_j[0] == atoms_and_indices[0][k1][m1][0] and angle_j[1] == atoms_and_indices[0][k1][m1][1]):
                bond_1_vector = bonds_vectors[k1][m1]
                
            if (angle_j[0] == atoms_and_indices[0][k1][m1][1] and angle_j[1] == atoms_and_indices[0][k1][m1][0]):
                bond_1_vector = -bonds_vectors[k1][m1]
                
            if (angle_j[0] == atoms_and_indices[0][k2][m2][0] and angle_j[3] == atoms_and_indices[0][k2][m2][1]):
                bond_2_vector = bonds_vectors[k2][m2]
                
            if (angle_j[0] == atoms_and_indices[0][k2][m2][1] and angle_j[3] == atoms_and_indices[0][k2][m2][0]):
                bond_2_vector = -bonds_vectors[k2][m2]
            
            theta = calc_angle(bond_1_vector,bond_2_vector)
            angles_theta[i].append(theta)
            angle_bond_1 = bonds_lengths[k1][m1]
            angle_bond_2 = bonds_lengths[k2][m2]
            angle_bonds_length[i].append([angle_bond_1,angle_bond_2])
 
    dihedrals_phi =  []
    dihedral_angles = []
    for i in range(len(dihedrals)):
        dihedrals_phi.append([])
        dihedral_angles.append([])
    for i in range(len(dihedrals)):
        for j in range(len(dihedrals[i])):
            dihedral_j = dihedrals[i][j]
            k1 = atoms_and_indices[4][i][j][0]
            m1 = atoms_and_indices[4][i][j][1]
            k2 = atoms_and_indices[4][i][j][2]
            m2 = atoms_and_indices[4][i][j][3]
            k3 = atoms_and_indices[4][i][j][4]
            m3 = atoms_and_indices[4][i][j][5]
            if (dihedral_j[0] == atoms_and_indices[0][k1][m1][0] and dihedral_j[2] == atoms_and_indices[0][k1][m1][1]):
                bond_1_vect = bonds_vectors[k1][m1]
            if (dihedral_j[0] == atoms_and_indices[0][k1][m1][1] and dihedral_j[2] == atoms_and_indices[0][k1][m1][0]):
                bond_1_vect = -bonds_vectors[k1][m1]
                
            if (dihedral_j[2] == atoms_and_indices[0][k2][m2][0] and dihedral_j[4] == atoms_and_indices[0][k2][m2][1]):
                bond_2_vect = bonds_vectors[k2][m2]
            if (dihedral_j[2] == atoms_and_indices[0][k2][m2][1] and dihedral_j[4] == atoms_and_indices[0][k2][m2][0]):
                bond_2_vect = -bonds_vectors[k2][m2]
                
            if (dihedral_j[4] == atoms_and_indices[0][k3][m3][0] and dihedral_j[6] == atoms_and_indices[0][k3][m3][1]):
                bond_3_vect = bonds_vectors[k3][m3]
            if (dihedral_j[4] == atoms_and_indices[0][k3][m3][1] and dihedral_j[6] == atoms_and_indices[0][k3][m3][0]):
                bond_3_vect = -bonds_vectors[k3][m3]
                        
            phi = calc_dihedral_phi(bond_1_vect,bond_2_vect,bond_3_vect)
            dihedrals_phi[i].append(phi)
            dihedral_angle_1 = calc_angle(-bond_1_vect,bond_2_vect)
            dihedral_angle_2 = calc_angle(-bond_2_vect,bond_3_vect)
            dihedral_angles[i].append([dihedral_angle_1,dihedral_angle_2])
    
    params = [bonds_vectors,bonds_lengths,angles_theta,dihedrals_phi,dihedral_angles,angle_bonds_length]
    return params

def calc_delta_stretch_potential(affected_bonds, bond_params_SG, bond_params_displaced_SG, equil_bond_params, counting_duplicates, atom):
    delta_stretch_potential = np.zeros(shape = (len(bond_params_SG), 1), dtype=float)

    for index, affected_bonds in enumerate(affected_bonds[atom-1]):
        i = affected_bonds[0]
        j = affected_bonds[1]
        stretch_potential_SG = (1/counting_duplicates[i][j])*(1/2)*(bond_params_SG[i][j] - equil_bond_params[i][j])**2
        stretch_potential_SG_displaced = (1/counting_duplicates[i][j])*(1/2)*(bond_params_displaced_SG[i][j] - equil_bond_params[i][j])**2
        delta_stretch_potential[i] += (stretch_potential_SG - stretch_potential_SG_displaced)*10000

    return delta_stretch_potential

def calc_delta_bend_potential(affected_angles, angle_params_SG, angle_params_SG_displaced, equil_angle_params, angle_type, atom):
    delta_bend_potential = np.zeros(shape = (len(equil_angle_params), 1), dtype=float)
    for index, affected_angle in enumerate(affected_angles[atom-1]):
        i = affected_angle[0]
        j = affected_angle[1]
        if angle_type[i] == 0:
            
            if (abs(equil_angle_params[i][j] - np.pi) + abs(angle_params_SG[i][j] - np.pi)) < 1e-6:
                bend_potential_SG_i = 0
            else:
                cos_term_1_SG = 2*(math.cos(angle_params_SG[i][j]) - math.cos(equil_angle_params[i][j]))**2
                sin_term_1_SG = 3*((math.sin(equil_angle_params[i][j]))**2)
                sin_term_2_SG = (math.sin(angle_params_SG[i][j]))**2
                tanh_term_1_SG = math.tanh(2*math.sin(angle_params_SG[i][j]/2))
                tanh_term_2_SG = math.tanh(2*math.sin(equil_angle_params[i][j]/2))
                bend_potential_SG_i = (cos_term_1_SG)/((sin_term_1_SG*(tanh_term_1_SG/tanh_term_2_SG))+sin_term_2_SG)
    
            
            if (abs(equil_angle_params[i][j] - np.pi) + abs(angle_params_SG_displaced[i][j] - np.pi)) < 1e-6:
                bend_potential_SG_displaced_i = 0
            else:
                cos_term_1_SGD = 2*(math.cos(angle_params_SG_displaced[i][j]) - math.cos(equil_angle_params[i][j]))**2
                sin_term_1_SGD = 3*((math.sin(equil_angle_params[i][j]))**2)
                sin_term_2_SGD = (math.sin(angle_params_SG_displaced[i][j]))**2
                tanh_term_1_SGD = math.tanh(2*math.sin(angle_params_SG_displaced[i][j]/2))
                tanh_term_2_SGD = math.tanh(2*math.sin(equil_angle_params[i][j]/2))
                bend_potential_SG_displaced_i = (cos_term_1_SGD)/((sin_term_1_SGD*(tanh_term_1_SGD/tanh_term_2_SGD))+sin_term_2_SGD)
                
            delta_bend_potential[i] += (bend_potential_SG_i - bend_potential_SG_displaced_i)*10000
    delta_bend_potential_final = [element for element, flag in zip(delta_bend_potential, angle_type) if flag == 0]
                         
    return delta_bend_potential_final

def calc_delta_bond_bond_potential(affected_angles, angle_bond_params_SG, angle_bond_params_SGD ,equil_angle_bond_params, atom):
    delta_bond_bond_potential = np.zeros(shape = (len(equil_angle_bond_params), 1), dtype=float)
    for index, affected_angle in enumerate(affected_angles[atom-1]):
        i = affected_angle[0]
        j = affected_angle[1]
        term_1_SG = angle_bond_params_SG[i][j][0] - equil_angle_bond_params[i][j][0]
        term_2_SG = angle_bond_params_SG[i][j][1] - equil_angle_bond_params[i][j][1]
        bond_bond_potential_SG_i = term_1_SG*term_2_SG
        
        term_1_SGD = angle_bond_params_SGD[i][j][0] - equil_angle_bond_params[i][j][0]
        term_2_SGD = angle_bond_params_SGD[i][j][1] - equil_angle_bond_params[i][j][1]
        bond_bond_potential_SGD_i = term_1_SGD*term_2_SGD
        
        delta_bond_bond_potential[i] += (bond_bond_potential_SG_i - bond_bond_potential_SGD_i)*10000
    
    return delta_bond_bond_potential

def calc_delta_bond_angle_potential(affected_angles, angle_bond_params_SG, angle_bond_params_SGD, equil_angle_bond_params, angle_params_SG, angle_params_SGD, equil_angle_params, atom):
    delta_bond_angle_potential = np.zeros(shape = (len(equil_angle_bond_params), 1), dtype=float)
    for index, affected_angle in enumerate(affected_angles[atom-1]):
        i = affected_angle[0]
        j = affected_angle[1]
        r1_dev_SG = angle_bond_params_SG[i][j][0] - equil_angle_bond_params[i][j][0]
        r2_dev_SG = angle_bond_params_SG[i][j][1] - equil_angle_bond_params[i][j][1]
        theta_dev_SG = angle_params_SG[i][j] - equil_angle_params[i][j]
        bond_angle_potential_SG_i = (r1_dev_SG + r2_dev_SG) * theta_dev_SG

        r1_dev_SGD = angle_bond_params_SGD[i][j][0] - equil_angle_bond_params[i][j][0]
        r2_dev_SGD = angle_bond_params_SGD[i][j][1] - equil_angle_bond_params[i][j][1]
        theta_dev_SGD = angle_params_SGD[i][j] - equil_angle_params[i][j]
        bond_angle_potential_SGD_i = (r1_dev_SGD + r2_dev_SGD) * theta_dev_SGD

        delta_bond_angle_potential[i] += (bond_angle_potential_SG_i - bond_angle_potential_SGD_i)*10000

    return delta_bond_angle_potential

def calc_delta_angle_angle_potential(affected_angle_pairs, angle_params_SG, angle_params_SGD, equil_angle_params, n_angle_angle_potential, atom):
    delta_angle_angle_potential = np.zeros(shape=(n_angle_angle_potential, 1), dtype=float)
    for (col_aa, i1_aa, j1_aa, i2_aa, j2_aa) in affected_angle_pairs[atom-1]:
        theta1_dev_SG = angle_params_SG[i1_aa][j1_aa] - equil_angle_params[i1_aa][j1_aa]
        theta2_dev_SG = angle_params_SG[i2_aa][j2_aa] - equil_angle_params[i2_aa][j2_aa]
        aa_potential_SG = theta1_dev_SG * theta2_dev_SG

        theta1_dev_SGD = angle_params_SGD[i1_aa][j1_aa] - equil_angle_params[i1_aa][j1_aa]
        theta2_dev_SGD = angle_params_SGD[i2_aa][j2_aa] - equil_angle_params[i2_aa][j2_aa]
        aa_potential_SGD = theta1_dev_SGD * theta2_dev_SGD

        delta_angle_angle_potential[col_aa] += (aa_potential_SG - aa_potential_SGD)*10000

    return delta_angle_angle_potential

def calc_delta_torsion1_potential(affected_dihedral, dihedral_params_SG, dihedral_params_SGD, equil_dihedral_params, mode1_dihedral_type, dihedral_type, counting_duplicates, dihedral_angles_params_SG, dihedral_angles_params_SGD, dihedral_angles_eq_params, adjusted_c_list, atom):
    delta_torsion1_potential = np.zeros(shape = (len(mode1_dihedral_type), 1), dtype=float)
    
    angle_threshold = 130*np.pi/180
    K = 2.815891616117388
    
    for index, affected_dihedral in enumerate(affected_dihedrals[atom-1]):
        i = affected_dihedral[0]
        j = affected_dihedral[1]
        if mode1_dihedral_type[i] == 1 and dihedral_type[i] != 'CO_Rotatable':

            theta_1_SG = dihedral_angles_params_SG[i][j][0]
            theta_2_SG = dihedral_angles_params_SG[i][j][1]
            theta_1_SGD = dihedral_angles_params_SGD[i][j][0]
            theta_2_SGD = dihedral_angles_params_SGD[i][j][1]
            theta_eq_1 = dihedral_angles_eq_params[i][j][0]
            theta_eq_2 = dihedral_angles_eq_params[i][j][1]
            if theta_eq_1 < angle_threshold and theta_eq_2 < angle_threshold:
                H_ADDT_1_SG = 1; H_ADDT_1_SGD = 1; J_ADDT_1_SG = 1; J_ADDT_1_SGD = 1;
            else:
                cht_SG_1 = math.cos(theta_1_SG/2)
                cht_SG_2 = math.cos(theta_2_SG/2)
                cht_SGD_1 = math.cos(theta_1_SGD/2)
                cht_SGD_2 = math.cos(theta_2_SGD/2)
                
                cht_eq_1 = math.cos(theta_eq_1/2)
                cht_eq_2 = math.cos(theta_eq_2/2)
                
                P1_SG_1 = (cht_SG_1 + 3*pow(cht_SG_1,3))/4
                P1_SG_2 = (cht_SG_2 + 3*pow(cht_SG_2,3))/4
				
                P1_SGD_1 = (cht_SGD_1 + 3*pow(cht_SGD_1,3))/4
                P1_SGD_2 = (cht_SGD_2 + 3*pow(cht_SGD_2,3))/4				
                
                P1_eq_1 = (cht_eq_1 + 3*pow(cht_eq_1,3))/4
                P1_eq_2 = (cht_eq_2 + 3*pow(cht_eq_2,3))/4
                
                f1_1_ratio_SG = math.tanh(K*P1_SG_1)/math.tanh(K*P1_eq_1)
                f1_2_ratio_SG = math.tanh(K*P1_SG_2)/math.tanh(K*P1_eq_2)
                
                f1_1_ratio_SGD = math.tanh(K*P1_SGD_1)/math.tanh(K*P1_eq_1)
                f1_2_ratio_SGD = math.tanh(K*P1_SGD_2)/math.tanh(K*P1_eq_2)
                
                H_ADDT_1_SG = f1_1_ratio_SG*f1_2_ratio_SG
                H_ADDT_1_SGD = f1_1_ratio_SGD*f1_2_ratio_SGD
                
                J_ADDT_1_SG = (1/4)*(f1_1_ratio_SG**2 + 1)*(f1_2_ratio_SG**2 + 1)
                J_ADDT_1_SGD = (1/4)*(f1_1_ratio_SGD**2 + 1)*(f1_2_ratio_SGD**2 + 1)
                            
            torsion1_potential_SG_i = (1/counting_duplicates[i][j])*(J_ADDT_1_SG - H_ADDT_1_SG*math.cos(dihedral_params_SG[i][j] - equil_dihedral_params[i][j]))
            torsion1_potential_SGD_i = (1/counting_duplicates[i][j])*(J_ADDT_1_SGD - H_ADDT_1_SGD*math.cos(dihedral_params_SGD[i][j] - equil_dihedral_params[i][j]))
            delta_torsion1_potential[i] += (torsion1_potential_SG_i - torsion1_potential_SGD_i)*10000
        
        elif mode1_dihedral_type[i] == 1 and dihedral_type[i] == 'CO_Rotatable':
            
            theta_1_SG = dihedral_angles_params_SG[i][j][0]
            theta_2_SG = dihedral_angles_params_SG[i][j][1]
            theta_1_SGD = dihedral_angles_params_SGD[i][j][0]
            theta_2_SGD = dihedral_angles_params_SGD[i][j][1]
            theta_eq_1 = dihedral_angles_eq_params[i][j][0]
            theta_eq_2 = dihedral_angles_eq_params[i][j][1]
            if theta_eq_1 < angle_threshold and theta_eq_2 < angle_threshold:
                H_CO_1_SG = 1; H_CO_2_SG = 1; H_CO_3_SG = 1; H_CO_4_SG = 1; H_CO_1_SGD = 1; H_CO_2_SGD = 1; H_CO_3_SGD = 1; H_CO_4_SGD = 1;
                J_CO_1_SG = 1; J_CO_2_SG = 1; J_CO_3_SG = 1; J_CO_4_SG = 1; J_CO_1_SGD = 1; J_CO_2_SGD = 1; J_CO_3_SGD = 1; J_CO_4_SGD = 1;
            else:
                cht_SG_1 = math.cos(theta_1_SG/2)
                cht_SG_2 = math.cos(theta_2_SG/2)
                cht_SGD_1 = math.cos(theta_1_SGD/2)
                cht_SGD_2 = math.cos(theta_2_SGD/2)
                
                cht_eq_1 = math.cos(theta_eq_1/2)
                cht_eq_2 = math.cos(theta_eq_2/2)
                
                P1_SG_1 = (cht_SG_1 + 3*pow(cht_SG_1,3))/4
                P1_SG_2 = (cht_SG_2 + 3*pow(cht_SG_2,3))/4
                P2_SG_1 = (3*pow(cht_SG_1,2) + pow(cht_SG_1,4))/4
                P2_SG_2 = (3*pow(cht_SG_2,2) + pow(cht_SG_2,4))/4 
                P3_SG_1 = (6*pow(cht_SG_1,3) - 3*pow(cht_SG_1,5) + pow(cht_SG_1,7))/4
                P3_SG_2 = (6*pow(cht_SG_2,3) - 3*pow(cht_SG_2,5) + pow(cht_SG_2,7))/4     
                P4_SG_1 = (10*pow(cht_SG_1,4) - 9*pow(cht_SG_1,6) + 3*pow(cht_SG_1,8))/4
                P4_SG_2 = (10*pow(cht_SG_2,4) - 9*pow(cht_SG_2,6) + 3*pow(cht_SG_2,8))/4
				
                P1_SGD_1 = (cht_SGD_1 + 3*pow(cht_SGD_1,3))/4
                P1_SGD_2 = (cht_SGD_2 + 3*pow(cht_SGD_2,3))/4
                P2_SGD_1 = (3*pow(cht_SGD_1,2) + pow(cht_SGD_1,4))/4
                P2_SGD_2 = (3*pow(cht_SGD_2,2) + pow(cht_SGD_2,4))/4 
                P3_SGD_1 = (6*pow(cht_SGD_1,3) - 3*pow(cht_SGD_1,5) + pow(cht_SGD_1,7))/4
                P3_SGD_2 = (6*pow(cht_SGD_2,3) - 3*pow(cht_SGD_2,5) + pow(cht_SGD_2,7))/4     
                P4_SGD_1 = (10*pow(cht_SGD_1,4) - 9*pow(cht_SGD_1,6) + 3*pow(cht_SGD_1,8))/4
                P4_SGD_2 = (10*pow(cht_SGD_2,4) - 9*pow(cht_SGD_2,6) + 3*pow(cht_SGD_2,8))/4				
                
                P1_eq_1 = (cht_eq_1 + 3*pow(cht_eq_1,3))/4
                P1_eq_2 = (cht_eq_2 + 3*pow(cht_eq_2,3))/4
                P2_eq_1 = (3*pow(cht_eq_1,2) + pow(cht_eq_1,4))/4
                P2_eq_2 = (3*pow(cht_eq_2,2) + pow(cht_eq_2,4))/4 
                P3_eq_1 = (6*pow(cht_eq_1,3) - 3*pow(cht_eq_1,5) + pow(cht_eq_1,7))/4
                P3_eq_2 = (6*pow(cht_eq_2,3) - 3*pow(cht_eq_2,5) + pow(cht_eq_2,7))/4     
                P4_eq_1 = (10*pow(cht_eq_1,4) - 9*pow(cht_eq_1,6) + 3*pow(cht_eq_1,8))/4
                P4_eq_2 = (10*pow(cht_eq_2,4) - 9*pow(cht_eq_2,6) + 3*pow(cht_eq_2,8))/4
                
                f1_1_ratio_SG = math.tanh(K*P1_SG_1)/math.tanh(K*P1_eq_1)
                f1_2_ratio_SG = math.tanh(K*P1_SG_2)/math.tanh(K*P1_eq_2)
                f2_1_ratio_SG = math.tanh(K*P2_SG_1)/math.tanh(K*P2_eq_1)
                f2_2_ratio_SG = math.tanh(K*P2_SG_2)/math.tanh(K*P2_eq_2)
                f3_1_ratio_SG = math.tanh(K*P3_SG_1)/math.tanh(K*P3_eq_1)
                f3_2_ratio_SG = math.tanh(K*P3_SG_2)/math.tanh(K*P3_eq_2)
                f4_1_ratio_SG = math.tanh(K*P4_SG_1)/math.tanh(K*P4_eq_1)
                f4_2_ratio_SG = math.tanh(K*P4_SG_2)/math.tanh(K*P4_eq_2)
                
                f1_1_ratio_SGD = math.tanh(K*P1_SGD_1)/math.tanh(K*P1_eq_1)
                f1_2_ratio_SGD = math.tanh(K*P1_SGD_2)/math.tanh(K*P1_eq_2)
                f2_1_ratio_SGD = math.tanh(K*P2_SGD_1)/math.tanh(K*P2_eq_1)
                f2_2_ratio_SGD = math.tanh(K*P2_SGD_2)/math.tanh(K*P2_eq_2)
                f3_1_ratio_SGD = math.tanh(K*P3_SGD_1)/math.tanh(K*P3_eq_1)
                f3_2_ratio_SGD = math.tanh(K*P3_SGD_2)/math.tanh(K*P3_eq_2)
                f4_1_ratio_SGD = math.tanh(K*P4_SGD_1)/math.tanh(K*P4_eq_1)
                f4_2_ratio_SGD = math.tanh(K*P4_SGD_2)/math.tanh(K*P4_eq_2)
                
                H_CO_1_SG = f1_1_ratio_SG*f1_2_ratio_SG
                H_CO_2_SG = f2_1_ratio_SG*f2_2_ratio_SG
                H_CO_3_SG = f3_1_ratio_SG*f3_2_ratio_SG
                H_CO_4_SG = f4_1_ratio_SG*f4_2_ratio_SG
                
                H_CO_1_SGD = f1_1_ratio_SGD*f1_2_ratio_SGD
                H_CO_2_SGD = f2_1_ratio_SGD*f2_2_ratio_SGD
                H_CO_3_SGD = f3_1_ratio_SGD*f3_2_ratio_SGD
                H_CO_4_SGD = f4_1_ratio_SGD*f4_2_ratio_SGD
                
                f2_1_ratio_SG_over_f1_1_ratio_SG = 0; f2_2_ratio_SG_over_f1_2_ratio_SG = 0;
                f2_1_ratio_SGD_over_f1_1_ratio_SGD = 0; f2_2_ratio_SGD_over_f1_2_ratio_SGD = 0;
                f3_1_ratio_SG_over_f1_1_ratio_SG = 0; f3_2_ratio_SG_over_f1_2_ratio_SG = 0;
                f3_1_ratio_SGD_over_f1_1_ratio_SGD = 0; f3_2_ratio_SGD_over_f1_2_ratio_SGD = 0;
                f4_1_ratio_SG_over_f2_1_ratio_SG = 0; f4_2_ratio_SG_over_f2_2_ratio_SG = 0;
                f4_1_ratio_SGD_over_f2_1_ratio_SGD = 0; f4_2_ratio_SGD_over_f2_2_ratio_SGD = 0;
                
                if abs(f1_1_ratio_SG) > 1e-9:
                    f2_1_ratio_SG_over_f1_1_ratio_SG = f2_1_ratio_SG/f1_1_ratio_SG;
                    f3_1_ratio_SG_over_f1_1_ratio_SG = f3_1_ratio_SG/f1_1_ratio_SG;
                if abs(f1_2_ratio_SG) > 1e-9:
                    f2_2_ratio_SG_over_f1_2_ratio_SG = f2_2_ratio_SG/f1_2_ratio_SG;
                    f3_2_ratio_SG_over_f1_2_ratio_SG = f3_2_ratio_SG/f1_2_ratio_SG;
                if abs(f2_1_ratio_SG) > 1e-9:
                    f4_1_ratio_SG_over_f2_1_ratio_SG = f4_1_ratio_SG/f2_1_ratio_SG;
                if abs(f2_2_ratio_SG) > 1e-9:
                    f4_2_ratio_SG_over_f2_2_ratio_SG = f4_2_ratio_SG/f2_2_ratio_SG;
                
                if abs(f1_1_ratio_SGD) > 1e-9:
                    f2_1_ratio_SGD_over_f1_1_ratio_SGD = f2_1_ratio_SGD/f1_1_ratio_SGD;
                    f3_1_ratio_SGD_over_f1_1_ratio_SGD = f3_1_ratio_SGD/f1_1_ratio_SGD;
                if abs(f1_2_ratio_SGD) > 1e-9:
                    f2_2_ratio_SGD_over_f1_2_ratio_SGD = f2_2_ratio_SGD/f1_2_ratio_SGD;
                    f3_2_ratio_SGD_over_f1_2_ratio_SGD = f3_2_ratio_SGD/f1_2_ratio_SGD;
                if abs(f2_1_ratio_SGD) > 1e-9:
                    f4_1_ratio_SGD_over_f2_1_ratio_SGD = f4_1_ratio_SGD/f2_1_ratio_SGD;
                if abs(f2_2_ratio_SGD) > 1e-9:
                    f4_2_ratio_SGD_over_f2_2_ratio_SGD = f4_2_ratio_SGD/f2_2_ratio_SGD;
                    
                J_CO_1_SG = (1/4)*(f1_1_ratio_SG**2 + 1)*(f1_2_ratio_SG**2 + 1)
                J_CO_1_SGD = (1/4)*(f1_1_ratio_SGD**2 + 1)*(f1_2_ratio_SGD**2 + 1)
                J_CO_2_SG = (1/4)*(f2_1_ratio_SG_over_f1_1_ratio_SG**2 + f1_1_ratio_SG**2)*(f2_2_ratio_SG_over_f1_2_ratio_SG**2 + f1_2_ratio_SG**2)
                J_CO_2_SGD = (1/4)*(f2_1_ratio_SGD_over_f1_1_ratio_SGD**2 + f1_1_ratio_SGD**2)*(f2_2_ratio_SGD_over_f1_2_ratio_SGD**2 + f1_2_ratio_SGD**2)
                J_CO_3_SG = (1/4)*(f3_1_ratio_SG_over_f1_1_ratio_SG**2 + f1_1_ratio_SG**2)*(f3_2_ratio_SG_over_f1_2_ratio_SG**2 + f1_2_ratio_SG**2)
                J_CO_3_SGD = (1/4)*(f3_1_ratio_SGD_over_f1_1_ratio_SGD**2 + f1_1_ratio_SGD**2)*(f3_2_ratio_SGD_over_f1_2_ratio_SGD**2 + f1_2_ratio_SGD**2)
                J_CO_4_SG = (1/4)*(f4_1_ratio_SG_over_f2_1_ratio_SG**2 + f2_1_ratio_SG**2)*(f4_2_ratio_SG_over_f2_2_ratio_SG**2 + f2_2_ratio_SG**2)
                J_CO_4_SGD = (1/4)*(f4_1_ratio_SGD_over_f2_1_ratio_SGD**2 + f2_1_ratio_SGD**2)*(f4_2_ratio_SGD_over_f2_2_ratio_SGD**2 + f2_2_ratio_SGD**2)
            
            index_CO_type = adjusted_c_list[0].index(i)
            torsion1_potential_SG_i = 0
            torsion1_potential_SGD_i = 0
            adjusted_c_i = adjusted_c_list[2][index_CO_type]
            phi_SG_ij = dihedral_params_SG[i][j]
            phi_SGD_ij = dihedral_params_SGD[i][j]
            phi_eq_ij = equil_dihedral_params[i][j]
            one_over_counting_duplicates_ij = 1/counting_duplicates[i][j]
            for m_index in range(4):
                m_i = m_index+1
                adjusted_c = adjusted_c_i[m_index]
                
                if m_i == 1 and adjusted_c != 0:
                    torsion1_potential_SG_i += (one_over_counting_duplicates_ij)*adjusted_c*(H_CO_1_SG*math.cos(m_i*phi_SG_ij) - J_CO_1_SG*math.cos(m_i*phi_eq_ij))
                    torsion1_potential_SGD_i += (one_over_counting_duplicates_ij)*adjusted_c*(H_CO_1_SGD*math.cos(m_i*phi_SGD_ij) - J_CO_1_SGD*math.cos(m_i*phi_eq_ij))
                if m_i == 2 and adjusted_c != 0:
                    torsion1_potential_SG_i += (one_over_counting_duplicates_ij)*adjusted_c*(H_CO_2_SG*math.cos(m_i*phi_SG_ij) - J_CO_2_SG*math.cos(m_i*phi_eq_ij))
                    torsion1_potential_SGD_i += (one_over_counting_duplicates_ij)*adjusted_c*(H_CO_2_SGD*math.cos(m_i*phi_SGD_ij) - J_CO_2_SGD*math.cos(m_i*phi_eq_ij))
                if m_i == 3 and adjusted_c != 0:
                    torsion1_potential_SG_i += (one_over_counting_duplicates_ij)*adjusted_c*(H_CO_3_SG*math.cos(m_i*phi_SG_ij) - J_CO_3_SG*math.cos(m_i*phi_eq_ij))
                    torsion1_potential_SGD_i += (one_over_counting_duplicates_ij)*adjusted_c*(H_CO_3_SGD*math.cos(m_i*phi_SGD_ij) - J_CO_3_SGD*math.cos(m_i*phi_eq_ij))
                if m_i == 4 and adjusted_c != 0:
                    torsion1_potential_SG_i += (one_over_counting_duplicates_ij)*adjusted_c*(H_CO_4_SG*math.cos(m_i*phi_SG_ij) - J_CO_4_SG*math.cos(m_i*phi_eq_ij))
                    torsion1_potential_SGD_i += (one_over_counting_duplicates_ij)*adjusted_c*(H_CO_4_SGD*math.cos(m_i*phi_SGD_ij) - J_CO_4_SGD*math.cos(m_i*phi_eq_ij))
                    
            delta_torsion1_potential[i] += (torsion1_potential_SG_i - torsion1_potential_SGD_i)*10000
    
    delta_torsion1_potential_final = [element for element, flag in zip(delta_torsion1_potential, mode1_dihedral_type) if flag == 1]
    
    return delta_torsion1_potential_final

def calc_delta_torsion2_potential(affected_dihedral, dihedral_params_SG, dihedral_params_SGD, equil_dihedral_params, mode2_dihedral_type, dihedral_type, counting_duplicates, dihedral_angles_params_SG, dihedral_angles_params_SGD, dihedral_angles_eq_params, atom):
    delta_torsion2_potential = np.zeros(shape = (len(mode2_dihedral_type), 1), dtype=float)
    
    angle_threshold = 130*np.pi/180
    K = 2.815891616117388
    
    for index, affected_dihedral in enumerate(affected_dihedrals[atom-1]):
        i = affected_dihedral[0]
        j = affected_dihedral[1]
        if mode2_dihedral_type[i] == 1 and dihedral_type[i] == 'Rotatable':
            
            theta_1_SG = dihedral_angles_params_SG[i][j][0]
            theta_2_SG = dihedral_angles_params_SG[i][j][1]
            theta_1_SGD = dihedral_angles_params_SGD[i][j][0]
            theta_2_SGD = dihedral_angles_params_SGD[i][j][1]
            theta_eq_1 = dihedral_angles_eq_params[i][j][0]
            theta_eq_2 = dihedral_angles_eq_params[i][j][1]
            if theta_eq_1 < angle_threshold and theta_eq_2 < angle_threshold:
                J_ADDT_2_SG = 1; J_ADDT_2_SGD = 1; H_ADDT_2_SG = 1; H_ADDT_2_SGD = 1;
            else:
                cht_SG_1 = math.cos(theta_1_SG/2)
                cht_SG_2 = math.cos(theta_2_SG/2)
                cht_SGD_1 = math.cos(theta_1_SGD/2)
                cht_SGD_2 = math.cos(theta_2_SGD/2)
                
                cht_eq_1 = math.cos(theta_eq_1/2)
                cht_eq_2 = math.cos(theta_eq_2/2)
                
                P1_SG_1 = (cht_SG_1 + 3*pow(cht_SG_1,3))/4
                P1_SG_2 = (cht_SG_2 + 3*pow(cht_SG_2,3))/4
                P2_SG_1 = (3*pow(cht_SG_1,2) + pow(cht_SG_1,4))/4
                P2_SG_2 = (3*pow(cht_SG_2,2) + pow(cht_SG_2,4))/4 
				
                P1_SGD_1 = (cht_SGD_1 + 3*pow(cht_SGD_1,3))/4
                P1_SGD_2 = (cht_SGD_2 + 3*pow(cht_SGD_2,3))/4
                P2_SGD_1 = (3*pow(cht_SGD_1,2) + pow(cht_SGD_1,4))/4
                P2_SGD_2 = (3*pow(cht_SGD_2,2) + pow(cht_SGD_2,4))/4 				
                
                P1_eq_1 = (cht_eq_1 + 3*pow(cht_eq_1,3))/4
                P1_eq_2 = (cht_eq_2 + 3*pow(cht_eq_2,3))/4
                P2_eq_1 = (3*pow(cht_eq_1,2) + pow(cht_eq_1,4))/4
                P2_eq_2 = (3*pow(cht_eq_2,2) + pow(cht_eq_2,4))/4 
                
                f1_1_ratio_SG = math.tanh(K*P1_SG_1)/math.tanh(K*P1_eq_1)
                f1_2_ratio_SG = math.tanh(K*P1_SG_2)/math.tanh(K*P1_eq_2)
                f2_1_ratio_SG = math.tanh(K*P2_SG_1)/math.tanh(K*P2_eq_1)
                f2_2_ratio_SG = math.tanh(K*P2_SG_2)/math.tanh(K*P2_eq_2)
                
                f1_1_ratio_SGD = math.tanh(K*P1_SGD_1)/math.tanh(K*P1_eq_1)
                f1_2_ratio_SGD = math.tanh(K*P1_SGD_2)/math.tanh(K*P1_eq_2)
                f2_1_ratio_SGD = math.tanh(K*P2_SGD_1)/math.tanh(K*P2_eq_1)
                f2_2_ratio_SGD = math.tanh(K*P2_SGD_2)/math.tanh(K*P2_eq_2)
                
                H_ADDT_2_SG = f2_1_ratio_SG*f2_2_ratio_SG
                H_ADDT_2_SGD = f2_1_ratio_SGD*f2_2_ratio_SGD
                
                f2_1_ratio_SG_over_f1_1_ratio_SG = 0; f2_2_ratio_SG_over_f1_2_ratio_SG = 0;
                f2_1_ratio_SGD_over_f1_1_ratio_SGD = 0; f2_2_ratio_SGD_over_f1_2_ratio_SGD = 0;
                
                if abs(f1_1_ratio_SG) > 1e-9:
                    f2_1_ratio_SG_over_f1_1_ratio_SG = f2_1_ratio_SG/f1_1_ratio_SG;
                if abs(f1_2_ratio_SG) > 1e-9:
                    f2_2_ratio_SG_over_f1_2_ratio_SG = f2_2_ratio_SG/f1_2_ratio_SG;
                if abs(f1_1_ratio_SGD) > 1e-9:
                    f2_1_ratio_SGD_over_f1_1_ratio_SGD = f2_1_ratio_SGD/f1_1_ratio_SGD;
                if abs(f1_2_ratio_SGD) > 1e-9:
                    f2_2_ratio_SGD_over_f1_2_ratio_SGD = f2_2_ratio_SGD/f1_2_ratio_SGD;
                    
                J_ADDT_2_SG = (1/4)*(f2_1_ratio_SG_over_f1_1_ratio_SG**2 + f1_1_ratio_SG**2)*(f2_2_ratio_SG_over_f1_2_ratio_SG**2 + f1_2_ratio_SG**2)
                J_ADDT_2_SGD = (1/4)*(f2_1_ratio_SGD_over_f1_1_ratio_SGD**2 + f1_1_ratio_SGD**2)*(f2_2_ratio_SGD_over_f1_2_ratio_SGD**2 + f1_2_ratio_SGD**2)
                
            torsion2_potential_SG_i = (1/counting_duplicates[i][j])*(J_ADDT_2_SG - H_ADDT_2_SG*math.cos(2 *(dihedral_params_SG[i][j] - equil_dihedral_params[i][j])))
            torsion2_potential_SGD_i = (1/counting_duplicates[i][j])*(J_ADDT_2_SGD - H_ADDT_2_SGD*math.cos(2 *(dihedral_params_SGD[i][j] - equil_dihedral_params[i][j])))
            delta_torsion2_potential[i] += (torsion2_potential_SG_i - torsion2_potential_SGD_i)*10000
        
    delta_torsion2_potential_final = [element for element, flag1, flag2 in zip(delta_torsion2_potential, mode2_dihedral_type, dihedral_type) if flag1 == 1 and flag2 == 'Rotatable']
    return delta_torsion2_potential_final

def calc_delta_torsion3_potential(affected_dihedrals, dihedral_params_SG, dihedral_params_SGD, equil_dihedral_params, mode3_dihedral_type, dihedral_type, counting_duplicates, dihedral_angles_params_SG, dihedral_angles_params_SGD, dihedral_angles_eq_params, atom):
    delta_torsion3_potential = np.zeros(shape = (len(mode3_dihedral_type), 1), dtype=float)
    
    angle_threshold = 130*np.pi/180
    K = 2.815891616117388
    
    for index, affected_dihedral in enumerate(affected_dihedrals[atom-1]):
        i = affected_dihedral[0]
        j = affected_dihedral[1]
        if mode3_dihedral_type[i] == 1 and dihedral_type[i] == 'Rotatable':
            
            theta_1_SG = dihedral_angles_params_SG[i][j][0]
            theta_2_SG = dihedral_angles_params_SG[i][j][1]
            theta_1_SGD = dihedral_angles_params_SGD[i][j][0]
            theta_2_SGD = dihedral_angles_params_SGD[i][j][1]
            theta_eq_1 = dihedral_angles_eq_params[i][j][0]
            theta_eq_2 = dihedral_angles_eq_params[i][j][1]
            if theta_eq_1 < angle_threshold and theta_eq_2 < angle_threshold:
                J_ADDT_3_SG = 1; J_ADDT_3_SGD = 1; H_ADDT_3_SG = 1; H_ADDT_3_SGD = 1;
            else:
                cht_SG_1 = math.cos(theta_1_SG/2)
                cht_SG_2 = math.cos(theta_2_SG/2)
                cht_SGD_1 = math.cos(theta_1_SGD/2)
                cht_SGD_2 = math.cos(theta_2_SGD/2)
                
                cht_eq_1 = math.cos(theta_eq_1/2)
                cht_eq_2 = math.cos(theta_eq_2/2)
                
                P1_SG_1 = (cht_SG_1 + 3*pow(cht_SG_1,3))/4
                P1_SG_2 = (cht_SG_2 + 3*pow(cht_SG_2,3))/4
                P3_SG_1 = (6*pow(cht_SG_1,3) - 3*pow(cht_SG_1,5) + pow(cht_SG_1,7))/4
                P3_SG_2 = (6*pow(cht_SG_2,3) - 3*pow(cht_SG_2,5) + pow(cht_SG_2,7))/4     
				
                P1_SGD_1 = (cht_SGD_1 + 3*pow(cht_SGD_1,3))/4
                P1_SGD_2 = (cht_SGD_2 + 3*pow(cht_SGD_2,3))/4
                P3_SGD_1 = (6*pow(cht_SGD_1,3) - 3*pow(cht_SGD_1,5) + pow(cht_SGD_1,7))/4
                P3_SGD_2 = (6*pow(cht_SGD_2,3) - 3*pow(cht_SGD_2,5) + pow(cht_SGD_2,7))/4     				
                
                P1_eq_1 = (cht_eq_1 + 3*pow(cht_eq_1,3))/4
                P1_eq_2 = (cht_eq_2 + 3*pow(cht_eq_2,3))/4
                P3_eq_1 = (6*pow(cht_eq_1,3) - 3*pow(cht_eq_1,5) + pow(cht_eq_1,7))/4
                P3_eq_2 = (6*pow(cht_eq_2,3) - 3*pow(cht_eq_2,5) + pow(cht_eq_2,7))/4     
                
                f1_1_ratio_SG = math.tanh(K*P1_SG_1)/math.tanh(K*P1_eq_1)
                f1_2_ratio_SG = math.tanh(K*P1_SG_2)/math.tanh(K*P1_eq_2)
                f3_1_ratio_SG = math.tanh(K*P3_SG_1)/math.tanh(K*P3_eq_1)
                f3_2_ratio_SG = math.tanh(K*P3_SG_2)/math.tanh(K*P3_eq_2)
                
                f1_1_ratio_SGD = math.tanh(K*P1_SGD_1)/math.tanh(K*P1_eq_1)
                f1_2_ratio_SGD = math.tanh(K*P1_SGD_2)/math.tanh(K*P1_eq_2)
                f3_1_ratio_SGD = math.tanh(K*P3_SGD_1)/math.tanh(K*P3_eq_1)
                f3_2_ratio_SGD = math.tanh(K*P3_SGD_2)/math.tanh(K*P3_eq_2)
                
                H_ADDT_3_SG = f3_1_ratio_SG*f3_2_ratio_SG
                H_ADDT_3_SGD = f3_1_ratio_SGD*f3_2_ratio_SGD
                
                f3_1_ratio_SG_over_f1_1_ratio_SG = 0; f3_2_ratio_SG_over_f1_2_ratio_SG = 0;
                f3_1_ratio_SGD_over_f1_1_ratio_SGD = 0; f3_2_ratio_SGD_over_f1_2_ratio_SGD = 0;
                
                if abs(f1_1_ratio_SG) > 1e-9:
                    f3_1_ratio_SG_over_f1_1_ratio_SG = f3_1_ratio_SG/f1_1_ratio_SG;
                if abs(f1_2_ratio_SG) > 1e-9:
                    f3_2_ratio_SG_over_f1_2_ratio_SG = f3_2_ratio_SG/f1_2_ratio_SG;
                if abs(f1_1_ratio_SGD) > 1e-9:
                    f3_1_ratio_SGD_over_f1_1_ratio_SGD = f3_1_ratio_SGD/f1_1_ratio_SGD;
                if abs(f1_2_ratio_SGD) > 1e-9:
                    f3_2_ratio_SGD_over_f1_2_ratio_SGD = f3_2_ratio_SGD/f1_2_ratio_SGD;
                
                J_ADDT_3_SG = (1/4)*(f3_1_ratio_SG_over_f1_1_ratio_SG**2 + f1_1_ratio_SG**2)*(f3_2_ratio_SG_over_f1_2_ratio_SG**2 + f1_2_ratio_SG**2)
                J_ADDT_3_SGD = (1/4)*(f3_1_ratio_SGD_over_f1_1_ratio_SGD**2 + f1_1_ratio_SGD**2)*(f3_2_ratio_SGD_over_f1_2_ratio_SGD**2 + f1_2_ratio_SGD**2)
            
            torsion3_potential_SG_i = (1/counting_duplicates[i][j])*(J_ADDT_3_SG - H_ADDT_3_SG*math.cos(3 *(dihedral_params_SG[i][j] - equil_dihedral_params[i][j])))
            torsion3_potential_SGD_i = (1/counting_duplicates[i][j])*(J_ADDT_3_SGD - H_ADDT_3_SGD*math.cos(3 *(dihedral_params_SGD[i][j] - equil_dihedral_params[i][j])))
            delta_torsion3_potential[i] += (torsion3_potential_SG_i - torsion3_potential_SGD_i)*10000
        
    delta_torsion3_potential_final = [element for element, flag1, flag2 in zip(delta_torsion3_potential, mode3_dihedral_type, dihedral_type) if flag1 == 1 and flag2 == 'Rotatable']
    return  delta_torsion3_potential_final

def calc_delta_torsion4_potential(affected_dihedral, dihedral_params_SG, dihedral_params_SGD, equil_dihedral_params, mode4_dihedral_type, dihedral_type, counting_duplicates, dihedral_angles_params_SG, dihedral_angles_params_SGD, dihedral_angles_eq_params,atom):
    delta_torsion4_potential = np.zeros(shape = (len(mode4_dihedral_type), 1), dtype=float)
    
    angle_threshold = 130*np.pi/180
    K = 2.815891616117388
    
    for index, affected_dihedral in enumerate(affected_dihedrals[atom-1]):
        i = affected_dihedral[0]
        j = affected_dihedral[1]
        if mode4_dihedral_type[i] == 1 and dihedral_type[i] == 'Rotatable':
            
            theta_1_SG = dihedral_angles_params_SG[i][j][0]
            theta_2_SG = dihedral_angles_params_SG[i][j][1]
            theta_1_SGD = dihedral_angles_params_SGD[i][j][0]
            theta_2_SGD = dihedral_angles_params_SGD[i][j][1]
            theta_eq_1 = dihedral_angles_eq_params[i][j][0]
            theta_eq_2 = dihedral_angles_eq_params[i][j][1]
            if theta_eq_1 < angle_threshold and theta_eq_2 < angle_threshold:
                J_ADDT_4_SG = 1; J_ADDT_4_SGD = 1; H_ADDT_4_SG = 1; H_ADDT_4_SGD = 1;
            else:
                cht_SG_1 = math.cos(theta_1_SG/2)
                cht_SG_2 = math.cos(theta_2_SG/2)
                cht_SGD_1 = math.cos(theta_1_SGD/2)
                cht_SGD_2 = math.cos(theta_2_SGD/2)
                
                cht_eq_1 = math.cos(theta_eq_1/2)
                cht_eq_2 = math.cos(theta_eq_2/2)
                
                P2_SG_1 = (3*pow(cht_SG_1,2) + pow(cht_SG_1,4))/4
                P2_SG_2 = (3*pow(cht_SG_2,2) + pow(cht_SG_2,4))/4 
                P4_SG_1 = (10*pow(cht_SG_1,4) - 9*pow(cht_SG_1,6) + 3*pow(cht_SG_1,8))/4
                P4_SG_2 = (10*pow(cht_SG_2,4) - 9*pow(cht_SG_2,6) + 3*pow(cht_SG_2,8))/4
				
                P2_SGD_1 = (3*pow(cht_SGD_1,2) + pow(cht_SGD_1,4))/4
                P2_SGD_2 = (3*pow(cht_SGD_2,2) + pow(cht_SGD_2,4))/4 
                P4_SGD_1 = (10*pow(cht_SGD_1,4) - 9*pow(cht_SGD_1,6) + 3*pow(cht_SGD_1,8))/4
                P4_SGD_2 = (10*pow(cht_SGD_2,4) - 9*pow(cht_SGD_2,6) + 3*pow(cht_SGD_2,8))/4				
                
                P2_eq_1 = (3*pow(cht_eq_1,2) + pow(cht_eq_1,4))/4
                P2_eq_2 = (3*pow(cht_eq_2,2) + pow(cht_eq_2,4))/4
                P4_eq_1 = (10*pow(cht_eq_1,4) - 9*pow(cht_eq_1,6) + 3*pow(cht_eq_1,8))/4
                P4_eq_2 = (10*pow(cht_eq_2,4) - 9*pow(cht_eq_2,6) + 3*pow(cht_eq_2,8))/4
                
                f2_1_ratio_SG = math.tanh(K*P2_SG_1)/math.tanh(K*P2_eq_1)
                f2_2_ratio_SG = math.tanh(K*P2_SG_2)/math.tanh(K*P2_eq_2)
                f4_1_ratio_SG = math.tanh(K*P4_SG_1)/math.tanh(K*P4_eq_1)
                f4_2_ratio_SG = math.tanh(K*P4_SG_2)/math.tanh(K*P4_eq_2)
                
                f2_1_ratio_SGD = math.tanh(K*P2_SGD_1)/math.tanh(K*P2_eq_1)
                f2_2_ratio_SGD = math.tanh(K*P2_SGD_2)/math.tanh(K*P2_eq_2)
                f4_1_ratio_SGD = math.tanh(K*P4_SGD_1)/math.tanh(K*P4_eq_1)
                f4_2_ratio_SGD = math.tanh(K*P4_SGD_2)/math.tanh(K*P4_eq_2)
                
                H_ADDT_4_SG = f4_1_ratio_SG*f4_2_ratio_SG
                H_ADDT_4_SGD = f4_1_ratio_SGD*f4_2_ratio_SGD
                
                f4_1_ratio_SG_over_f2_1_ratio_SG = 0; f4_2_ratio_SG_over_f2_2_ratio_SG = 0;
                f4_1_ratio_SGD_over_f2_1_ratio_SGD = 0; f4_2_ratio_SGD_over_f2_2_ratio_SGD = 0;
                
                if abs(f2_1_ratio_SG) > 1e-9:
                    f4_1_ratio_SG_over_f2_1_ratio_SG = f4_1_ratio_SG/f2_1_ratio_SG;
                if abs(f2_2_ratio_SG) > 1e-9:
                    f4_2_ratio_SG_over_f2_2_ratio_SG = f4_2_ratio_SG/f2_2_ratio_SG;   
                if abs(f2_1_ratio_SGD) > 1e-9:
                    f4_1_ratio_SGD_over_f2_1_ratio_SGD = f4_1_ratio_SGD/f2_1_ratio_SGD;
                if abs(f2_2_ratio_SGD) > 1e-9:
                    f4_2_ratio_SGD_over_f2_2_ratio_SGD = f4_2_ratio_SGD/f2_2_ratio_SGD;
                    
                J_ADDT_4_SG = (1/4)*(f4_1_ratio_SG_over_f2_1_ratio_SG**2 + f2_1_ratio_SG**2)*(f4_2_ratio_SG_over_f2_2_ratio_SG**2 + f2_2_ratio_SG**2)
                J_ADDT_4_SGD = (1/4)*(f4_1_ratio_SGD_over_f2_1_ratio_SGD**2 + f2_1_ratio_SGD**2)*(f4_2_ratio_SGD_over_f2_2_ratio_SGD**2 + f2_2_ratio_SGD**2)
                
            torsion4_potential_SG_i = (1/counting_duplicates[i][j])*(J_ADDT_4_SG - H_ADDT_4_SG*math.cos(4 *(dihedral_params_SG[i][j] - equil_dihedral_params[i][j])))
            torsion4_potential_SGD_i = (1/counting_duplicates[i][j])*(J_ADDT_4_SGD - H_ADDT_4_SGD*math.cos(4 *(dihedral_params_SGD[i][j] - equil_dihedral_params[i][j])))
            delta_torsion4_potential[i] += (torsion4_potential_SG_i - torsion4_potential_SGD_i)*10000
    
    delta_torsion4_potential_final = [element for element, flag1, flag2 in zip(delta_torsion4_potential, mode4_dihedral_type, dihedral_type) if flag1 == 1 and flag2 == 'Rotatable']
    return  delta_torsion4_potential_final

def calc_delta_torsion5_potential(affected_dihedrals, dihedral_params_SG, dihedral_params_SGD, equil_dihedral_params, mode5_dihedral_type, dihedral_type, counting_duplicates, sign_dihedral_type, dihedral_angles_params_SG, dihedral_angles_params_SGD, dihedral_angles_eq_params,atom):
    delta_torsion5_potential = np.zeros(shape = (len(mode5_dihedral_type), 1), dtype=float)

    angle_threshold = 130*np.pi/180
    K = 2.815891616117388
    
    for index, affected_dihedral in enumerate(affected_dihedrals[atom-1]):
        i = affected_dihedral[0]
        j = affected_dihedral[1]
        if mode5_dihedral_type[i] == 1 and dihedral_type[i] == 'Rotatable':
            
            theta_1_SG = dihedral_angles_params_SG[i][j][0]
            theta_2_SG = dihedral_angles_params_SG[i][j][1]
            theta_1_SGD = dihedral_angles_params_SGD[i][j][0]
            theta_2_SGD = dihedral_angles_params_SGD[i][j][1]
            theta_eq_1 = dihedral_angles_eq_params[i][j][0]
            theta_eq_2 = dihedral_angles_eq_params[i][j][1]
            if theta_eq_1 < angle_threshold and theta_eq_2 < angle_threshold:
                H_ADDT_1_SG = 1; H_ADDT_1_SGD = 1; H_ADDT_3_SG = 1; H_ADDT_3_SGD = 1;
            else:
                cht_SG_1 = math.cos(theta_1_SG/2)
                cht_SG_2 = math.cos(theta_2_SG/2)
                cht_SGD_1 = math.cos(theta_1_SGD/2)
                cht_SGD_2 = math.cos(theta_2_SGD/2)
                
                cht_eq_1 = math.cos(theta_eq_1/2)
                cht_eq_2 = math.cos(theta_eq_2/2)
                
                P1_SG_1 = (cht_SG_1 + 3*pow(cht_SG_1,3))/4
                P1_SG_2 = (cht_SG_2 + 3*pow(cht_SG_2,3))/4 
                P3_SG_1 = (6*pow(cht_SG_1,3) - 3*pow(cht_SG_1,5) + pow(cht_SG_1,7))/4
                P3_SG_2 = (6*pow(cht_SG_2,3) - 3*pow(cht_SG_2,5) + pow(cht_SG_2,7))/4     
				
                P1_SGD_1 = (cht_SGD_1 + 3*pow(cht_SGD_1,3))/4
                P1_SGD_2 = (cht_SGD_2 + 3*pow(cht_SGD_2,3))/4 
                P3_SGD_1 = (6*pow(cht_SGD_1,3) - 3*pow(cht_SGD_1,5) + pow(cht_SGD_1,7))/4
                P3_SGD_2 = (6*pow(cht_SGD_2,3) - 3*pow(cht_SGD_2,5) + pow(cht_SGD_2,7))/4     				
                
                P1_eq_1 = (cht_eq_1 + 3*pow(cht_eq_1,3))/4
                P1_eq_2 = (cht_eq_2 + 3*pow(cht_eq_2,3))/4
                P3_eq_1 = (6*pow(cht_eq_1,3) - 3*pow(cht_eq_1,5) + pow(cht_eq_1,7))/4
                P3_eq_2 = (6*pow(cht_eq_2,3) - 3*pow(cht_eq_2,5) + pow(cht_eq_2,7))/4     
                
                f1_1_ratio_SG = math.tanh(K*P1_SG_1)/math.tanh(K*P1_eq_1)
                f1_2_ratio_SG = math.tanh(K*P1_SG_2)/math.tanh(K*P1_eq_2)
                f3_1_ratio_SG = math.tanh(K*P3_SG_1)/math.tanh(K*P3_eq_1)
                f3_2_ratio_SG = math.tanh(K*P3_SG_2)/math.tanh(K*P3_eq_2)
                
                f1_1_ratio_SGD = math.tanh(K*P1_SGD_1)/math.tanh(K*P1_eq_1)
                f1_2_ratio_SGD = math.tanh(K*P1_SGD_2)/math.tanh(K*P1_eq_2)
                f3_1_ratio_SGD = math.tanh(K*P3_SGD_1)/math.tanh(K*P3_eq_1)
                f3_2_ratio_SGD = math.tanh(K*P3_SGD_2)/math.tanh(K*P3_eq_2)
                
                H_ADDT_1_SG = f1_1_ratio_SG*f1_2_ratio_SG
                H_ADDT_1_SGD = f1_1_ratio_SGD*f1_2_ratio_SGD
                
                H_ADDT_3_SG = f3_1_ratio_SG*f3_2_ratio_SG
                H_ADDT_3_SGD = f3_1_ratio_SGD*f3_2_ratio_SGD
                
            torsion5_potential_SG_i = (1/counting_duplicates[i][j])*sign_dihedral_type[i][j]*(3*math.sin(dihedral_params_SG[i][j] - equil_dihedral_params[i][j])*H_ADDT_1_SG - math.sin(3*(dihedral_params_SG[i][j] - equil_dihedral_params[i][j]))*H_ADDT_3_SG)/np.sqrt(10)
            torsion5_potential_SGD_i = (1/counting_duplicates[i][j])*sign_dihedral_type[i][j]*(3*math.sin(dihedral_params_SGD[i][j] - equil_dihedral_params[i][j])*H_ADDT_1_SGD - math.sin(3*(dihedral_params_SGD[i][j] - equil_dihedral_params[i][j]))*H_ADDT_3_SGD)/np.sqrt(10)
            delta_torsion5_potential[i] += (torsion5_potential_SG_i - torsion5_potential_SGD_i)*10000
    
    delta_torsion5_potential_final = [element for element, flag1, flag2 in zip(delta_torsion5_potential, mode5_dihedral_type, dihedral_type) if flag1 == 1 and flag2 == 'Rotatable']
    return  delta_torsion5_potential_final

def calc_delta_torsion6_potential(affected_dihedral, dihedral_params_SG, dihedral_params_SGD, equil_dihedral_params, mode6_dihedral_type, dihedral_type, counting_duplicates, sign_dihedral_type, dihedral_angles_params_SG, dihedral_angles_params_SGD, dihedral_angles_eq_params,atom):
    delta_torsion6_potential = np.zeros(shape = (len(mode6_dihedral_type), 1), dtype=float)

    angle_threshold = 130*np.pi/180
    K = 2.815891616117388
    
    for index, affected_dihedral in enumerate(affected_dihedrals[atom-1]):
        i = affected_dihedral[0]
        j = affected_dihedral[1]
        if mode6_dihedral_type[i] == 1 and dihedral_type[i] == 'Rotatable':
            
            theta_1_SG = dihedral_angles_params_SG[i][j][0]
            theta_2_SG = dihedral_angles_params_SG[i][j][1]
            theta_1_SGD = dihedral_angles_params_SGD[i][j][0]
            theta_2_SGD = dihedral_angles_params_SGD[i][j][1]
            theta_eq_1 = dihedral_angles_eq_params[i][j][0]
            theta_eq_2 = dihedral_angles_eq_params[i][j][1]
            if theta_eq_1 < angle_threshold and theta_eq_2 < angle_threshold:
                H_ADDT_2_SG = 1; H_ADDT_2_SGD = 1; H_ADDT_4_SG = 1; H_ADDT_4_SGD = 1;
            else:
                cht_SG_1 = math.cos(theta_1_SG/2)
                cht_SG_2 = math.cos(theta_2_SG/2)
                cht_SGD_1 = math.cos(theta_1_SGD/2)
                cht_SGD_2 = math.cos(theta_2_SGD/2)
                
                cht_eq_1 = math.cos(theta_eq_1/2)
                cht_eq_2 = math.cos(theta_eq_2/2)
                
                P2_SG_1 = (3*pow(cht_SG_1,2) + pow(cht_SG_1,4))/4
                P2_SG_2 = (3*pow(cht_SG_2,2) + pow(cht_SG_2,4))/4     
                P4_SG_1 = (10*pow(cht_SG_1,4) - 9*pow(cht_SG_1,6) + 3*pow(cht_SG_1,8))/4
                P4_SG_2 = (10*pow(cht_SG_2,4) - 9*pow(cht_SG_2,6) + 3*pow(cht_SG_2,8))/4
				
                P2_SGD_1 = (3*pow(cht_SGD_1,2) + pow(cht_SGD_1,4))/4
                P2_SGD_2 = (3*pow(cht_SGD_2,2) + pow(cht_SGD_2,4))/4      
                P4_SGD_1 = (10*pow(cht_SGD_1,4) - 9*pow(cht_SGD_1,6) + 3*pow(cht_SGD_1,8))/4
                P4_SGD_2 = (10*pow(cht_SGD_2,4) - 9*pow(cht_SGD_2,6) + 3*pow(cht_SGD_2,8))/4				
                
                P2_eq_1 = (3*pow(cht_eq_1,2) + pow(cht_eq_1,4))/4
                P2_eq_2 = (3*pow(cht_eq_2,2) + pow(cht_eq_2,4))/4      
                P4_eq_1 = (10*pow(cht_eq_1,4) - 9*pow(cht_eq_1,6) + 3*pow(cht_eq_1,8))/4
                P4_eq_2 = (10*pow(cht_eq_2,4) - 9*pow(cht_eq_2,6) + 3*pow(cht_eq_2,8))/4
                
                f2_1_ratio_SG = math.tanh(K*P2_SG_1)/math.tanh(K*P2_eq_1)
                f2_2_ratio_SG = math.tanh(K*P2_SG_2)/math.tanh(K*P2_eq_2)
                f4_1_ratio_SG = math.tanh(K*P4_SG_1)/math.tanh(K*P4_eq_1)
                f4_2_ratio_SG = math.tanh(K*P4_SG_2)/math.tanh(K*P4_eq_2)
                
                f2_1_ratio_SGD = math.tanh(K*P2_SGD_1)/math.tanh(K*P2_eq_1)
                f2_2_ratio_SGD = math.tanh(K*P2_SGD_2)/math.tanh(K*P2_eq_2)
                f4_1_ratio_SGD = math.tanh(K*P4_SGD_1)/math.tanh(K*P4_eq_1)
                f4_2_ratio_SGD = math.tanh(K*P4_SGD_2)/math.tanh(K*P4_eq_2)
                
                H_ADDT_2_SG = f2_1_ratio_SG*f2_2_ratio_SG
                H_ADDT_2_SGD = f2_1_ratio_SGD*f2_2_ratio_SGD
                
                H_ADDT_4_SG = f4_1_ratio_SG*f4_2_ratio_SG
                H_ADDT_4_SGD = f4_1_ratio_SGD*f4_2_ratio_SGD
				
            torsion6_potential_SG_i = (1/counting_duplicates[i][j])*sign_dihedral_type[i][j]*( 2*math.sin(2*(dihedral_params_SG[i][j] - equil_dihedral_params[i][j]))*H_ADDT_2_SG - math.sin(4*(dihedral_params_SG[i][j] - equil_dihedral_params[i][j]))*H_ADDT_4_SG)/np.sqrt(5)
            torsion6_potential_SGD_i = (1/counting_duplicates[i][j])*sign_dihedral_type[i][j]*( 2*math.sin(2*(dihedral_params_SGD[i][j] - equil_dihedral_params[i][j]))*H_ADDT_2_SGD - math.sin(4*(dihedral_params_SGD[i][j] - equil_dihedral_params[i][j]))*H_ADDT_4_SGD)/np.sqrt(5)
            delta_torsion6_potential[i] += (torsion6_potential_SG_i - torsion6_potential_SGD_i)*10000
    
    delta_torsion6_potential_final = [element for element, flag1, flag2 in zip(delta_torsion6_potential, mode6_dihedral_type, dihedral_type) if flag1 == 1 and flag2 == 'Rotatable']
    return  delta_torsion6_potential_final

def calc_delta_torsion7_potential(affected_dihedral, dihedral_params_SG, dihedral_params_SGD, equil_dihedral_params, mode7_dihedral_type, dihedral_type, counting_duplicates, sign_dihedral_type, dihedral_angles_params_SG, dihedral_angles_params_SGD, dihedral_angles_eq_params,atom):
    delta_torsion7_potential = np.zeros(shape = (len(mode7_dihedral_type), 1), dtype=float)

    angle_threshold = 130*np.pi/180
    K = 2.815891616117388
    
    for index, affected_dihedral in enumerate(affected_dihedrals[atom-1]):
        i = affected_dihedral[0]
        j = affected_dihedral[1]
        if mode7_dihedral_type[i] == 1 and dihedral_type[i] == 'Rotatable':
            
            theta_1_SG = dihedral_angles_params_SG[i][j][0]
            theta_2_SG = dihedral_angles_params_SG[i][j][1]
            theta_1_SGD = dihedral_angles_params_SGD[i][j][0]
            theta_2_SGD = dihedral_angles_params_SGD[i][j][1]
            theta_eq_1 = dihedral_angles_eq_params[i][j][0]
            theta_eq_2 = dihedral_angles_eq_params[i][j][1]
            if theta_eq_1 < angle_threshold and theta_eq_2 < angle_threshold:
                H_ADDT_1_SG = 1; H_ADDT_2_SG = 1; H_ADDT_3_SG = 1; H_ADDT_4_SG = 1; H_ADDT_1_SGD = 1; H_ADDT_2_SGD = 1; H_ADDT_3_SGD = 1; H_ADDT_4_SGD = 1;
            else:
                cht_SG_1 = math.cos(theta_1_SG/2)
                cht_SG_2 = math.cos(theta_2_SG/2)
                cht_SGD_1 = math.cos(theta_1_SGD/2)
                cht_SGD_2 = math.cos(theta_2_SGD/2)
                
                cht_eq_1 = math.cos(theta_eq_1/2)
                cht_eq_2 = math.cos(theta_eq_2/2)
                
                P1_SG_1 = (cht_SG_1 + 3*pow(cht_SG_1,3))/4
                P1_SG_2 = (cht_SG_2 + 3*pow(cht_SG_2,3))/4
                P2_SG_1 = (3*pow(cht_SG_1,2) + pow(cht_SG_1,4))/4
                P2_SG_2 = (3*pow(cht_SG_2,2) + pow(cht_SG_2,4))/4 
                P3_SG_1 = (6*pow(cht_SG_1,3) - 3*pow(cht_SG_1,5) + pow(cht_SG_1,7))/4
                P3_SG_2 = (6*pow(cht_SG_2,3) - 3*pow(cht_SG_2,5) + pow(cht_SG_2,7))/4     
                P4_SG_1 = (10*pow(cht_SG_1,4) - 9*pow(cht_SG_1,6) + 3*pow(cht_SG_1,8))/4
                P4_SG_2 = (10*pow(cht_SG_2,4) - 9*pow(cht_SG_2,6) + 3*pow(cht_SG_2,8))/4
				
                P1_SGD_1 = (cht_SGD_1 + 3*pow(cht_SGD_1,3))/4
                P1_SGD_2 = (cht_SGD_2 + 3*pow(cht_SGD_2,3))/4
                P2_SGD_1 = (3*pow(cht_SGD_1,2) + pow(cht_SGD_1,4))/4
                P2_SGD_2 = (3*pow(cht_SGD_2,2) + pow(cht_SGD_2,4))/4 
                P3_SGD_1 = (6*pow(cht_SGD_1,3) - 3*pow(cht_SGD_1,5) + pow(cht_SGD_1,7))/4
                P3_SGD_2 = (6*pow(cht_SGD_2,3) - 3*pow(cht_SGD_2,5) + pow(cht_SGD_2,7))/4     
                P4_SGD_1 = (10*pow(cht_SGD_1,4) - 9*pow(cht_SGD_1,6) + 3*pow(cht_SGD_1,8))/4
                P4_SGD_2 = (10*pow(cht_SGD_2,4) - 9*pow(cht_SGD_2,6) + 3*pow(cht_SGD_2,8))/4				
                
                P1_eq_1 = (cht_eq_1 + 3*pow(cht_eq_1,3))/4
                P1_eq_2 = (cht_eq_2 + 3*pow(cht_eq_2,3))/4
                P2_eq_1 = (3*pow(cht_eq_1,2) + pow(cht_eq_1,4))/4
                P2_eq_2 = (3*pow(cht_eq_2,2) + pow(cht_eq_2,4))/4 
                P3_eq_1 = (6*pow(cht_eq_1,3) - 3*pow(cht_eq_1,5) + pow(cht_eq_1,7))/4
                P3_eq_2 = (6*pow(cht_eq_2,3) - 3*pow(cht_eq_2,5) + pow(cht_eq_2,7))/4     
                P4_eq_1 = (10*pow(cht_eq_1,4) - 9*pow(cht_eq_1,6) + 3*pow(cht_eq_1,8))/4
                P4_eq_2 = (10*pow(cht_eq_2,4) - 9*pow(cht_eq_2,6) + 3*pow(cht_eq_2,8))/4
                
                f1_1_ratio_SG = math.tanh(K*P1_SG_1)/math.tanh(K*P1_eq_1)
                f1_2_ratio_SG = math.tanh(K*P1_SG_2)/math.tanh(K*P1_eq_2)
                f2_1_ratio_SG = math.tanh(K*P2_SG_1)/math.tanh(K*P2_eq_1)
                f2_2_ratio_SG = math.tanh(K*P2_SG_2)/math.tanh(K*P2_eq_2)
                f3_1_ratio_SG = math.tanh(K*P3_SG_1)/math.tanh(K*P3_eq_1)
                f3_2_ratio_SG = math.tanh(K*P3_SG_2)/math.tanh(K*P3_eq_2)
                f4_1_ratio_SG = math.tanh(K*P4_SG_1)/math.tanh(K*P4_eq_1)
                f4_2_ratio_SG = math.tanh(K*P4_SG_2)/math.tanh(K*P4_eq_2)
                
                f1_1_ratio_SGD = math.tanh(K*P1_SGD_1)/math.tanh(K*P1_eq_1)
                f1_2_ratio_SGD = math.tanh(K*P1_SGD_2)/math.tanh(K*P1_eq_2)
                f2_1_ratio_SGD = math.tanh(K*P2_SGD_1)/math.tanh(K*P2_eq_1)
                f2_2_ratio_SGD = math.tanh(K*P2_SGD_2)/math.tanh(K*P2_eq_2)
                f3_1_ratio_SGD = math.tanh(K*P3_SGD_1)/math.tanh(K*P3_eq_1)
                f3_2_ratio_SGD = math.tanh(K*P3_SGD_2)/math.tanh(K*P3_eq_2)
                f4_1_ratio_SGD = math.tanh(K*P4_SGD_1)/math.tanh(K*P4_eq_1)
                f4_2_ratio_SGD = math.tanh(K*P4_SGD_2)/math.tanh(K*P4_eq_2)
                
                H_ADDT_1_SG = f1_1_ratio_SG*f1_2_ratio_SG
                H_ADDT_2_SG = f2_1_ratio_SG*f2_2_ratio_SG
                H_ADDT_3_SG = f3_1_ratio_SG*f3_2_ratio_SG
                H_ADDT_4_SG = f4_1_ratio_SG*f4_2_ratio_SG
                
                H_ADDT_1_SGD = f1_1_ratio_SGD*f1_2_ratio_SGD
                H_ADDT_2_SGD = f2_1_ratio_SGD*f2_2_ratio_SGD
                H_ADDT_3_SGD = f3_1_ratio_SGD*f3_2_ratio_SGD
                H_ADDT_4_SGD = f4_1_ratio_SGD*f4_2_ratio_SGD
			
            delta_phi_SG = dihedral_params_SG[i][j] - equil_dihedral_params[i][j]
            delta_phi_SGD = dihedral_params_SGD[i][j] - equil_dihedral_params[i][j]
            torsion7_potential_SG_i = (1/counting_duplicates[i][j])*sign_dihedral_type[i][j]*((math.sin(delta_phi_SG)*H_ADDT_1_SG  + 3*math.sin(3*delta_phi_SG)*H_ADDT_3_SG) - (math.sin(2*delta_phi_SG)*H_ADDT_2_SG + 2*math.sin(4*delta_phi_SG)*H_ADDT_4_SG))/np.sqrt(15)
            torsion7_potential_SGD_i = (1/counting_duplicates[i][j])*sign_dihedral_type[i][j]*((math.sin(delta_phi_SGD)*H_ADDT_1_SGD  + 3*math.sin(3*delta_phi_SGD)*H_ADDT_3_SGD) - (math.sin(2*delta_phi_SGD)*H_ADDT_2_SGD + 2*math.sin(4*delta_phi_SGD)*H_ADDT_4_SGD))/np.sqrt(15)
            delta_torsion7_potential[i] += (torsion7_potential_SG_i - torsion7_potential_SGD_i)*10000
    
    delta_torsion7_potential_final = [element for element, flag1, flag2 in zip(delta_torsion7_potential, mode7_dihedral_type, dihedral_type) if flag1 == 1 and flag2 == 'Rotatable']
    return  delta_torsion7_potential_final

def calc_delta_torsion8_potential(affected_dihedral, dihedral_params_SG, dihedral_params_SGD, mode8_dihedral_type, dihedral_type, counting_duplicates, dihedral_angles_params_SG, dihedral_angles_params_SGD,atom):
    delta_torsion8_potential = np.zeros(shape = (len(mode8_dihedral_type), 1), dtype=float)

    K = 2.815891616117388
    tanh_K = math.tanh(K)
    
    for index, affected_dihedral in enumerate(affected_dihedrals[atom-1]):
        i = affected_dihedral[0]
        j = affected_dihedral[1]
        if mode8_dihedral_type[i] == 1 and dihedral_type[i] == 'Linear':
            theta_1_SG = dihedral_angles_params_SG[i][j][0]
            theta_2_SG = dihedral_angles_params_SG[i][j][1]
            theta_1_SGD = dihedral_angles_params_SGD[i][j][0]
            theta_2_SGD = dihedral_angles_params_SGD[i][j][1]

            cht_SG_1 = math.cos(theta_1_SG/2)
            cht_SG_2 = math.cos(theta_2_SG/2)
            cht_SGD_1 = math.cos(theta_1_SGD/2)
            cht_SGD_2 = math.cos(theta_2_SGD/2)
            
            P1_SG_1 = (cht_SG_1 + 3*pow(cht_SG_1,3))/4
            P1_SG_2 = (cht_SG_2 + 3*pow(cht_SG_2,3))/4
				
            P1_SGD_1 = (cht_SGD_1 + 3*pow(cht_SGD_1,3))/4
            P1_SGD_2 = (cht_SGD_2 + 3*pow(cht_SGD_2,3))/4
            
            fp1_SG = math.tanh(K*P1_SG_1)*math.tanh(K*P1_SG_2)/(tanh_K**2)
            fp1_SG_pw2 = fp1_SG*fp1_SG
            
            fp1_SGD = math.tanh(K*P1_SGD_1)*math.tanh(K*P1_SGD_2)/(tanh_K**2)
            fp1_SGD_pw2 = fp1_SGD*fp1_SGD
                
            torsion8_potential_SG_i = (1/counting_duplicates[i][j])*fp1_SG_pw2*(1 - math.cos(2*dihedral_params_SG[i][j]))
            torsion8_potential_SGD_i = (1/counting_duplicates[i][j])*fp1_SGD_pw2*(1 - math.cos(2*dihedral_params_SGD[i][j]))
            
            delta_torsion8_potential[i] += (torsion8_potential_SG_i - torsion8_potential_SGD_i)*10000
                
    delta_torsion8_potential_final = [element for element, flag1, flag2 in zip(delta_torsion8_potential, mode8_dihedral_type, dihedral_type) if flag1 == 1 and flag2 == 'Linear']
    return  delta_torsion8_potential_final
               
def calc_delta_torsion9_potential(affected_dihedral, dihedral_params_SG, dihedral_params_SGD, mode9_dihedral_type, dihedral_type, counting_duplicates, dihedral_angles_params_SG, dihedral_angles_params_SGD,atom):
    delta_torsion9_potential = np.zeros(shape = (len(mode9_dihedral_type), 1), dtype=float)

    K = 2.815891616117388
    tanh_K = math.tanh(K)
    
    for index, affected_dihedral in enumerate(affected_dihedrals[atom-1]):
        i = affected_dihedral[0]
        j = affected_dihedral[1]
        if mode9_dihedral_type[i] == 1 and dihedral_type[i] == 'Linear':
            theta_1_SG = dihedral_angles_params_SG[i][j][0]
            theta_2_SG = dihedral_angles_params_SG[i][j][1]
            theta_1_SGD = dihedral_angles_params_SGD[i][j][0]
            theta_2_SGD = dihedral_angles_params_SGD[i][j][1]

            cht_SG_1 = math.cos(theta_1_SG/2)
            cht_SG_2 = math.cos(theta_2_SG/2)
            cht_SGD_1 = math.cos(theta_1_SGD/2)
            cht_SGD_2 = math.cos(theta_2_SGD/2)
            
            P1_SG_1 = (cht_SG_1 + 3*pow(cht_SG_1,3))/4
            P1_SG_2 = (cht_SG_2 + 3*pow(cht_SG_2,3))/4
				
            P1_SGD_1 = (cht_SGD_1 + 3*pow(cht_SGD_1,3))/4
            P1_SGD_2 = (cht_SGD_2 + 3*pow(cht_SGD_2,3))/4
            
            fp1_SG = math.tanh(K*P1_SG_1)*math.tanh(K*P1_SG_2)/(tanh_K**2)
            fp1_SG_pw2 = fp1_SG*fp1_SG
            
            fp1_SGD = math.tanh(K*P1_SGD_1)*math.tanh(K*P1_SGD_2)/(tanh_K**2)
            fp1_SGD_pw2 = fp1_SGD*fp1_SGD  
                
            torsion9_potential_SG_i = (1/counting_duplicates[i][j])*fp1_SG_pw2*(1 + math.cos(2*dihedral_params_SG[i][j]))
            torsion9_potential_SGD_i = (1/counting_duplicates[i][j])*fp1_SGD_pw2*(1 + math.cos(2*dihedral_params_SGD[i][j]))
            
            delta_torsion9_potential[i] += (torsion9_potential_SG_i - torsion9_potential_SGD_i)*10000
                
    delta_torsion9_potential_final = [element for element, flag1, flag2 in zip(delta_torsion9_potential, mode9_dihedral_type, dihedral_type) if flag1 == 1 and flag2 == 'Linear']
    return  delta_torsion9_potential_final

def choose_bonds(bonds, natoms):
    affected_bonds = []
    for atom in range(1, natoms+1):   
        affected_atom = []
        for bond in range(len(bonds)):
            for i in range(len(bonds[bond])):
                single_line = bonds[bond][i]
                if atom == single_line[0]:
                    affected_atom.append([single_line[4], i])
                if atom == single_line[2]:
                    affected_atom.append([single_line[4], i])
        affected_bonds.append(affected_atom)
    # Structure [ [ [bond type containing atom 1, bond# for atom1],... all bond type containing atom 1] ...., [[bond type for atomn, bond# for atomn]]  ]
    return affected_bonds

def choose_angles(angles, natoms):
    affected_angles = []
    for atom in range(1, natoms+1):   
        affected_atom = []
        for angle in range(len(angles)):    
            for i in range(len(angles[angle])): 
                single_line = angles[angle][i]
                if atom == single_line[0]:
                    affected_atom.append([single_line[5], i])
                if atom == single_line[1]:
                    affected_atom.append([single_line[5], i])
                if atom == single_line[3]:
                    affected_atom.append([single_line[5], i])
        affected_angles.append(affected_atom)
    return affected_angles

def choose_dihedrals(dihedrals, natoms):
    affected_dihedrals = []
    for atom in range(1, natoms+1):   
        affected_atom = []
        for dihedral in range(len(dihedrals)):    
            for i in range(len(dihedrals[dihedral])): 
                single_line = dihedrals[dihedral][i]
                if atom == single_line[0]:
                    affected_atom.append([single_line[8], i])
                if atom == single_line[2]:
                    affected_atom.append([single_line[8], i])
                if atom == single_line[4]:
                    affected_atom.append([single_line[8], i])
                if atom == single_line[6]:
                    affected_atom.append([single_line[8], i])
        affected_dihedrals.append(affected_atom)
    return affected_dihedrals

def calc_params_vect_displaced_atoms(affected_bonds, affected_angles, affected_dihedrals, bond_type, angle_type, dihedral_type, atom, displaced_geom, params_vect_single_geom, atoms_and_indices, M):
    params_vect_displaced = copy.deepcopy(params_vect_single_geom)

    for index, affected_bonds in enumerate(affected_bonds[atom-1]):
        i = affected_bonds[0]
        j = affected_bonds[1]
        if any(bond_type[i][j][1]):
            A_atom_number = bond_type[i][j][2]
            B_atom_number = bond_type[i][j][0]
        else:
            A_atom_number = bond_type[i][j][0]
            B_atom_number = bond_type[i][j][2]
        R_A = displaced_geom[A_atom_number-1, :]
        R_B = displaced_geom[B_atom_number-1, :]
        R_B_pos_indice = calc_closest_image_pos(R_A, R_B, M)
        R_B_close = R_B_pos_indice[0]
        bond_length = np.linalg.norm(R_B_close-R_A)
        params_vect_displaced[0][i][j] = R_B_close-R_A
        params_vect_displaced[1][i][j] = bond_length
    

    for index, affected_angle in enumerate(affected_angles[atom-1]):
        i = affected_angle[0]
        j = affected_angle[1]
        angle_j = angle_type[i][j]
        k1 = atoms_and_indices[3][i][j][0]
        m1 = atoms_and_indices[3][i][j][1]
        k2 = atoms_and_indices[3][i][j][2]
        m2 = atoms_and_indices[3][i][j][3]
        if (angle_j[0] == atoms_and_indices[0][k1][m1][0] and angle_j[1] == atoms_and_indices[0][k1][m1][1]):
            bond_1_vector_displaced = params_vect_displaced[0][k1][m1]
        if (angle_j[0] == atoms_and_indices[0][k1][m1][1] and angle_j[1] == atoms_and_indices[0][k1][m1][0]):
            bond_1_vector_displaced = -params_vect_displaced[0][k1][m1]
        if (angle_j[0] == atoms_and_indices[0][k2][m2][0] and angle_j[3] == atoms_and_indices[0][k2][m2][1]):
            bond_2_vector_displaced = params_vect_displaced[0][k2][m2]
        if (angle_j[0] == atoms_and_indices[0][k2][m2][1] and angle_j[3] == atoms_and_indices[0][k2][m2][0]):
            bond_2_vector_displaced = -params_vect_displaced[0][k2][m2]
        theta_displaced = calc_angle(bond_1_vector_displaced,bond_2_vector_displaced)
        params_vect_displaced[2][i][j] = theta_displaced
        angle_bond_1 = params_vect_displaced[1][k1][m1]
        angle_bond_2 = params_vect_displaced[1][k2][m2]
        params_vect_displaced[5][i][j] = [angle_bond_1, angle_bond_2]
        
    for index, affected_dihedral in enumerate(affected_dihedrals[atom-1]):
        i = affected_dihedral[0]
        j = affected_dihedral[1]
        dihedral_j = dihedral_type[i][j]
        k1 = atoms_and_indices[4][i][j][0]
        m1 = atoms_and_indices[4][i][j][1]
        k2 = atoms_and_indices[4][i][j][2]
        m2 = atoms_and_indices[4][i][j][3]
        k3 = atoms_and_indices[4][i][j][4]
        m3 = atoms_and_indices[4][i][j][5]
        if (dihedral_j[0] == atoms_and_indices[0][k1][m1][0] and dihedral_j[2] == atoms_and_indices[0][k1][m1][1]):
            bond_1_vector_displaced = params_vect_displaced[0][k1][m1]
        if (dihedral_j[0] == atoms_and_indices[0][k1][m1][1] and dihedral_j[2] == atoms_and_indices[0][k1][m1][0]):
            bond_1_vector_displaced = -params_vect_displaced[0][k1][m1]
            
        if (dihedral_j[2] == atoms_and_indices[0][k2][m2][0] and dihedral_j[4] == atoms_and_indices[0][k2][m2][1]):
            bond_2_vector_displaced = params_vect_displaced[0][k2][m2]
        if (dihedral_j[2] == atoms_and_indices[0][k2][m2][1] and dihedral_j[4] == atoms_and_indices[0][k2][m2][0]):
            bond_2_vector_displaced = -params_vect_displaced[0][k2][m2]
            
        if (dihedral_j[4] == atoms_and_indices[0][k3][m3][0] and dihedral_j[6] == atoms_and_indices[0][k3][m3][1]):
            bond_3_vector_displaced = params_vect_displaced[0][k3][m3]
        if (dihedral_j[4] == atoms_and_indices[0][k3][m3][1] and dihedral_j[6] == atoms_and_indices[0][k3][m3][0]):
            bond_3_vector_displaced = -params_vect_displaced[0][k3][m3]    
            
        phi_displaced = calc_dihedral_phi(bond_1_vector_displaced,bond_2_vector_displaced,bond_3_vector_displaced)
        params_vect_displaced[3][i][j] = phi_displaced
        dihedral_angle_1 = calc_angle(-bond_1_vector_displaced,bond_2_vector_displaced)
        dihedral_angle_2 = calc_angle(-bond_2_vector_displaced,bond_3_vector_displaced)
        params_vect_displaced[4][i][j] = [dihedral_angle_1,dihedral_angle_2]
                
    return params_vect_displaced


start_time = time.time()
base_dir = './geometry_and_typing_files/'
Material_name = 'GEMCIT'
Flag_linear = False
path1_bonds = base_dir + Material_name +'_bond_and_UB_instances_list.txt'
path2_atom_type_bonds = base_dir + Material_name +'_bond_and_UB_types_list.txt'
path3_angles = base_dir  + Material_name +'_angle_instances_list.txt'
path4_atom_type_angles = base_dir  + Material_name +'_angle_types_list.txt'
path5_dihedrals = base_dir  + Material_name +'_dihedral_instances_list.txt'
path6_atom_type_dihedrals = base_dir  + Material_name +'_dihedral_types_list.txt'
path7_atom_types_with_coordinates = base_dir  + Material_name +'_atom_types_with_coordinates.txt'
path8_geom_force_array = base_dir + 'geom_forces_array_'+ Material_name +'_w_Hess.csv'
path9_xyz = base_dir + Material_name + '.xyz'

f1_bonds = open(path1_bonds,'r')
f2_att_bonds = open(path2_atom_type_bonds,'r')
f3_angles = open(path3_angles,'r')
f4_att_angles = open(path4_atom_type_angles,'r')
f5_dihedrals = open(path5_dihedrals,'r')
f6_att_dihedrals = open(path6_atom_type_dihedrals,'r')    
f7_att_with_coordinates = open(path7_atom_types_with_coordinates,'r')

lines1_bonds = f1_bonds.readlines()
lines2_att_bonds = f2_att_bonds.readlines()
lines3_angles = f3_angles.readlines()
lines4_att_angles = f4_att_angles.readlines()
lines5_dihedrals = f5_dihedrals.readlines()
lines6_att_dihedrals = f6_att_dihedrals.readlines()
lines7_att_with_coordinates = f7_att_with_coordinates.readlines()

geom_forces = open(path8_geom_force_array,'r');
lines_geom_forces = geom_forces.readlines()

f9_xyz = open(path9_xyz, 'r')
lines9_xyz = f9_xyz.readlines()
number_atom = int(lines9_xyz[0].split()[0])
natoms = len(lines7_att_with_coordinates)-6;

print('\nThe refcode is: ',Material_name)
print('\nNumber of atoms in the unit cell are: ',number_atom)

atomic_symbol_and_number = []
count_atom_n = 1
for line_index in range(len(lines9_xyz)):
    line = lines9_xyz[line_index]
    if 'jmolscript' in line:
        currentlinenum = lines9_xyz.index(line)
        index_s = line.index('[')
        index_e = line.index(']')
        lat_str = line[index_s+1:index_e]
        a_start = lat_str.index('{')
        a_end = lat_str.index('}')
        a = lat_str[a_start+1:a_end-1].split()
        lat2_str = lat_str[a_end+1:index_e]
        b_str_start = lat2_str.index('{')
        b_str_end = lat2_str.index('}')
        b = lat2_str[b_str_start+1:b_str_end-1].split()
        lat3_str = lat2_str[b_str_end+1:index_e]
        c_str_start = lat3_str.index('{')
        c_str_end = lat3_str.index('}')
        c = lat3_str[c_str_start+1:c_str_end-1].split()
        a_vector = np.array([float(a[0]), float(a[1]), float(a[2])])
        b_vector = np.array([float(b[0]), float(b[1]), float(b[2])])
        c_vector = np.array([float(c[0]), float(c[1]), float(c[2])])
        Lattice_matrix = np.array([a_vector,b_vector,c_vector])
    if 2<= line_index <=(1+number_atom):
        line_split = line.split()
        atomic_symbol_i = line_split[0]
        atomic_symbol_and_number.append([count_atom_n,atomic_symbol_i])
        count_atom_n += 1
        
print('\nLattice vectors in units of Angstrom: ')
print(Lattice_matrix)

pairs_bond = []  
for j in range(0,len(lines1_bonds)):
    L_01 = lines1_bonds[j]
    b = []
    for i in range(7):
        b.append([])
    ind_1 = L_01.index('[')
    ind_2 = L_01.index(',')
    b_0 = L_01[ind_1+1:ind_2-1]
    b[0]= int(b_0)
    str_2 = L_01[ind_2+1:]
    ind_3 = str_2.index('[')
    ind_4 = str_2.index(',')
    b_1_0 = int(str_2[ind_3+1:ind_4])
    str_3 = str_2[ind_4+1:]
    ind_5 = str_3.index(',')
    b_1_1 = int(str_3[0:ind_5])
    str_4 = str_3[ind_5+1:]
    ind_6 = str_4.index(']')
    b_1_2 = int(str_4[0:ind_6])
    b_1 = [b_1_0,b_1_1,b_1_2]
    b[1]=b_1  
    str_5 = str_4[ind_6+1:]
    ind_7 = str_5.index(',')
    str_6 = str_5[ind_7+1:]
    ind_8 = str_6.index(',')
    b_2 = int(str_6[0:ind_8])
    b[2] = int(b_2)
    str_7 = str_6[ind_8+1:]
    ind_9 = str_7.index('[')
    ind_10 = str_7.index(',')
    b_3_0 = int(str_7[ind_9+1:ind_10])
    str_8 = str_7[ind_10+1:]
    ind_11 = str_8.index(',')
    b_3_1 = int(str_8[0:ind_11])
    str_9 = str_8[ind_11+1:]
    ind_12 = str_9.index(']')
    b_3_2 = int(str_9[0:ind_12])
    b_3 = [b_3_0,b_3_1,b_3_2]
    b[3]=b_3  
    ind_13 = str_9.index(',')
    str_10 = str_9[ind_13+1:]
    ind_14 = str_10.index(',')
    b_4 = int(str_10[0:ind_14])
    b[4] = b_4
    str_11 = str_10[ind_14+1:]
    ind_15 = str_11.index(',')
    b_5 = str_11[0:ind_15]
    b[5] = int(b_5)
    str_12 = str_11[ind_15+1:]
    ind_16 = str_12.index(']')
    b_6 = int(str_12[0:ind_16-1])
    b[6] = b_6
    pairs_bond.append(b)

triplet_angle = []
for i in range(0,len(lines3_angles)):
    L_0 = lines3_angles[i]
    a=[]
    for i in range(7):
        a.append([])
    index_1 = L_0.index('[')
    index_2 = L_0.index(',')
    a_0 = L_0[index_1+1:index_2-1]
    a[0]= int(a_0)
    str_2 = L_0[index_2+1:-1]
    index_3 = str_2.index(',')
    a_1 = str_2[0:index_3-1]
    a[1] = int(a_1)
    str_3 = str_2[index_3+1:-1]
    index_4 = str_3.index('[')
    index_5 = str_3.index(',')
    a_2_0 = int(str_3[index_4+1:index_5])
    str_4 = str_3[index_5+1:-1]
    index_6 = str_4.index(',')
    a_2_1 = int(str_4[0:index_6])
    str_5 = str_4[index_6+1:-1]
    index_7 = str_5.index(']')
    a_2_2 = int(str_5[0:index_7])
    a_2 = [a_2_0,a_2_1,a_2_2]
    a[2]=a_2
    index_8 = str_5.index(',')
    str_6 = str_5[index_8+1:-1]
    index_9 = str_6.index(',')
    a_3 = int(str_6[0:index_9])
    a[3]=a_3
    str_7 = str_6[index_9+1:-1]
    index_10 = str_7.index('[')
    index_11 = str_7.index(',')
    a_4_0 = int(str_7[index_10+1:index_11])
    str_8 = str_7[index_11+1:-1]
    index_12 = str_8.index(',')
    a_4_1 = int(str_8[0:index_12])
    str_9 = str_8[index_12+1:-1]
    index_13 = str_9.index(']')
    a_4_2 = int(str_9[0:index_13])
    a_4 = [a_4_0,a_4_1,a_4_2]
    a[4]=a_4
    index_14 = str_9.index(',')
    str_10 = str_9[index_14+1:]
    index_15 = str_10.index(',')
    a_5 = int(str_10[0:index_15])
    a[5] = a_5
    index_16 = L_0.rfind(',')
    index_17 = L_0.rfind(']')
    str_11 = L_0[index_16+1:index_17]
    a_6 = int(str_11)
    a[6]=a_6
    triplet_angle.append(a)

dihedrals = []
for i in range(0,len(lines5_dihedrals)):
    L_0 = lines5_dihedrals[i]
    a=[]
    for i in range(12):
        a.append([])
    index_1 = L_0.index('[')
    index_2 = L_0.index(',')
    a_0 = L_0[index_1+1:index_2-1]
    a[0]= int(a_0)
    str_2 = L_0[index_2+1:]
    index_3 = str_2.index('[')
    index_4 = str_2.index(',')
    a_1_0 = int(str_2[index_3+1:index_4])
    str_3 = str_2[index_4+1:]
    index_5 = str_3.index(',')
    a_1_1 = int(str_3[0:index_5])
    str_4 = str_3[index_5+1:]
    index_6 = str_4.index(']')
    a_1_2 = int(str_4[0:index_6])
    a_1 = [a_1_0,a_1_1,a_1_2]
    a[1]=a_1
    index_6_ = str_4.index(',')
    str_5 = str_4[index_6_+1:]
    index_7 = str_5.index(',')
    a_2 = str_5[0:index_7]
    a[2] = int(a_2)
    str_6 = str_5[index_7+1:]
    index_8 = str_6.index('[')
    index_9 = str_6.index(',')
    a_3_0 = int(str_6[index_8+1:index_9])
    str_7 = str_6[index_9+1:]
    index_10 = str_7.index(',')
    a_3_1 = int(str_7[0:index_10])
    str_8 = str_7[index_10+1:]
    index_11 = str_8.index(']')
    a_3_2 = int(str_8[0:index_11])
    a_3 = [a_3_0,a_3_1,a_3_2]
    a[3]=a_3
    index_12 = str_8.index(',')
    str_9 = str_8[index_12+1:]
    index_13 = str_9.index(',')
    a_4 = int(str_9[0:index_13])
    a[4]=a_4
    str_10 = str_9[index_13+1:]
    index_14 = str_10.index('[')
    index_15 = str_10.index(',')
    a_5_0 = int(str_10[index_14+1:index_15])
    str_11 = str_10[index_15+1:]
    index_16 = str_11.index(',')
    a_5_1 = int(str_11[0:index_16])
    str_12 = str_11[index_16+1:]
    index_17 = str_12.index(']')
    a_5_2 = int(str_12[0:index_17])
    a_5 = [a_5_0,a_5_1,a_5_2]
    a[5]=a_5
    index_18 = str_12.index(',')
    str_13 = str_12[index_18+1:]
    index_19 = str_13.index(',')
    a_6 = int(str_13[0:index_19])
    a[6]=a_6
    str_14 = str_13[index_19+1:]
    index_20 = str_14.index('[')
    index_21 = str_14.index(',')
    a_7_0 = int(str_14[index_20+1:index_21])
    str_15 = str_14[index_21+1:]
    index_22 = str_15.index(',')
    a_7_1 = int(str_15[0:index_22])
    str_16 = str_15[index_22+1:]
    index_23 = str_16.index(']')
    a_7_2 = int(str_16[0:index_23])
    a_7 = [a_7_0,a_7_1,a_7_2]
    a[7]=a_7
    index_24 = str_16.index(',')
    str_17 = str_16[index_24+1:]
    index_25 = str_17.index(',')
    a_8 = int(str_17[0:index_25])
    a[8]=a_8
    index_26 = str_17.index(',')
    str_18 = str_17[index_26+1:]
    index_27 = str_18.index(',')
    a_8 = int(str_18[0:index_27])
    a[9]=a_8
    index_28 = str_18.index(',')
    str_19 = str_18[index_28+1:]
    index_29 = str_19.index(',')
    a_9 = str(str_19[0:index_29])
    a[10] = a_9
    index_30 = str_19.index(',')
    str_20 = str_19[index_30+1:]
    index_31 = str_20.index(']')
    a_10 = int(str_20[0:index_31])
    a[11]=a_10
    dihedrals.append(a)    
      
natoms = len(lines7_att_with_coordinates)-6;
ngeoms_train = round(len(lines_geom_forces)/natoms)


nstrech_potential = len(lines2_att_bonds)
nbend_potential = 0
angle_3_4_0_type = []
for i in range(len(lines4_att_angles)):
    c = 0
    for j in range(len(triplet_angle)):
        if triplet_angle[j][5] == i and c == 0:
            if triplet_angle[j][6] == 0:
                angle_3_4_0_type.append(triplet_angle[j][6])
                nbend_potential += 1
                c = 1
            else:
                angle_3_4_0_type.append(triplet_angle[j][6])
                c = 1

nbond_bond_potential = len(lines4_att_angles)            
nbond_angle_potential = len(lines4_att_angles)

# angle-angle cross term: discover which pairs of angle TYPES co-occur at a
# shared central (vertex) atom somewhere in the molecule. Built from
# triplet_angle directly since it is available before angle_type is populated.
central_atom_groups_aa = {}
for j_aa in range(len(triplet_angle)):
    c_aa = triplet_angle[j_aa][0]
    type_i_aa = triplet_angle[j_aa][5]
    central_atom_groups_aa.setdefault(c_aa, []).append((type_i_aa, j_aa))

# Only keep angle-angle type pairs that map onto a LAMMPS-representable
# 4-atom "improper" topology, i.e. the two angles must ALSO share one
# terminal atom (in addition to the vertex) at EVERY physical occurrence
# of the pair in the molecule. Type pairs with any fully-disjoint (5-atom)
# instance -- which no native LAMMPS bonded category supports -- are
# excluded from the candidate set entirely, so the fit and the LAMMPS
# angle_angle_cross implementation stay fully consistent: every angle_angle
# term this fit can produce is guaranteed realizable in LAMMPS, no post-hoc
# dropping needed downstream.
_aa_candidate_instance_pairs = {}
for c_aa, entries_aa in central_atom_groups_aa.items():
    for a_aa in range(len(entries_aa)):
        for b_aa in range(a_aa+1, len(entries_aa)):
            type_i1_aa, j1_aa = entries_aa[a_aa]
            type_i2_aa, j2_aa = entries_aa[b_aa]
            if type_i1_aa == type_i2_aa:
                continue
            type_pair_aa = (min(type_i1_aa, type_i2_aa), max(type_i1_aa, type_i2_aa))
            _aa_candidate_instance_pairs.setdefault(type_pair_aa, []).append((j1_aa, j2_aa))

aa_pair_to_col = {}
_aa_excluded_5atom = []
for type_pair_aa, _inst_pairs in _aa_candidate_instance_pairs.items():
    _representable = True
    for _j1_aa, _j2_aa in _inst_pairs:
        _terms1_aa = {triplet_angle[_j1_aa][1], triplet_angle[_j1_aa][3]}
        _terms2_aa = {triplet_angle[_j2_aa][1], triplet_angle[_j2_aa][3]}
        if len(_terms1_aa & _terms2_aa) != 1:
            _representable = False
            break
    if _representable:
        aa_pair_to_col[type_pair_aa] = len(aa_pair_to_col)
    else:
        _aa_excluded_5atom.append(type_pair_aa)
print(f"angle-angle candidate scoping: {len(aa_pair_to_col)} type pairs kept (4-atom representable), "
      f"{len(_aa_excluded_5atom)} excluded (need 5-atom support): {_aa_excluded_5atom}")


# --- GROMACS-portability variant: exclude ALL angle-angle cross terms ---
# GROMACS's bonded interaction library has native support for bond-bond
# and bond-angle cross terms (used by class-II-style force fields) but no
# angle-angle cross term slot -- unlike bond_bond_cross/bond_angle_cross,
# there is no GROMACS home for angle_angle_cross at all. Clearing
# aa_pair_to_col here removes every angle-angle candidate from the LASSO/
# PSD-constrained fit itself (not a post-hoc drop) -- same technique as the
# 5-atom-only exclusion just above, extended to ALL angle-angle pairs, so
# the resulting force field only uses cross terms that have a direct home
# in both LAMMPS and GROMACS.
aa_pair_to_col.clear()

n_angle_angle_potential = len(aa_pair_to_col)
aa_col_to_type_pair = {v: k for k, v in aa_pair_to_col.items()}
ntorsion_potential = 0

list_torsion_type = []
list_torsion_type.append([])
list_torsion_type.append([])

list_stretch_type_info_ks = []
list_stretch_type_info_ks.append([])
list_stretch_type_info_ks.append([])
list_stretch_type_info_ks.append([])

list_stretch_type_info_ks[0] = [x for x in range(len(lines2_att_bonds))]
list_stretch_type_info_ks[1] = len(lines2_att_bonds)*['HARMONIC_BOND']
list_stretch_type_info_ks[2] = [x for x in range(0,nstrech_potential)]

list_bend_type_info_ks = []
list_bend_type_info_ks.append([])
list_bend_type_info_ks.append([])
list_bend_type_info_ks.append([])


list_bend_type_info_ks[0] = [x for x in range(nbend_potential)]
list_bend_type_info_ks[1] = nbend_potential*['MANZ_BEND']
list_bend_type_info_ks[2] = [x for x in range(nstrech_potential,(nstrech_potential+nbend_potential))]

list_bbc_type_info_ks = []
list_bbc_type_info_ks.append([])
list_bbc_type_info_ks.append([])
list_bbc_type_info_ks.append([])

list_bbc_type_info_ks[0] = [x for x in range(len(lines4_att_angles))]
list_bbc_type_info_ks[1] = len(lines4_att_angles)*['BOND_BOND_CROSS']
list_bbc_type_info_ks[2] = [x for x in range((nstrech_potential+nbend_potential),(nstrech_potential+nbend_potential+nbond_bond_potential))]

list_bac_type_info_ks = []
list_bac_type_info_ks.append([])
list_bac_type_info_ks.append([])
list_bac_type_info_ks.append([])

list_bac_type_info_ks[0] = [x for x in range(len(lines4_att_angles))]
list_bac_type_info_ks[1] = len(lines4_att_angles)*['BOND_ANGLE_CROSS']
list_bac_type_info_ks[2] = [x for x in range((nstrech_potential+nbend_potential+nbond_bond_potential),(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential))]

list_aac_type_info_ks = []
list_aac_type_info_ks.append([])
list_aac_type_info_ks.append([])
list_aac_type_info_ks.append([])

list_aac_type_info_ks[0] = ['angle_type_{}_x_angle_type_{}'.format(aa_col_to_type_pair[x][0], aa_col_to_type_pair[x][1]) for x in range(n_angle_angle_potential)]
list_aac_type_info_ks[1] = n_angle_angle_potential*['ANGLE_ANGLE_CROSS']
list_aac_type_info_ks[2] = [x for x in range((nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential),(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential))]

list_dihedral_type_info_ks = []
list_dihedral_type_info_ks.append([])
list_dihedral_type_info_ks.append([])
list_dihedral_type_info_ks.append([])
list_dihedral_type_info_ks.append([])


list_dihedral_type_info_ks[0] = [x for x in range(len(lines6_att_dihedrals))]

index_RD_in_X_array = []
index_RD_in_X_array.append([])
index_RD_in_X_array.append([])

number_of_rotatable = 0
number_of_linear = 0
number_of_linear_false_list = []
linear_type_indices = []
linear_k_indicies = []
for i in range(len(lines6_att_dihedrals)):
    if lines6_att_dihedrals[i].split()[-1] == 'Nonrotatable':
        ntorsion_potential += 1
        list_torsion_type[0].append('Nonrotatable')
        list_torsion_type[1].append([1])
    elif lines6_att_dihedrals[i].split()[-1] == 'Hindered':
        ntorsion_potential += 1
        list_torsion_type[0].append('Hindered')
        list_torsion_type[1].append([1])
    elif lines6_att_dihedrals[i].split()[-1] == 'Linear' and Flag_linear == True:
        ntorsion_potential += 2
        list_torsion_type[0].append('Linear')
        list_torsion_type[1].append([8,9])
        number_of_linear += 2
        linear_type_indices.append(i)
        linear_k_indicies.append([])
    elif lines6_att_dihedrals[i].split()[-1] == 'Linear' and Flag_linear == False:
        list_torsion_type[0].append('Linear_False_flag')
        list_torsion_type[1].append([-1])
        number_of_linear_false_list.append(i)
    elif lines6_att_dihedrals[i].split()[-1] == 'Rotatable':
        number_of_rotatable += 1
        ntorsion_potential += 4
        list_torsion_type[0].append('Rotatable')
        index_RD_in_X_array[0].append(i)
        index_RD_in_X_array[1].append([])
        list_torsion_type[1].append([1,2,3,4,5,6,7])
        
npotentials = nstrech_potential + nbend_potential + nbond_bond_potential + nbond_angle_potential + n_angle_angle_potential + ntorsion_potential

if Flag_linear == False:
    c = 0
    start_tor_ = npotentials-ntorsion_potential
    for i in range(len(lines6_att_dihedrals)):
        if i not in number_of_linear_false_list:
            list_dihedral_type_info_ks[3].append(start_tor_+ c)
            c+=1
        else:
            list_dihedral_type_info_ks[3].append(-1)
else:
    list_dihedral_type_info_ks[3] = [x for x in range((npotentials-ntorsion_potential),(npotentials-ntorsion_potential+len(lines6_att_dihedrals)))]

list_dihedral_type_info_ks[1] = list_torsion_type[0].copy()
list_dihedral_type_info_ks[2] = list_torsion_type[1]

Flag_RD = 'False'
if len(index_RD_in_X_array[0])>0:
    Flag_RD = 'True'
else:
    print('\nThis Material has no rotatable dihedrals.')

### rotatable part 1

if Flag_RD == 'True':
    path8_geom_array = base_dir +  Material_name +'_geoms_SPDFT.csv'
    path_E0 = base_dir + Material_name + '_E_SPDFT.csv'
    path_RD_type = base_dir + "RD_type_ind_SPDFT.txt"

    file_RD_type = open(path_RD_type,'r')
    lines_RD_type = file_RD_type.readlines()
    geom_array_file = open(path8_geom_array,'r');
    lines_geoms = geom_array_file.readlines()
    
    ngeoms_SPDFT = round(len(lines_geoms)/natoms)
    
    geoms_SPDFT = np.zeros(shape = (len(lines_geoms), 3), dtype=float)
    print('\n========================================================================================================')
    print('\nSelecting modes and fitting force constants for rotatable dihedrals')
    print('\nNumber of geometries in the single point DFT set are: ',ngeoms_SPDFT)
    
    for i in range(len(lines_geoms)):
        line_i = [float(j) for j in lines_geoms[i].split(',')]
        row_geom = np.array(line_i[0:3])
        geoms_SPDFT[i,:] = row_geom

### end of rotatable part 1
geoms = np.zeros(shape = (len(lines_geom_forces), 3), dtype=float)
geom_forces_array = np.zeros(shape = (len(lines_geom_forces), 6), dtype=float)
rows = len(geom_forces_array)

for i in range(len(lines_geom_forces)):
    line_i = [float(j) for j in lines_geom_forces[i].split(',')]
    row_geom = np.array(line_i[0:3])
    row_geom_force = np.array(line_i)
    geoms[i,:] = row_geom
    geom_forces_array[i,:] = row_geom_force

rows = len(geom_forces_array)
forces = np.zeros(shape = (3*rows, 1), dtype=float)

for i in range(1,rows+1):
     if i % natoms == 0:
         c=0
         for j in range(1,natoms+1):
            forces[3*i - j - 2 - c , 0] = geom_forces_array[i-j , 3]
            forces[3*i - j - 1 - c, 0] = geom_forces_array[i-j , 4]
            forces[3*i - j - c, 0] = geom_forces_array[i-j , 5];
            c = 2*j


bond_type = []
list_counting_bonds = []
for i in range(nstrech_potential):
    bond_type.append([])
    list_counting_bonds.append([])
    
UB_type_list = []
for i in range(nstrech_potential):
    c = 0
    for j in range(len(pairs_bond)):
        if pairs_bond[j][-3] == i:
            bond_type[i].append(pairs_bond[j])
            list_counting_bonds[i].append(pairs_bond[j][-1])
            if pairs_bond[j][-2] == 2 and c == 0:
                c = 1
    if c == 1:
        UB_type_list.append(i)
        list_stretch_type_info_ks[1][i] = 'HARMONIC_UREYBRADLEY'
                
     
angle_type = []
for i in range(nbond_bond_potential):
    angle_type.append([])

for i in range(nbond_bond_potential):
    for j in range(len(triplet_angle)):
        if triplet_angle[j][-2] == i:
            angle_type[i].append(triplet_angle[j]) 

dihedral_type = []
list_sign_dihedral_type = []
list_counting_dihedral = []
for i in range(len(lines6_att_dihedrals)):
    dihedral_type.append([])
    list_sign_dihedral_type.append([])
    list_counting_dihedral.append([])
    
for i in range(len(lines6_att_dihedrals)):
    for j in range(len(dihedrals)):
        if dihedrals[j][8] == i:
            dihedral_type[i].append(dihedrals[j])
            list_sign_dihedral_type[i].append(dihedrals[j][9])
            list_counting_dihedral[i].append(dihedrals[j][11])

path_instances_per_bond_type = base_dir + 'instances_per_stretch_type.txt'
path_instances_per_angle_type = base_dir + 'instances_per_angle_type.txt'
path_instances_per_dihedral_type = base_dir + 'instances_per_dihedral_type.txt'

file_instances_per_bond_type = open(path_instances_per_bond_type,'w+')
file_instances_per_angle_type = open(path_instances_per_angle_type,'w+')
file_instances_per_dihedral_type = open(path_instances_per_dihedral_type,'w+')

for i in range(len(bond_type)):
    c = 0
    for j in range(len(bond_type[i])):
        count_num = bond_type[i][j][-1]
        c += 1/count_num
    c_int = round(c)
    file_instances_per_bond_type.write(str(c_int)+'\n')

file_instances_per_bond_type.close()

for i in range(len(angle_type)):
    len_ind_angle_type = len(angle_type[i])
    file_instances_per_angle_type.write(str(len_ind_angle_type)+'\n')

file_instances_per_angle_type.close()

for i in range(len(dihedral_type)):
    c = 0
    for j in range(len(dihedral_type[i])):
        count_num = dihedral_type[i][j][-1]
        c += 1/count_num
    c_int = round(c)
    file_instances_per_dihedral_type.write(str(c_int)+'\n')

file_instances_per_dihedral_type.close()

affected_bonds = choose_bonds(bond_type, natoms) 
affected_angles = choose_angles(angle_type, natoms) 

angle_angle_instance_pairs = []
for i_aa in range(len(angle_type)):
    for j_aa in range(len(angle_type[i_aa])):
        c_aa = angle_type[i_aa][j_aa][0]
        for i2_aa in range(len(angle_type)):
            if i2_aa <= i_aa:
                continue
            type_pair_aa = (i_aa, i2_aa) if i_aa < i2_aa else (i2_aa, i_aa)
            if type_pair_aa not in aa_pair_to_col:
                continue
            for j2_aa in range(len(angle_type[i2_aa])):
                if angle_type[i2_aa][j2_aa][0] == c_aa:
                    col_aa = aa_pair_to_col[type_pair_aa]
                    angle_angle_instance_pairs.append((col_aa, i_aa, j_aa, i2_aa, j2_aa))

affected_angle_pairs = []
for atom_aa in range(1, natoms+1):
    affected_atom_aa = []
    for (col_aa, i1_aa, j1_aa, i2_aa, j2_aa) in angle_angle_instance_pairs:
        if [i1_aa, j1_aa] in affected_angles[atom_aa-1] or [i2_aa, j2_aa] in affected_angles[atom_aa-1]:
            affected_atom_aa.append((col_aa, i1_aa, j1_aa, i2_aa, j2_aa))
    affected_angle_pairs.append(affected_atom_aa)
affected_dihedrals = choose_dihedrals(dihedral_type, natoms)

atom_vectors_indices = calc_atoms_and_indices(bond_type, angle_type, dihedral_type)


### rotatable part 2
adjusted_c_list = []
adjusted_c_list.append([])
adjusted_c_list.append([])
adjusted_c_list.append([])

if Flag_RD == 'True':

    equil_geom = geoms_SPDFT[0:natoms, :];            
    params_vect_equil = calc_params_vect(equil_geom,bond_type,angle_type,dihedral_type,atom_vectors_indices,Lattice_matrix)
    
    dihedrals_phi_n = []
    dihedrals_angles_n = []
    
    dihedrals_phi_n = params_vect_equil[3]
    dihedrals_angles_n = params_vect_equil[4]
    
    dihedrals_phi_avg = []
    dihedrals_angles_avg = []

    for i in range(len(dihedrals_phi_n)):
        dihedrals_phi_avg.append([])
        dihedrals_angles_avg.append([])
        dihedral_angle_1 = 0
        dihedral_angle_2 = 0
        for k in range(len(dihedrals_phi_n[i])):
            dihedral_angle_1 += dihedrals_angles_n[i][k][0]
            dihedral_angle_2 += dihedrals_angles_n[i][k][1]
        
        avg_dihedral_angle_1 = dihedral_angle_1/len(dihedrals_angles_n[i])
        avg_dihedral_angle_2 = dihedral_angle_2/len(dihedrals_angles_n[i])
        
        abs_dihedrals_phi_n = [abs(ele) for ele in dihedrals_phi_n[i]]
        ave_dihedrals_phi_type = sum(abs_dihedrals_phi_n)/len(dihedrals_phi_n[i])
        for j in range(len(dihedrals_phi_n[i])):
            if dihedrals_phi_n[i][j]>=0:
                sign = 1
            else:
                sign = -1 
            dihedrals_phi_avg[i].append(sign*ave_dihedrals_phi_type)
            dihedrals_angles_avg[i].append([avg_dihedral_angle_1,avg_dihedral_angle_2])
			
    params_vect_equil[3] = dihedrals_phi_avg
    params_vect_equil[4] = dihedrals_angles_avg
	
    csv_phi_E = open(path_E0,'r')
    lines_csv = csv_phi_E.readlines()
    
    list_type_RD = []
    list_ind_RD = []
    
    flag = 0
    for line in lines_RD_type:
        if Material_name in line:
            flag = 1
            continue
        if flag == 1:
            if 'RD type index' in line:
                RD_type_i = line.split()[-1]
                list_type_RD.append(int(RD_type_i))
            if 'RD type individual index' in line:
                RD_index_i = line.split()[-1]
                list_ind_RD.append(int(RD_index_i))
            if 'Material_name:' in line:
                flag = 0
   
    RD_type = list_type_RD
    RD_index = list_ind_RD
    
    
    number_points_E_type = 36
    phi_list = np.array(range(-170,190,10))*np.pi/180
    count = 1
    
    valid_indices_list = []
    valid_indices_num = 0
    all_RD_type_ind_num = 0
    dE_list_total = np.zeros((len(lines_csv)-1),)
    line_split_0 = lines_csv[0].split(',')
    E_0_opt = float(line_split_0[0])
    
    print('\nRotatable dihedrals types and selected modes:\n')
    c1_SPDFT = []
    c_type_rot = 0
    
    
    for types_RD in range(len(RD_type)):
        
        dphi_list = np.zeros((number_points_E_type,))
        dE_list = np.zeros((number_points_E_type,))
        dE_list_sym = np.zeros((number_points_E_type,))
        E_0_list= np.zeros((number_points_E_type,))
        y_check = np.zeros((number_points_E_type,))
        
        c = 0
        for i in range(1+number_points_E_type*types_RD,number_points_E_type + number_points_E_type*types_RD + 1):
            line_split = lines_csv[i].split(',')
            E_0_list[c] = float(line_split[0])
            c += 1
        E_avg = np.mean(E_0_list)
        sum_dE_list_squared = 0
        sum_dE_list_sym_squared = 0
        for i in range(0,number_points_E_type):
            dE_list[i] = E_0_list[i] - E_avg
            sum_dE_list_squared += dE_list[i]**2
            if (abs(phi_list[i]) == 0.0) or (abs(phi_list[i]) == np.pi):
                dE_list_sym[i] = 0.0
            else:
                index_negative_phi = np.argmax(phi_list == -phi_list[i])
                dE_list_sym[i] = E_0_list[i] - E_0_list[index_negative_phi]
            sum_dE_list_sym_squared += dE_list_sym[i]**2
        
        sym_flag_value = 0.5*np.sqrt(sum_dE_list_sym_squared/sum_dE_list_squared)
        
        dE_list_total[number_points_E_type*types_RD:number_points_E_type*types_RD+number_points_E_type] = E_0_list[:] - E_0_opt
        dE_2_list = pow(dE_list,2)
        
        RD_type_i = RD_type[types_RD]
        RD_index_i = RD_index[types_RD]
        
        if sym_flag_value <= 0.01:
            list_torsion_type[0][RD_type_i] = 'CO_Rotatable'
            sym_flag = 'CO_Rotatable'
        else:
            sym_flag = 'Rotatable'
        
        phi_eq = params_vect_equil[3][RD_type_i][RD_index_i]
        
        dphi_list = phi_list - phi_eq
        sqrt_avg_dE2 = np.sqrt(np.mean(dE_2_list))

        C2_E = 1/(sqrt_avg_dE2)
        
        if sym_flag == 'Rotatable':
            modes = [1,2,3,4,5,6,7]
            check_values = []
            for m in modes:
                for i in range(len(dphi_list)):
                    if m <= 4:
                        y_check[i] = -(np.cos(m*dphi_list[i])*np.sqrt(2))*(dE_list[i]*C2_E)
                    elif m == 5:
                        y_check[i] = ((3*np.sin(dphi_list[i]) - np.sin(3*dphi_list[i]))/(np.sqrt(10)))*(dE_list[i]*C2_E)*np.sqrt(2)
                    elif m == 6:
                        y_check[i] = ((2*np.sin(2*dphi_list[i]) - np.sin(4*dphi_list[i]))/(np.sqrt(5)))*(dE_list[i]*C2_E)*np.sqrt(2)
                    elif m == 7:
                        y_check[i] = ((np.sin(dphi_list[i])  - np.sin(2*dphi_list[i]) + 3*np.sin(3*dphi_list[i]) - 2*np.sin(4*dphi_list[i]))/(np.sqrt(15)))*(dE_list[i]*C2_E)*np.sqrt(2)
                check_value = np.mean(y_check)
                check_values.append(abs(check_value))
            
            check_value_tot = 0
            for i in range(len(check_values)):
                check_value_tot += check_values[i]**2
                
            if check_value_tot > 1.01:
                print('\nError: smart selection sum(c_i^2) > 1')
                sys.exit()
            else:
                if 0.01 < sym_flag_value <= 0.1:
                    thresh_val = 0.01
                elif sym_flag_value > 0.1:
                    thresh_val = 0.1
                valid_indices = list(filter(lambda x: check_values[x] > thresh_val, range(len(check_values))))
                valid_indices_list.append(valid_indices)
                valid_indices_num += len(valid_indices)
                modes_valid = [modes[mode] for mode in valid_indices]
                if len(modes_valid) == 1:
                    c1_SPDFT.append(0)
                elif len(modes_valid) > 1:
                     c1_SPDFT_list_i = [-np.inf]*len(modes_valid)
                     c1_SPDFT = c1_SPDFT + c1_SPDFT_list_i
                list_torsion_type[1][RD_type_i] = modes_valid
                all_RD_type_ind_num += len(params_vect_equil[3][RD_type_i])
                str_dihed_modes = '{:^7}  {:^15}  {:^15}'.format(str(RD_type_i),str(modes_valid)[1:-1],sym_flag)
                print(str_dihed_modes)
                c_type_rot += 1
        else:
            modes = [1,2,3,4]
            check_values = []
            check_values_with_sign = []
            for m in modes:
                for i in range(len(phi_list)):
                    y_check[i] = (np.cos(m*phi_list[i])*np.sqrt(2))*(dE_list[i]*C2_E)
                check_value = np.mean(y_check)
                check_values.append(abs(check_value))
                check_values_with_sign.append(check_value)
                
            check_value_tot = 0
            for i in range(len(check_values)):
                check_value_tot += check_values[i]**2
                
            if check_value_tot > 1.01:
                print('\nError: smart selection sum(c_i^2) > 1')
                sys.exit()
            else:
                selected_c_i = np.zeros(4)
                valid_indices = list(filter(lambda x: check_values[x] > 1e-3, range(len(check_values))))
                valid_indices_list.append(valid_indices)
                valid_indices_num += 1
                c1_SPDFT.append(0)
                modes_valid = [modes[mode] for mode in valid_indices]
                list_torsion_type[1][RD_type_i] = [1]
                all_RD_type_ind_num += len(params_vect_equil[3][RD_type_i])
                str_dihed_modes = '{:^7}  {:^15}  {:^15}'.format(str(RD_type_i),str(modes_valid)[1:-1],sym_flag)
                print(str_dihed_modes)
                c_type_rot += 1
                for i in range(len(valid_indices)):
                    selected_c_i[valid_indices[i]] = check_values_with_sign[valid_indices[i]]
                adjusted_c_list[0].append(RD_type_i)
                adjusted_c_list[1].append(modes_valid)
                adjusted_c_list[2].append(selected_c_i)
                
            
    print('\nTotal number of rotatable dihedral types: ',str(c_type_rot))        
    X_array_SPDFT =  np.zeros(shape = (len(lines_csv)-1, valid_indices_num), dtype=float)
    dphi_list_all = np.zeros(shape = (all_RD_type_ind_num, len(lines_csv)-1), dtype=float)
    phi_list_all = np.zeros(shape = (all_RD_type_ind_num, len(lines_csv)-1), dtype=float)
    phi_eq_list_all = np.zeros(shape = (all_RD_type_ind_num, len(lines_csv)-1), dtype=float)
    dihedral_angle_A_list = np.zeros(shape = (all_RD_type_ind_num, len(lines_csv)-1), dtype=float)
    dihedral_angle_B_list = np.zeros(shape = (all_RD_type_ind_num, len(lines_csv)-1), dtype=float)
    dihedral_angle_A_eq_list = np.zeros(shape = (all_RD_type_ind_num, len(lines_csv)-1), dtype=float)
    dihedral_angle_B_eq_list = np.zeros(shape = (all_RD_type_ind_num, len(lines_csv)-1), dtype=float)
    
    for geom_number in range(2,ngeoms_SPDFT+1):
        i = geom_number*natoms
        single_geom = geoms_SPDFT[i-natoms:i, :]
        params_vect_single_geom = calc_params_vect(single_geom, bond_type, angle_type,dihedral_type,atom_vectors_indices,Lattice_matrix)

        c = 0
        for RD_type_i in RD_type:
            for ind_type_i in range(len(params_vect_equil[3][RD_type_i])):
                dphi_list_all[c][geom_number-2] = -(params_vect_equil[3][RD_type_i][ind_type_i] - params_vect_single_geom[3][RD_type_i][ind_type_i])
                phi_list_all[c][geom_number-2] = params_vect_single_geom[3][RD_type_i][ind_type_i]
                phi_eq_list_all[c][geom_number-2] = params_vect_equil[3][RD_type_i][ind_type_i]
                dihedral_angle_A_list[c][geom_number-2] = params_vect_single_geom[4][RD_type_i][ind_type_i][0]
                dihedral_angle_B_list[c][geom_number-2] = params_vect_single_geom[4][RD_type_i][ind_type_i][1]
                dihedral_angle_A_eq_list[c][geom_number-2] = params_vect_equil[4][RD_type_i][ind_type_i][0]
                dihedral_angle_B_eq_list[c][geom_number-2] = params_vect_equil[4][RD_type_i][ind_type_i][1]
                c += 1
    
    modes = [1,2,3,4,5,6,7]
    for i in range(len(dE_list_total)):
        c = 0
        count = 0
        for j in range((len(valid_indices_list))):
            RD_type_i = RD_type[j]
            if list_torsion_type[0][RD_type_i] == 'CO_Rotatable':    
                len_RD_type = len(params_vect_equil[4][RD_type_i])
                index_CO_rot = adjusted_c_list[0].index(RD_type_i)
                adjusted_c_i = adjusted_c_list[2][index_CO_rot]
                list_counting = []
                for list_index in range(len(dihedral_type[RD_type_i])):
                        list_counting_i = dihedral_type[RD_type_i][list_index][11]
                        list_counting.append(list_counting_i)
                
                X_array_SPDFT[i,c] = calc_torsion_potential_CO(phi_list_all.T[i][count:count + len_RD_type],phi_eq_list_all.T[i][count:count + len_RD_type],adjusted_c_i,list_counting,dihedral_angle_A_list.T[i][count:count + len_RD_type],dihedral_angle_B_list.T[i][count:count + len_RD_type],dihedral_angle_A_eq_list.T[i][count:count + len_RD_type],dihedral_angle_B_eq_list.T[i][count:count + len_RD_type])
                c += 1
            else:
                for k in range(len(valid_indices_list[j])):
                    valid_indices_i = valid_indices_list[j][k]
                    len_RD_type = len(params_vect_equil[4][RD_type_i])
                    sym_rot_type = list_torsion_type[0][RD_type_i]
                    list_sign = []
                    list_counting = []
                    for list_index in range(len(dihedral_type[RD_type_i])):
                        list_sign_i = dihedral_type[RD_type_i][list_index][9]
                        list_conting_i = dihedral_type[RD_type_i][list_index][11]
                        list_sign.append(list_sign_i)
                        list_counting.append(list_conting_i)
                    
                    X_array_SPDFT[i,c] = calc_torsion_potential(dphi_list_all.T[i][count:count + len_RD_type],modes[valid_indices_i],list_sign,list_counting,dihedral_angle_A_list.T[i][count:count + len_RD_type],dihedral_angle_B_list.T[i][count:count + len_RD_type],dihedral_angle_A_eq_list.T[i][count:count + len_RD_type],dihedral_angle_B_eq_list.T[i][count:count + len_RD_type])
                    c += 1
            count += len_RD_type
    
    for i in range(X_array_SPDFT.shape[1]):
        X_array_SPDFT[:,i] = X_array_SPDFT[:,i] - np.mean(X_array_SPDFT[:,i]) 
        
    dE_list_total = dE_list_total - np.mean(dE_list_total)
    
    SST_SPDFT = 0;
    for i in range(len(X_array_SPDFT)):
        SST_SPDFT = SST_SPDFT + np.power((dE_list_total[i]),2)
        
    n_strech_and_bend_and_bond_bond = nstrech_potential + nbend_potential + nbond_bond_potential + nbond_angle_potential + n_angle_angle_potential

    deleted_columns_index = []
    c = 0
    modes = [1,2,3,4,5,6,7]
    for m in modes:
       for i in range(len(list_torsion_type[0])):
          if m in list_torsion_type[1][i]:
             if (list_torsion_type[0][i] == 'Rotatable') or (list_torsion_type[0][i] == 'CO_Rotatable'):
                index_torsion = n_strech_and_bend_and_bond_bond + c
                index_RD = index_RD_in_X_array[0].index(i)
                index_RD_in_X_array[1][index_RD].append(index_torsion)
             elif list_torsion_type[0][i] == 'Nonrotatable' or list_torsion_type[0][i] == 'Hindered':
                 index_torsion = n_strech_and_bend_and_bond_bond + c
                 list_dihedral_type_info_ks[3][i] = index_torsion
             elif list_torsion_type[0][i] == 'Linear':
                 index_torsion = n_strech_and_bend_and_bond_bond + c
                 list_dihedral_type_info_ks[3][i].append(index_torsion)
             c += 1
    if Flag_linear == True:
        modes_linear = [8,9]
        for m in modes_linear: 
            for i in range(len(linear_type_indices)):
                type_linear = linear_type_indices[i]
                if m in list_torsion_type[1][type_linear]:
                    index_torsion = n_strech_and_bend_and_bond_bond + c
                    linear_k_indicies[i].append(index_torsion)
                    c += 1
                    
        for i in range(len(linear_type_indices)):
              list_dihedral_type_info_ks[3][linear_type_indices[i]] = linear_k_indicies[i] 
    
    X_array_column_num = c + n_strech_and_bend_and_bond_bond
    X_array =  np.zeros(shape = (3*rows+len(X_array_SPDFT), X_array_column_num), dtype=float)
    for i in range(len(index_RD_in_X_array[0])):
        list_dihedral_type_info_ks[3][index_RD_in_X_array[0][i]] = index_RD_in_X_array[1][i]
else:
    if Flag_linear == True:
        c = 0
        modes_linear = [8,9]
        for m in modes_linear: 
            for i in range(len(linear_type_indices)):
                type_linear = linear_type_indices[i]
                if m in list_torsion_type[1][type_linear]:
                    index_torsion = n_strech_and_bend_and_bond_bond + c
                    linear_k_indicies[i].append(index_torsion)
                    c += 1
        for i in range(len(linear_type_indices)):
              list_dihedral_type_info_ks[3][linear_type_indices[i]] = linear_k_indicies[i]
              
    X_array_column_num = npotentials
    X_array =  np.zeros(shape = (3*rows, X_array_column_num), dtype=float)


### end of rotatable part 2


mode1_torsion_type = len(list_torsion_type[1])*[0]
mode2_torsion_type = len(list_torsion_type[1])*[0]
mode3_torsion_type = len(list_torsion_type[1])*[0]
mode4_torsion_type = len(list_torsion_type[1])*[0]
mode5_torsion_type = len(list_torsion_type[1])*[0]
mode6_torsion_type = len(list_torsion_type[1])*[0]
mode7_torsion_type = len(list_torsion_type[1])*[0]
mode8_torsion_type = len(list_torsion_type[1])*[0]
mode9_torsion_type = len(list_torsion_type[1])*[0]

for i in range(len(list_torsion_type[1])):
    for j in range(1,10):
        if j in list_torsion_type[1][i]:
            if j == 1:
                mode1_torsion_type[i] = 1
            if j == 2:
                mode2_torsion_type[i] = 1
            if j == 3:
                mode3_torsion_type[i] = 1
            if j == 4:
                mode4_torsion_type[i] = 1
            if j == 5:
                mode5_torsion_type[i] = 1
            if j == 6:
                mode6_torsion_type[i] = 1
            if j == 7:
                mode7_torsion_type[i] = 1
            if j == 8:
                mode8_torsion_type[i] = 1
            if j == 9:
                mode9_torsion_type[i] = 1
        
        
str_ADDT_CADT_CO = []
str_ADDT_CADT_CO_type = []
str_ADDT_CADT_CO.append(Material_name)
str_ADDT_CADT_CO_type.append(Material_name)
str_ADDT_CADT_CO = str_ADDT_CADT_CO + 7*[0] 
str_ADDT_CADT_CO_type = str_ADDT_CADT_CO_type + 7*[0]    
angle_threshold = 130*np.pi/180

equil_geom = geoms[0:natoms, :];            
params_vect_equil = calc_params_vect(equil_geom,bond_type,angle_type,dihedral_type,atom_vectors_indices,Lattice_matrix)

bonds_length_n = []
angles_theta_n = []
dihedrals_phi_n = []
dihedrals_angles_n = []
angles_bonds_n = []

bonds_length_n = params_vect_equil[1]
angles_theta_n = params_vect_equil[2]
dihedrals_phi_n = params_vect_equil[3]
dihedrals_angles_n = params_vect_equil[4]
angles_bonds_n = params_vect_equil[5]

bonds_length_avg = []
angles_theta_avg = []
dihedrals_phi_avg = []
dihedrals_angles_avg = []
angles_bonds_len_avg = []

for i in range(len(bonds_length_n)):
    bonds_length_avg.append([])
    ave_bond_length_type = sum(bonds_length_n[i])/len(bonds_length_n[i])    
    for j in range(len(bonds_length_n[i])):
        bonds_length_avg[i].append(ave_bond_length_type)
        
for i in range(len(angles_theta_n)):
    angles_theta_avg.append([])
    angles_bonds_len_avg.append([])
    angle_bond_length_1 = 0
    angle_bond_length_2 = 0
    
    for k in range(len(angles_theta_n[i])):
        angle_bond_length_1 += angles_bonds_n[i][k][0]
        angle_bond_length_2 += angles_bonds_n[i][k][1]
     
    avg_angle_bond_len_1 = angle_bond_length_1/len(angles_theta_n[i])
    avg_angle_bond_len_2 = angle_bond_length_2/len(angles_theta_n[i])
    
    ave_angles_theta_type = sum(angles_theta_n[i])/len(angles_theta_n[i])
    
    for j in range(len(angles_theta_n[i])):
        angles_theta_avg[i].append(ave_angles_theta_type)
        angles_bonds_len_avg[i].append([avg_angle_bond_len_1,avg_angle_bond_len_2])

dihedral_types_names = ['ADDT_MODE_ONE_ONLY','CADT_MODE_ONE_ONLY','ADDT_DIHEDRAL','CADT_DIHEDRAL','ADLD','CACO','ADCO']
for i in range(len(dihedrals_phi_n)):
    dihedrals_phi_avg.append([])
    dihedrals_angles_avg.append([])
    dihedral_angle_1 = 0
    dihedral_angle_2 = 0
    str_ADDT_CADT_CO_old = str_ADDT_CADT_CO.copy()
    flag_type = 0
    for k in range(len(dihedrals_phi_n[i])):
        dihedral_angle_1_i = dihedrals_angles_n[i][k][0]
        dihedral_angle_2_i = dihedrals_angles_n[i][k][1]
        
        if dihedral_angle_1_i < angle_threshold and dihedral_angle_2_i < angle_threshold:
                if list_torsion_type[0][i] == 'Rotatable':
                    str_ADDT_CADT_CO[4] += 1/float(list_counting_dihedral[i][k])
                    if flag_type == 0:
                        str_ADDT_CADT_CO_type[4] += 1
                        flag_type = 1
                elif list_torsion_type[0][i] == 'CO_Rotatable':
                    str_ADDT_CADT_CO[6] +=  1/float(list_counting_dihedral[i][k])
                    if flag_type == 0:
                        str_ADDT_CADT_CO_type[6] += 1
                        flag_type = 1
                else:
                    str_ADDT_CADT_CO[2] +=  1/float(list_counting_dihedral[i][k])
                    if flag_type == 0:
                        str_ADDT_CADT_CO_type[2] += 1
                        flag_type = 1
        else:
                if list_torsion_type[0][i] == 'Rotatable':
                    str_ADDT_CADT_CO[3] +=  1/float(list_counting_dihedral[i][k])
                    if flag_type == 0:
                        str_ADDT_CADT_CO_type[3] += 1
                        flag_type = 1
                elif list_torsion_type[0][i] == 'CO_Rotatable':
                    str_ADDT_CADT_CO[7] +=  1/float(list_counting_dihedral[i][k])
                    if flag_type == 0:
                        str_ADDT_CADT_CO_type[7] += 1
                        flag_type = 1
                elif list_torsion_type[0][i] == 'Nonrotatable' or list_torsion_type[0][i] == 'Hindered':
                    str_ADDT_CADT_CO[1] +=  1/float(list_counting_dihedral[i][k])
                    if flag_type == 0:
                        str_ADDT_CADT_CO_type[1] += 1
                        flag_type = 1
                elif list_torsion_type[0][i] == 'Linear':
                    str_ADDT_CADT_CO[5] +=  1/float(list_counting_dihedral[i][k])
                    if flag_type == 0:
                        str_ADDT_CADT_CO_type[5] += 1
                        flag_type = 1
        dihedral_angle_1 += dihedral_angle_1_i
        dihedral_angle_2 += dihedral_angle_2_i
    if str_ADDT_CADT_CO != str_ADDT_CADT_CO_old:
        index_diff = next(i for i in range(len(str_ADDT_CADT_CO)) if str_ADDT_CADT_CO[i] != str_ADDT_CADT_CO_old[i])
        list_dihedral_type_info_ks[1][i] = dihedral_types_names[index_diff-1]
    
    avg_dihedral_angle_1 = dihedral_angle_1/len(dihedrals_angles_n[i])
    avg_dihedral_angle_2 = dihedral_angle_2/len(dihedrals_angles_n[i])
    
    abs_dihedrals_phi_n = [abs(ele) for ele in dihedrals_phi_n[i]]
    ave_dihedrals_phi_type = sum(abs_dihedrals_phi_n)/len(dihedrals_phi_n[i])
    for j in range(len(dihedrals_phi_n[i])):
        if dihedrals_phi_n[i][j]>=0:
            sign = 1
        else:
            sign = -1 
        dihedrals_phi_avg[i].append(sign*ave_dihedrals_phi_type)
        dihedrals_angles_avg[i].append([avg_dihedral_angle_1,avg_dihedral_angle_2])

params_vect_equil[1] = bonds_length_avg
params_vect_equil[2] = angles_theta_avg
params_vect_equil[3] = dihedrals_phi_avg
params_vect_equil[4] = dihedrals_angles_avg
params_vect_equil[5] = angles_bonds_len_avg

print('\nThe following table is a breakdown of the torsion potential types (instnces) before force constants optimization:')
print('ADDT = angle-damped dihedral torsion')
print('CADT = constant amplitude dihedral torsion')
str1 = '{:^10}  {:^15}  {:^15}  {:^15}  {:^15}  {:^15}  {:^15}  {:^15}'.format('Material name','#ADDT non-rot/hind','#CADT non-rot/hind','#ADDT rot','#CADT rot','#ADLD','#CACO','#ADCO')
str_type_ins_0 = str_ADDT_CADT_CO_type[0]; str_type_ins_1 = '{} ({})'.format(str_ADDT_CADT_CO_type[1],int(str_ADDT_CADT_CO[1]));
str_type_ins_2 = '{} ({})'.format(str_ADDT_CADT_CO_type[2],int(str_ADDT_CADT_CO[2])); str_type_ins_3 = '{} ({})'.format(str_ADDT_CADT_CO_type[3],int(str_ADDT_CADT_CO[3]));
str_type_ins_4 = '{} ({})'.format(str_ADDT_CADT_CO_type[4],int(str_ADDT_CADT_CO[4])); str_type_ins_5 = '{} ({})'.format(str_ADDT_CADT_CO_type[5],int(str_ADDT_CADT_CO[5])); 
str_type_ins_6 = '{} ({})'.format(str_ADDT_CADT_CO_type[6],int(str_ADDT_CADT_CO[6])); str_type_ins_7 = '{} ({})'.format(str_ADDT_CADT_CO_type[7],int(str_ADDT_CADT_CO[7]));
str2 = '{:^10}      {:^15}      {:^15}   {:^15}   {:^15} {:^15}  {:^15}  {:^15}\n'.format(str_type_ins_0,str_type_ins_1,str_type_ins_2,str_type_ins_3,str_type_ins_4,str_type_ins_5,str_type_ins_6,str_type_ins_7)
print(str1)
print(str2)

dx = 0.0001;
dy = 0.0001;
dz = 0.0001;


for geom_number in range(1,ngeoms_train+1):
    i = geom_number*natoms
    single_geom = geoms[i-natoms:i, :]
    params_vect_single_geom = calc_params_vect(single_geom, bond_type, angle_type,dihedral_type,atom_vectors_indices,Lattice_matrix)
    
    for atom in range(1,natoms+1): 
        for coord in range(3):
            if coord == 0:
                displacement_vect = np.array([dx, 0.0, 0.0])
                X_row = 3*i - (3*natoms-1) + 3*(atom-1)
            elif coord == 1:
                displacement_vect = np.array([0.0, dy, 0.0])
                X_row = 3*i - (3*natoms-2) + 3*(atom-1)
            elif coord == 2:
                displacement_vect = np.array([0.0, 0.0, dz])
                X_row = 3*i - (3*natoms-3) + 3*(atom-1)
            displaced_array = calc_displaced_array(single_geom, atom, displacement_vect)
            params_vect_displaced = calc_params_vect_displaced_atoms(affected_bonds, affected_angles, affected_dihedrals, bond_type, angle_type, dihedral_type, atom, displaced_array, params_vect_single_geom, atom_vectors_indices, Lattice_matrix)
            stretch_potential = calc_delta_stretch_potential(affected_bonds, params_vect_single_geom[1], params_vect_displaced[1], params_vect_equil[1], list_counting_bonds, atom)
            bend_potential = calc_delta_bend_potential(affected_angles,params_vect_single_geom[2],params_vect_displaced[2],params_vect_equil[2],angle_3_4_0_type,atom)
            bond_bond_potential = calc_delta_bond_bond_potential(affected_angles,params_vect_single_geom[5],params_vect_displaced[5],params_vect_equil[5],atom)
            bond_angle_potential = calc_delta_bond_angle_potential(affected_angles,params_vect_single_geom[5],params_vect_displaced[5],params_vect_equil[5],params_vect_single_geom[2],params_vect_displaced[2],params_vect_equil[2],atom)
            angle_angle_potential = calc_delta_angle_angle_potential(affected_angle_pairs,params_vect_single_geom[2],params_vect_displaced[2],params_vect_equil[2],n_angle_angle_potential,atom)
            torsion1_potential = calc_delta_torsion1_potential(affected_dihedrals,params_vect_single_geom[3],params_vect_displaced[3],params_vect_equil[3],mode1_torsion_type,list_torsion_type[0],list_counting_dihedral,params_vect_single_geom[4],params_vect_displaced[4],params_vect_equil[4],adjusted_c_list,atom)
            torsion2_potential = calc_delta_torsion2_potential(affected_dihedrals,params_vect_single_geom[3],params_vect_displaced[3],params_vect_equil[3],mode2_torsion_type,list_torsion_type[0],list_counting_dihedral,params_vect_single_geom[4],params_vect_displaced[4],params_vect_equil[4],atom)
            torsion3_potential = calc_delta_torsion3_potential(affected_dihedrals,params_vect_single_geom[3],params_vect_displaced[3],params_vect_equil[3],mode3_torsion_type,list_torsion_type[0],list_counting_dihedral,params_vect_single_geom[4],params_vect_displaced[4],params_vect_equil[4],atom)
            torsion4_potential = calc_delta_torsion4_potential(affected_dihedrals,params_vect_single_geom[3],params_vect_displaced[3],params_vect_equil[3],mode4_torsion_type,list_torsion_type[0],list_counting_dihedral,params_vect_single_geom[4],params_vect_displaced[4],params_vect_equil[4],atom)
            torsion5_potential = calc_delta_torsion5_potential(affected_dihedrals,params_vect_single_geom[3],params_vect_displaced[3],params_vect_equil[3],mode5_torsion_type,list_torsion_type[0],list_counting_dihedral,list_sign_dihedral_type,params_vect_single_geom[4],params_vect_displaced[4],params_vect_equil[4],atom)
            torsion6_potential = calc_delta_torsion6_potential(affected_dihedrals,params_vect_single_geom[3],params_vect_displaced[3],params_vect_equil[3],mode6_torsion_type,list_torsion_type[0],list_counting_dihedral,list_sign_dihedral_type,params_vect_single_geom[4],params_vect_displaced[4],params_vect_equil[4],atom)
            torsion7_potential = calc_delta_torsion7_potential(affected_dihedrals,params_vect_single_geom[3],params_vect_displaced[3],params_vect_equil[3],mode7_torsion_type,list_torsion_type[0],list_counting_dihedral,list_sign_dihedral_type,params_vect_single_geom[4],params_vect_displaced[4],params_vect_equil[4],atom)
            torsion8_potential = calc_delta_torsion8_potential(affected_dihedrals,params_vect_single_geom[3],params_vect_displaced[3],mode8_torsion_type,list_torsion_type[0],list_counting_dihedral,params_vect_single_geom[4],params_vect_displaced[4],atom)
            torsion9_potential = calc_delta_torsion9_potential(affected_dihedrals,params_vect_single_geom[3],params_vect_displaced[3],mode9_torsion_type,list_torsion_type[0],list_counting_dihedral,params_vect_single_geom[4],params_vect_displaced[4],atom)

            
            counter = 0
            for index in range(len(stretch_potential)):
                X_array[X_row-1, index] = stretch_potential[index].item()

            counter += len(stretch_potential)
            if len(bend_potential) > 0:
                for index in range(len(bend_potential)):
                    X_array[X_row-1, index + counter] = bend_potential[index].item()

                counter += len(bend_potential)
            for index in range(len(bond_bond_potential)):
                X_array[X_row-1, index + counter] = bond_bond_potential[index].item()

            counter += len(bond_bond_potential)
            for index in range(len(bond_angle_potential)):
                X_array[X_row-1, index + counter] = bond_angle_potential[index].item()

            counter += len(bond_angle_potential)
            for index in range(len(angle_angle_potential)):
                X_array[X_row-1, index + counter] = angle_angle_potential[index].item()

            counter += len(angle_angle_potential)
            if len(torsion1_potential) > 0:
                for index in range(len(torsion1_potential)):
                    X_array[X_row-1, index + counter] = torsion1_potential[index].item()

                counter += len(torsion1_potential)
            if len(torsion2_potential) > 0:
                for index in range(len(torsion2_potential)):
                    X_array[X_row-1, index + counter] = torsion2_potential[index].item()

                counter += len(torsion2_potential)
            if len(torsion3_potential) > 0:
                for index in range(len(torsion3_potential)):
                    X_array[X_row-1, index + counter] = torsion3_potential[index].item()

                counter += len(torsion3_potential)
            if len(torsion4_potential) > 0:
                for index in range(len(torsion4_potential)):
                    X_array[X_row-1, index + counter] = torsion4_potential[index].item()

                counter += len(torsion4_potential)
            if len(torsion5_potential) > 0:
                for index in range(len(torsion5_potential)):
                    X_array[X_row-1, index + counter] = torsion5_potential[index].item()

                counter += len(torsion5_potential)
            if len(torsion6_potential) > 0:
                for index in range(len(torsion6_potential)):
                    X_array[X_row-1, index + counter] = torsion6_potential[index].item()

                counter += len(torsion6_potential)
            if len(torsion7_potential) > 0:
                for index in range(len(torsion7_potential)):
                    X_array[X_row-1, index + counter] = torsion7_potential[index].item()

                counter += len(torsion7_potential)
            if len(torsion8_potential) > 0:
                for index in range(len(torsion8_potential)):
                    X_array[X_row-1, index + counter] = torsion8_potential[index].item()

                counter += len(torsion8_potential)
            if len(torsion9_potential) > 0:
                for index in range(len(torsion9_potential)):
                    X_array[X_row-1, index + counter] = torsion9_potential[index].item()


#np.save(base_dir + 'X_array.npy', X_array)

#path_X = base_dir + 'X_array.npy'

#X_array =  np.load(path_X)


# rotatable part 3
if Flag_RD == 'True':
    index_rot = np.concatenate(index_RD_in_X_array[1])
    X_array[3*rows:,index_rot] = X_array_SPDFT
    
    dE_list_total = np.expand_dims(dE_list_total,axis=1)
    forces_energies = np.append(forces, dE_list_total, axis= 0)
    
    c1_final = np.zeros(shape = (2,X_array.shape[1]))
    c1_final[0,:] = 0
    c1_final[1,:] = np.inf
    c1_final[0,index_rot] = c1_SPDFT
    c1_final[0,(nstrech_potential+nbend_potential):(nstrech_potential+nbend_potential+nbond_bond_potential)] = -np.inf
    c1_final[0,(nstrech_potential+nbend_potential+nbond_bond_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential)] = -np.inf
    c1_final[0,(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential)] = -np.inf
    
    SST_forces = 0;
    for i in range(len(forces)):
        SST_forces = SST_forces + np.power((forces[i]),2).item();

    
    
    PF_final = np.zeros(shape = (1,X_array.shape[1]))
    w_forces_i = len(X_array)/SST_forces
    w_SPDFT_i = len(X_array)/SST_SPDFT
    
    
    for i in range(X_array.shape[1]):
        temp = 0
        for j in range(X_array.shape[0]):
            if j < 3*rows:
                w_r =  w_forces_i
            else:
                w_r =  w_SPDFT_i
             
            temp = temp + w_r*(X_array[j,i]**2)
        PF_final[0,i] = math.sqrt(temp/len(X_array))
    
    weights_forces = w_forces_i*np.ones(shape=(len(forces),1))
    weights_SPDFT = w_SPDFT_i*np.ones(shape=(len(dE_list_total),1))
    weights = np.append(weights_forces,weights_SPDFT,axis=0)

    # ---- PSD-constrained fit (replaces glmnet's unconstrained-except-box
    # LASSO): the bond_bond/bond_angle/angle_angle cross terms are fit
    # subject to their local coupling Hessians being positive-definite by
    # construction, instead of fitting freely then dropping/filtering
    # unstable terms after the fact. See cvxpy_constrained_solver.py. ----
    import sys as _sys
    _sys.path.insert(0, "/home/rgh_bio/savesteps_run/gemcitabine/Flexibility_parameters_calculations_final_input_files/GEMCIT/Average_with_cross")
    import cvxpy_constrained_solver as _cvxsolver

    _export = {
        "X_array": X_array,
        "forces_energies": forces_energies,
        "c1_final": c1_final,
        "PF_final": PF_final,
        "weights": weights,
        "natoms": natoms,
        "nstrech_potential": nstrech_potential,
        "nbend_potential": nbend_potential,
        "nbond_bond_potential": nbond_bond_potential,
        "nbond_angle_potential": nbond_angle_potential,
        "n_angle_angle_potential": n_angle_angle_potential,
        "list_stretch_type_info_ks_0": list_stretch_type_info_ks[0],
        "list_bend_type_info_ks_0": list_bend_type_info_ks[0],
        "list_bbc_type_info_ks_0": list_bbc_type_info_ks[0],
        "list_bac_type_info_ks_0": list_bac_type_info_ks[0],
        "list_aac_type_info_ks_0": list_aac_type_info_ks[0],
    }
    print('\n--------------------------------------------------------------------------------------------------------')
    print('\nPerforming PSD-constrained least-squares fitting of all interactions force-field parameters for the training set.')
    print('\nNumber of geometries in the training set are: ',ngeoms_train)
    _base_dir = "/home/rgh_bio/savesteps_run/gemcitabine/Flexibility_parameters_calculations_final_input_files/GEMCIT/Average/geometry_and_typing_files"
    b_final_0, b_final, _psd_info = _cvxsolver.solve(_export, _base_dir, lambda_grid_points=50, verbose=True)

    lambda_final = _psd_info["lambda_final"]
    n_parameters_final = _psd_info["n_params_final"]
    params_fit_lambda = [_psd_info["lambda_at_dense_end"], _psd_info["lambda_at_dense_end"]]
    params_fit_nparams = [_psd_info["n_params_at_dense_end"], _psd_info["n_params_at_dense_end"]]


        
    
    print('\nNumber of total flexibility parameters attempted for least-squares fitting: ', X_array.shape[1])
	
    print('\nNumber of flexibility parameters for all interactions when lambda -> 0 are: ', params_fit_nparams[-1])
    print('lambda -> 0 is: ',params_fit_lambda[-1])

    print('\nNumber of flexibility parameters for all interactions when lambda = lambda_best are: ', n_parameters_final)
    print('lambda = lambda_best is: ',lambda_final)
    
    print('\nThe set of all optimized forcefield parameters for lambda = labda_best is: ')
    print('\nparameters for stretches in units of eV/(Angstrom)^2:')
    print(b_final[0:nstrech_potential])
    print('\nparameters for bends in units of eV:')
    print(b_final[nstrech_potential:(nstrech_potential+nbend_potential)])
    print('\nparameters for bond_bond cross term in units of eV/(Angstrom)^2:')
    print(b_final[(nstrech_potential+nbend_potential):(nstrech_potential+nbend_potential+nbond_bond_potential)])
    print('\nparameters for bond_angle cross term in units of eV/Angstrom:')
    print(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential)])
    print('\nparameters for angle_angle cross term in units of (radian)^2:')
    print(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential)])
    print('\nparameters for torsions in units of eV:')
    print(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential):])
    
    print(' ')

    TA_UB_str = len(UB_type_list)
    TA_Bond_str = len(b_final[0:nstrech_potential])-TA_UB_str
    TA_bends = len(b_final[nstrech_potential:(nstrech_potential+nbend_potential)])
    TA_cross = len(b_final[(nstrech_potential+nbend_potential):(nstrech_potential+nbend_potential+nbond_bond_potential)])
    TA_bac = len(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential)])
    TA_aac = len(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential)])
    TA_rot = len(index_rot)
    TA_linear = 2*sum(mode8_torsion_type)
    TA_non_rot = len(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential):]) - TA_rot - TA_linear
    
    lambda0_UB_str = np.count_nonzero(b_final_0[UB_type_list])
    lambda0_Bond_str = np.count_nonzero(b_final_0[0:(nstrech_potential-len(UB_type_list))])
    lambda0_bends = np.count_nonzero(b_final_0[nstrech_potential:(nstrech_potential+nbend_potential)])
    lambda0_cross = np.count_nonzero(b_final_0[(nstrech_potential+nbend_potential):(nstrech_potential+nbend_potential+nbond_bond_potential)])
    lambda0_bac = np.count_nonzero(b_final_0[(nstrech_potential+nbend_potential+nbond_bond_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential)])
    lambda0_aac = np.count_nonzero(b_final_0[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential)])
    lambda0_rot = np.count_nonzero(b_final_0[index_rot])
    if TA_linear > 0:
        lambda0_linear = np.count_nonzero(b_final_0[-TA_linear:])
    else:
        lambda0_linear = 0
    lambda0_non_rot = np.count_nonzero(b_final_0[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential):]) - lambda0_rot - lambda0_linear
    
    lambdaB_UB_str = np.count_nonzero(b_final[UB_type_list])
    lambdaB_Bond_str = np.count_nonzero(b_final[0:(nstrech_potential-len(UB_type_list))])
    lambdaB_bends = np.count_nonzero(b_final[nstrech_potential:(nstrech_potential+nbend_potential)])
    lambdaB_cross = np.count_nonzero(b_final[(nstrech_potential+nbend_potential):(nstrech_potential+nbend_potential+nbond_bond_potential)])
    lambdaB_bac = np.count_nonzero(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential)])
    lambdaB_aac = np.count_nonzero(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential)])
    lambdaB_rot = np.count_nonzero(b_final[index_rot])
    if TA_linear > 0:
        lambdaB_linear = np.count_nonzero(b_final[-TA_linear:])
    else:
        lambdaB_linear = 0
    lambdaB_non_rot = np.count_nonzero(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential):]) - lambdaB_rot - lambdaB_linear
    
    str_table_K0 = '{:^35}  {:^50} \n'.format('Interaction','Number of k\'s')
    str_table_K1 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('           ','total_attempted','lambda -> 0','lambda = lambda_best')
    str_table_K2 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('bond streches',str(TA_Bond_str),str(lambda0_Bond_str),str(lambdaB_Bond_str))
    str_table_K3 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('Urey-Bradley streches',str(TA_UB_str),str(lambda0_UB_str),str(lambdaB_UB_str))
    str_table_K4 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('bends',str(TA_bends),str(lambda0_bends),str(lambdaB_bends))
    str_table_K5 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('bond-bond cross terms',str(TA_cross),str(lambda0_cross),str(lambdaB_cross))
    str_table_K5b = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('bond-angle cross terms',str(TA_bac),str(lambda0_bac),str(lambdaB_bac))
    str_table_K5c = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('angle-angle cross terms',str(TA_aac),str(lambda0_aac),str(lambdaB_aac))
    str_table_K6 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('non-rotatable/hindered dihedrals',str(TA_non_rot),str(lambda0_non_rot),str(lambdaB_non_rot))
    str_table_K7 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('dihedrals with linear bond angle',str(TA_linear),str(lambda0_linear),str(lambdaB_linear))
    str_table_K8 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('rotatable dihedrals',str(TA_rot),str(lambda0_rot),str(lambdaB_rot))
    
    
    print(str_table_K0); print(str_table_K1);print(str_table_K2); print(str_table_K3);print(str_table_K4); print(str_table_K5); print(str_table_K5b); print(str_table_K5c); print(str_table_K6); print(str_table_K7); print(str_table_K8)
    print(' ')
    str_table_K9 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('Total',str(len(b_final)),str(np.count_nonzero(b_final_0)),str(np.count_nonzero(b_final)))
    print(str_table_K9)
    
    
    
    pred_forces_final_0 = X_array[:3*rows,:] @ b_final_0
    pred_forces_final = X_array[:3*rows,:] @ b_final
    
    pred_energy_rot_0 = X_array[3*rows:,:] @ b_final_0
    pred_energy_rot = X_array[3*rows:,:] @ b_final
     
    file_atom_statistics_training = open(base_dir + 'atoms_statistics_training.txt','w+')
    file_atom_statistics_training.write(Material_name+'\n')
    str_1_head = '{:^10}  {:^10}  {:^20}  {:^20} \n'.format('atom_number','element','R-squared','RMSE(eV/Angstrom)')
    file_atom_statistics_training.write(str_1_head)
    for i in range(len(atomic_symbol_and_number)):
        atom_n_i = atomic_symbol_and_number[i][0]
        element_i = atomic_symbol_and_number[i][1]
        
        SSE_atom = 0;
        SST_atom = 0;
        for j in range(ngeoms_train):
            for k in range(3):
                index_jk = 3*i + k + j*3*natoms
                SSE_atom = SSE_atom + np.power(pred_forces_final[index_jk] - forces[index_jk],2);
                SST_atom = SST_atom + np.power((forces[index_jk]),2);
        
        R2_atom_i = 1 - SSE_atom/SST_atom
        RMSE_atom_i = np.sqrt(SSE_atom*natoms/len(forces))
        str_2_values = '{:^10}  {:^10}  {:^20}  {:^20} \n'.format(str(atom_n_i),element_i,str(R2_atom_i[0]),str(RMSE_atom_i[0]))
        file_atom_statistics_training.write(str_2_values)
    
    file_atom_statistics_training.close()
    
    #print('\nStatistics for training set in the parameters fitting for stretches, bends, and hindered/non-rotatable torsions:')
    
    
    SSE_forces_0 = 0;
    SSR_forces_0 = 0;
    SST_forces_0 = 0;
    for i in range(len(forces)):
    	SSE_forces_0 = SSE_forces_0 + np.power(pred_forces_final_0[i] - forces[i],2);
    	SSR_forces_0 = SSR_forces_0 + np.power((pred_forces_final_0[i]),2);
    	SST_forces_0 = SST_forces_0 + np.power((forces[i]),2);
        
    
    SSE_forces = 0;
    SSR_forces = 0;
    SST_forces = 0;
    for i in range(len(forces)):
    	SSE_forces = SSE_forces + np.power(pred_forces_final[i] - forces[i],2);
    	SSR_forces = SSR_forces + np.power((pred_forces_final[i]),2);
    	SST_forces = SST_forces + np.power((forces[i]),2).item();

    
    SSE_energy_rot_0 = 0;
    SSR_energy_rot_0 = 0;
    SST_energy_rot_0 = 0;
    for i in range(len(dE_list_total)):
    	SSE_energy_rot_0 = SSE_energy_rot_0 + np.power(pred_energy_rot_0[i] - dE_list_total[i],2);
    	SSR_energy_rot_0 = SSR_energy_rot_0 + np.power((pred_energy_rot_0[i]),2);
    	SST_energy_rot_0 = SST_energy_rot_0 + np.power((dE_list_total[i]),2);
        
    
    SSE_energy_rot = 0;
    SSR_energy_rot = 0;
    SST_energy_rot = 0;
    for i in range(len(dE_list_total)):
    	SSE_energy_rot = SSE_energy_rot + np.power(pred_energy_rot[i] - dE_list_total[i],2);
    	SSR_energy_rot = SSR_energy_rot + np.power((pred_energy_rot[i]),2);
    	SST_energy_rot = SST_energy_rot + np.power((dE_list_total[i]),2);
    
    print('\n--------------------------------------------------------------------------------------------------------')
    print('\nStatistics for training set in the parameters fitting for rotatable dihedrals:')
	
    print('\nSum of Squares Error (SSE) is in units of (eV)^2.')
    print('Sum of Squares Regression (SSR) is in units of (eV)^2.')
    print('Sum of Squares Total (SST) is in units of (eV)^2.')
    print('Root Mean Square Error (RMSE) is in units of (eV).')
    print('R-squared (R2) is unitless.\n')
    
    print('\nSSE for lambda -> 0 is: ', SSE_energy_rot_0)
    print('SSR for lambda -> 0 is: ', SSR_energy_rot_0)
    print('SST for lambda -> 0 is: ', SST_energy_rot_0)
    
    R2_energy_rot_0 = 1 - SSE_energy_rot_0/SST_energy_rot_0
    RMSE_energy_rot_0 = np.sqrt(SSE_energy_rot_0/len(dE_list_total))
    
    print('R2 for lambda -> 0 is: ',R2_energy_rot_0)
    print('RMSE for lambda -> 0 is: ',RMSE_energy_rot_0)
    
    print('\nSSE for lambda = lambda_best is: ', SSE_energy_rot)
    print('SSR for lambda = lambda_best is: ', SSR_energy_rot)
    print('SST for lambda = lambda_best is: ', SST_energy_rot)
    
    R2_energy_rot = 1 - SSE_energy_rot/SST_energy_rot
    RMSE_energy_rot = np.sqrt(SSE_energy_rot/len(dE_list_total))
    print('R2 for lambda = lambda_best is: ',R2_energy_rot)
    print('RMSE for lambda = lambda_best is: ',RMSE_energy_rot)
        
    print('\n--------------------------------------------------------------------------------------------------------')
    print('\nStatistics for training set in the parameters fitting for AIMD and Hessian calculations:')
	
    print('\nSum of Squares Error (SSE) is in units of (eV/Angstrom)^2.')
    print('Sum of Squares Regression (SSR) is in units of (eV/Angstrom)^2.')
    print('Sum of Squares Total (SST) is in units of (eV/Angstrom)^2.')
    print('Root Mean Square Error (RMSE) is in units of (eV/Angstrom).')
    print('R-squared (R2) is unitless.\n')
    
    print('\nSSE for lambda -> 0 is: ', SSE_forces_0)
    print('SSR for lambda -> 0 is: ', SSR_forces_0)
    print('SST for lambda -> 0 is: ', SST_forces_0)
    
    R2_forces_0 = 1 - SSE_forces_0/SST_forces_0
    RMSE_forces_0 = np.sqrt(SSE_forces_0/len(forces))
    
    print('R2 for lambda -> 0 is: ',R2_forces_0)
    print('RMSE for lambda -> 0 is: ',RMSE_forces_0)
    
    
    print('\nSSE for lambda = lambda_best is: ', SSE_forces)
    print('SSR for lambda = lambda_best is: ', SSR_forces)
    print('SST for lambda = lambda_best is: ', SST_forces)
    
    R2_forces = 1 - SSE_forces/SST_forces
    RMSE_forces = np.sqrt(SSE_forces/len(forces))
    print('R2 for lambda = lambda_best is: ',R2_forces)
    print('RMSE for lambda = lambda_best is: ',RMSE_forces)    
    
    
    print('\n========================================================================================================')
else:
    
    
    print('\n========================================================================================================')

    
    c1_final = np.zeros(shape = (2,X_array.shape[1]))
    c1_final[0,:] = 0
    c1_final[1,:] = np.inf
    c1_final[0,(nstrech_potential+nbend_potential):(nstrech_potential+nbend_potential+nbond_bond_potential)] = -np.inf
    c1_final[0,(nstrech_potential+nbend_potential+nbond_bond_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential)] = -np.inf
    c1_final[0,(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential)] = -np.inf
    
    PF_final = np.zeros(shape = (1,X_array.shape[1]))
    len_X_array = len(X_array)
    for i in range(X_array.shape[1]):
    	PF_final[0,i] = math.sqrt(sum([v**2 for v in X_array[:,i]])/len_X_array)
        
    params_fit = glmnet_python.glmnet(x = X_array.copy(), y = forces.copy(),family = 'gaussian',alpha = 1, nlambda = 100, thresh = 1e-10, cl = c1_final, penalty_factor = PF_final, standardize = False, standardize_resp = False, lambda_min = np.array([1e-5]), intr = False)   

    params_fit_lambda = params_fit['lambdau']
    params_fit_R2 = params_fit['dev']
    params_fit_nparams = params_fit['df']
    
    temp = 0
    lambda_j = params_fit_lambda[-1]
    R_sq_j = params_fit_R2[-1]
    n_params_j = params_fit_nparams[-1]
    for i in range(len(params_fit_lambda)-2,-1,-1):    
    	lambda_i = params_fit_lambda[i]
    	R_sq_i = params_fit_R2[i]
    	n_params_i = params_fit_nparams[i]
    	if n_params_i < n_params_j:
    		temp = ((3*natoms)/(1-R_sq_i))*((R_sq_j - R_sq_i)/(n_params_j - n_params_i))
    		
    		if temp > 0.5:
    			lambda_final = lambda_j
    			n_parameters_final = n_params_j
    			break
    		else:
    			lambda_j = lambda_i
    			R_sq_j = R_sq_i
    			n_params_j = n_params_i
    	
    b_final_0_= glmnetCoef(params_fit, s = scipy.float64([0]))
    b_final_ = glmnetCoef(params_fit, s = scipy.float64([lambda_final]))
    
    b_final_0 = np.zeros(shape = (len(b_final_0_[1:]), 1), dtype=float)
    b_final = np.zeros(shape = (len(b_final_[1:]), 1), dtype=float)
    
    for i in range(len(b_final_0)):
        b_final_0[i,0] = b_final_0_[i+1].item()

        b_final[i,0] = b_final_[i+1].item()

    
    #b_final = np.linalg.lstsq(X_array, forces, rcond=1.e-5)[0]
    print('\nPerforming least-squares fitting of all interactions force-field parameters for the training set.')
    print('\nNumber of geometries in the training set are: ',ngeoms_train)
    print('\nThe set of all optimized forcefield parameters for lambda = labda_best is: ')
    print('\nparameters for stretches in units of eV/(Angstrom)^2:')
    print(b_final[0:nstrech_potential])
    print('\nparameters for bends in units of eV:')
    print(b_final[nstrech_potential:(nstrech_potential+nbend_potential)])
    print('\nparameters for bond_bond cross terms in units of eV/(Angstrom)^2:')
    print(b_final[(nstrech_potential+nbend_potential):(nstrech_potential+nbend_potential+nbond_bond_potential)])
    print('\nparameters for bond_angle cross term in units of eV/Angstrom:')
    print(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential)])
    print('\nparameters for angle_angle cross term in units of (radian)^2:')
    print(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential)])
    print('\nparameters for torsions in units of eV:')
    print(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential):])
    print(' ')    
    TA_UB_str = len(UB_type_list)
    TA_Bond_str = len(b_final[0:nstrech_potential])-TA_UB_str
    TA_bends = len(b_final[nstrech_potential:(nstrech_potential+nbend_potential)])
    TA_cross = len(b_final[(nstrech_potential+nbend_potential):(nstrech_potential+nbend_potential+nbond_bond_potential)])
    TA_bac = len(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential)])
    TA_aac = len(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential)])
    TA_linear = 2*sum(mode8_torsion_type)
    TA_non_rot = len(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential):]) - TA_linear
    
    lambda0_UB_str = np.count_nonzero(b_final_0[UB_type_list])
    lambda0_Bond_str = np.count_nonzero(b_final_0[0:(nstrech_potential-len(UB_type_list))])
    lambda0_bends = np.count_nonzero(b_final_0[nstrech_potential:(nstrech_potential+nbend_potential)])
    lambda0_cross = np.count_nonzero(b_final_0[(nstrech_potential+nbend_potential):(nstrech_potential+nbend_potential+nbond_bond_potential)])
    lambda0_bac = np.count_nonzero(b_final_0[(nstrech_potential+nbend_potential+nbond_bond_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential)])
    lambda0_aac = np.count_nonzero(b_final_0[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential)])
    if TA_linear > 0:
        lambda0_linear = np.count_nonzero(b_final_0[-TA_linear:])
    else:
        lambda0_linear = 0
    lambda0_non_rot = np.count_nonzero(b_final_0[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential):]) - lambda0_linear
    
    lambdaB_UB_str = np.count_nonzero(b_final[UB_type_list])
    lambdaB_Bond_str = np.count_nonzero(b_final[0:(nstrech_potential-len(UB_type_list))])
    lambdaB_bends = np.count_nonzero(b_final[nstrech_potential:(nstrech_potential+nbend_potential)])
    lambdaB_cross = np.count_nonzero(b_final[(nstrech_potential+nbend_potential):(nstrech_potential+nbend_potential+nbond_bond_potential)])
    lambdaB_bac = np.count_nonzero(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential)])
    lambdaB_aac = np.count_nonzero(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential):(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential)])
    if TA_linear > 0:
        lambdaB_linear = np.count_nonzero(b_final[-TA_linear:])
    else:
        lambdaB_linear = 0
    lambdaB_non_rot = np.count_nonzero(b_final[(nstrech_potential+nbend_potential+nbond_bond_potential+nbond_angle_potential+n_angle_angle_potential):]) - lambdaB_linear
    
    str_table_K0 = '{:^35}  {:^50} \n'.format('Interaction','Number of k\'s')
    str_table_K1 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('           ','total_attempted','lambda -> 0','lambda = lambda_best')
    str_table_K2 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('bond streches',str(TA_Bond_str),str(lambda0_Bond_str),str(lambdaB_Bond_str))
    str_table_K3 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('Urey-Bradley streches',str(TA_UB_str),str(lambda0_UB_str),str(lambdaB_UB_str))
    str_table_K4 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('bends',str(TA_bends),str(lambda0_bends),str(lambdaB_bends))
    str_table_K5 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('bond-bond cross terms',str(TA_cross),str(lambda0_cross),str(lambdaB_cross))
    str_table_K5b = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('bond-angle cross terms',str(TA_bac),str(lambda0_bac),str(lambdaB_bac))
    str_table_K5c = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('angle-angle cross terms',str(TA_aac),str(lambda0_aac),str(lambdaB_aac))
    str_table_K6 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('non-rotatable/hindered dihedrals',str(TA_non_rot),str(lambda0_non_rot),str(lambdaB_non_rot))
    str_table_K7 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('dihedrals with linear bond angle',str(TA_linear),str(lambda0_linear),str(lambdaB_linear))
    
    print(str_table_K0); print(str_table_K1);print(str_table_K2); print(str_table_K3);print(str_table_K4); print(str_table_K5); print(str_table_K5b); print(str_table_K5c); print(str_table_K6); print(str_table_K7);
    print(' ')
    str_table_K8 = '{:^35}  {:^15}  {:^15}  {:^15} \n'.format('Total',str(len(b_final)),str(np.count_nonzero(b_final_0)),str(np.count_nonzero(b_final)))
    print(str_table_K8)
    
    pred_forces_0 = X_array @ b_final_0
    pred_forces = X_array @ b_final
    
    file_atom_statistics_training = open(base_dir + 'atoms_statistics_training.txt','w+')
    file_atom_statistics_training.write(Material_name+'\n')
    str_1_head = '{:^10}  {:^10}  {:^20}  {:^20} \n'.format('atom_number','element','R-squared','RMSE(eV/Angstrom)')
    file_atom_statistics_training.write(str_1_head)
    for i in range(len(atomic_symbol_and_number)):
        atom_n_i = atomic_symbol_and_number[i][0]
        element_i = atomic_symbol_and_number[i][1]
        
        SSE_atom = 0;
        SST_atom = 0;
        for j in range(ngeoms_train):
            for k in range(3):
                index_jk = 3*i + k + j*3*natoms
                SSE_atom = SSE_atom + np.power(pred_forces[index_jk] - forces[index_jk],2);
                SST_atom = SST_atom + np.power((forces[index_jk]),2);
        
        R2_atom_i = 1 - SSE_atom/SST_atom
        RMSE_atom_i = np.sqrt(SSE_atom*natoms/len(X_array))
        str_2_values = '{:^10}  {:^10}  {:^20}  {:^20} \n'.format(str(atom_n_i),element_i,str(R2_atom_i[0]),str(RMSE_atom_i[0]))
        file_atom_statistics_training.write(str_2_values)
    
    file_atom_statistics_training.close()
    
    print('\n--------------------------------------------------------------------------------------------------------')
    print('\nStatistics for training set:')
    
    print('\nSum of Squares Error (SSE) is in units of (eV/Angstrom)^2.')
    print('Sum of Squares Regression (SSR) is in units of (eV/Angstrom)^2.')
    print('Sum of Squares Total (SST) is in units of (eV/Angstrom)^2.')
    print('Root Mean Square Error (RMSE) is in units of (eV/Angstrom).')
    print('R-squared (R2) is unitless.\n')
    
    print('Number of total flexibility parameters attempted for least-squares fitting: ', X_array.shape[1])
	
    print('\nNumber of flexibility parameters when lambda -> 0 are: ', params_fit_nparams[-1])
    print('lambda -> 0 is: ',params_fit_lambda[-1])
    
    
    SSE_0 = 0;
    SSR_0 = 0;
    SST_0 = 0;
    
    for i in range(len(X_array)):
        SSE_0 = SSE_0 + np.power(pred_forces_0[i] - forces[i],2);
        SSR_0 = SSR_0 + np.power((pred_forces_0[i]),2);
        SST_0 = SST_0 + np.power((forces[i]),2);
    
    print('\nSSE for lambda -> 0 is: ', SSE_0)
    print('SSR for lambda -> 0 is: ', SSR_0)
    print('SST for lambda -> 0 is: ', SST_0)
    
    R2_0 = 1 - SSE_0/SST_0
    RMSE_0 = np.sqrt(SSE_0/len(X_array))
    
    print('R2 for lambda -> 0 is: ',R2_0)
    print('RMSE for lambda -> 0 is: ',RMSE_0)
    
    
    print('\nNumber of flexibility parameters when lambda = lambda_best are: ', n_parameters_final)
    print('lambda = lambda_best is: ',lambda_final)
    
    SSE = 0;
    SST = 0;
    SSR = 0;
    for i in range(len(X_array)):
        SSE = SSE + np.power(pred_forces[i] - forces[i],2);
        SSR = SSR + np.power((pred_forces[i]),2);
        SST = SST + np.power((forces[i]),2);
    
    print('\nSSE for lambda = lambda_best is: ', SSE)
    print('SSR for lambda = lambda_best is: ', SSR)
    print('SST for lambda = lambda_best is: ', SST)
    
    R2 = 1 - SSE/SST
    RMSE = np.sqrt(SSE/len(X_array))
    print('R2 for lambda = lambda_best is: ',R2)
    print('RMSE for lambda = lambda_best is: ',RMSE)
    
    f1_bonds.close()
    f2_att_bonds.close()
    f3_angles.close()
    f4_att_angles.close()
    f5_dihedrals.close()
    f6_att_dihedrals.close() 
    f7_att_with_coordinates.close()
    geom_forces.close()
    f9_xyz.close()
    
    print('\n========================================================================================================')


# print force constants

path_optimized_force_constants = base_dir + Material_name +'_optimized_force_constants.txt'
file_optimized_force_constants = open(path_optimized_force_constants,'w+')

file_optimized_force_constants.write('parameters for stretches in units of eV/(Angstrom)^2:\n\n')
str_fc_str = '{:^15} {:^15} {:^20}\n'.format('type','potential_name','k_values')
file_optimized_force_constants.write(str_fc_str)
for i in range(len(list_stretch_type_info_ks[0])):
    type_str_i = list_stretch_type_info_ks[0][i]
    type_RASPA_str_i = list_stretch_type_info_ks[1][i]
    k_val_str_index_i = list_stretch_type_info_ks[2][i]
    if b_final[k_val_str_index_i] != 0:
        str_str_i = '{:^15} {:^15} {:^20}\n'.format(str(type_str_i),type_RASPA_str_i,str(b_final[k_val_str_index_i]))
        file_optimized_force_constants.write(str_str_i)

file_optimized_force_constants.write('\n\nparameters for bends in units of eV:\n\n')
str_fc_bend = '{:^15} {:^15} {:^20}\n'.format('type','potential_name','k_values')
file_optimized_force_constants.write(str_fc_bend)
for i in range(len(list_bend_type_info_ks[0])):
    type_bnd_i = list_bend_type_info_ks[0][i]
    type_RASPA_bnd_i = list_bend_type_info_ks[1][i]
    k_val_bnd_index_i = list_bend_type_info_ks[2][i]
    if b_final[k_val_bnd_index_i] != 0:
        str_bnd_i = '{:^15} {:^15} {:^20}\n'.format(str(type_bnd_i),type_RASPA_bnd_i,str(b_final[k_val_bnd_index_i]))
        file_optimized_force_constants.write(str_bnd_i)
    
file_optimized_force_constants.write('\n\nparameters for bond_bond cross terms in units of eV/(Angstrom)^2:\n\n')
str_fc_bbc = '{:^15} {:^30} {:^30}\n'.format('type','potential_name','k_values')
file_optimized_force_constants.write(str_fc_bbc)
for i in range(len(list_bbc_type_info_ks[0])):
    type_bbc_i = list_bbc_type_info_ks[0][i]
    type_RASPA_bbc_i = list_bbc_type_info_ks[1][i]
    k_val_bbc_index_i = list_bbc_type_info_ks[2][i]
    if b_final[k_val_bbc_index_i] != 0:
        str_bbc_i = '{:^15} {:^30} {:^30}\n'.format(str(type_bbc_i),type_RASPA_bbc_i,str(b_final[k_val_bbc_index_i]))
        file_optimized_force_constants.write(str_bbc_i)
    
file_optimized_force_constants.write('\n\nparameters for bond_angle cross terms in units of eV/Angstrom:\n\n')
str_fc_bac = '{:^15} {:^30} {:^30}\n'.format('type','potential_name','k_values')
file_optimized_force_constants.write(str_fc_bac)
for i in range(len(list_bac_type_info_ks[0])):
    type_bac_i = list_bac_type_info_ks[0][i]
    type_RASPA_bac_i = list_bac_type_info_ks[1][i]
    k_val_bac_index_i = list_bac_type_info_ks[2][i]
    if b_final[k_val_bac_index_i] != 0:
        str_bac_i = '{:^15} {:^30} {:^30}\n'.format(str(type_bac_i),type_RASPA_bac_i,str(b_final[k_val_bac_index_i]))
        file_optimized_force_constants.write(str_bac_i)
    
file_optimized_force_constants.write('\n\nparameters for angle_angle cross terms in units of (radian)^2:\n\n')
str_fc_aac = '{:^15} {:^30} {:^30}\n'.format('type','potential_name','k_values')
file_optimized_force_constants.write(str_fc_aac)
for i in range(len(list_aac_type_info_ks[0])):
    type_aac_i = list_aac_type_info_ks[0][i]
    type_RASPA_aac_i = list_aac_type_info_ks[1][i]
    k_val_aac_index_i = list_aac_type_info_ks[2][i]
    if b_final[k_val_aac_index_i] != 0:
        str_aac_i = '{:^15} {:^30} {:^30}\n'.format(str(type_aac_i),type_RASPA_aac_i,str(b_final[k_val_aac_index_i]))
        file_optimized_force_constants.write(str_aac_i)
    
file_optimized_force_constants.write('\n\nparameters for torsions in units of eV:\n\n')
str_fc_tor = '{:^15} {:^30} {:^30} {:^40}\n'.format('type','potential_name','selected_modes','k_values')
file_optimized_force_constants.write(str_fc_tor)
for i in range(len(list_dihedral_type_info_ks[0])):
    type_tor_i = list_dihedral_type_info_ks[0][i]
    type_RASPA_tor_i = list_dihedral_type_info_ks[1][i]
    selected_modes_i = list_dihedral_type_info_ks[2][i]
    k_val_tor_index_i = list_dihedral_type_info_ks[3][i]
    if type_tor_i in adjusted_c_list[0]:
        index_CO = adjusted_c_list[0].index(type_tor_i)
        selected_modes_i = adjusted_c_list[1][index_CO]
        adjusted_c_values = adjusted_c_list[2][index_CO]
        if b_final[k_val_tor_index_i][0] != 0:
            non_zero_constrained_c_values = [adjusted_c_values[x-1]*b_final[k_val_tor_index_i][0] for x in selected_modes_i]
            k_new = [x[0] for x in non_zero_constrained_c_values]
            str_tor_i = '{:^15} {:^30} {:^30} {:^40}\n'.format(str(type_tor_i),type_RASPA_tor_i,str(selected_modes_i),str(k_new))
            file_optimized_force_constants.write(str_tor_i)
    if list_torsion_type[0][type_tor_i] == 'Rotatable' or list_torsion_type[0][type_tor_i] == 'Linear':
        k_values = [x[0] for x in b_final[k_val_tor_index_i]]
        new_selected_modes = []
        new_ks = []
        c = 0
        for k_index in range(len(k_values)):
            if k_values[k_index] != 0:
                new_selected_modes.append(selected_modes_i[k_index])
                new_ks.append(k_values[k_index])
                c = 1
        if c==1:
            str_tor_i = '{:^15} {:^30} {:^30} {:^40}\n'.format(str(type_tor_i),type_RASPA_tor_i,str(new_selected_modes),str(new_ks))
            file_optimized_force_constants.write(str_tor_i)
    elif list_torsion_type[0][type_tor_i] == 'Nonrotatable' or list_torsion_type[0][type_tor_i] == 'Hindered':
        k_values = [x for x in b_final[k_val_tor_index_i]]
        new_selected_modes = []
        new_ks = []
        c = 0
        for k_index in range(len(k_values)):
            if k_values[k_index] != 0:
                new_selected_modes.append(selected_modes_i[k_index])
                new_ks.append(k_values[k_index])
                c = 1
        if c == 1:       
            str_tor_i = '{:^15} {:^30} {:^30} {:^40}\n'.format(str(type_tor_i),type_RASPA_tor_i,str(new_selected_modes),str(new_ks))
            file_optimized_force_constants.write(str_tor_i)                                                

file_optimized_force_constants.close()

### validation
                                                                                                       
path8_geom_force_array_val = base_dir + 'geom_forces_array_' + Material_name +'_val.csv'

geom_forces_val = open(path8_geom_force_array_val,'r');
lines_geom_forces_val = geom_forces_val.readlines()
geom_forces_array_val = np.empty(shape = (0,6),dtype=float)
geoms_val = np.empty(shape = (0,3),dtype=float)


geoms_val = np.zeros(shape = (len(lines_geom_forces_val), 3), dtype=float)
geom_forces_array_val = np.zeros(shape = (len(lines_geom_forces_val), 6), dtype=float)
ngeoms_val = round(len(lines_geom_forces_val)/natoms)

print('\nNumber of geometries in the validation set are: ',ngeoms_val)

for i in range(len(lines_geom_forces_val)):
    line_i = [float(j) for j in lines_geom_forces_val[i].split(',')]
    row_geom = np.array(line_i[0:3])
    row_geom_force = np.array(line_i)
    geoms_val[i,:] = row_geom
    geom_forces_array_val[i,:] = row_geom_force

rows_val = len(geom_forces_array_val)
forces_val = np.zeros(shape = (3*rows_val, 1), dtype=float)

for i in range(1,rows_val+1):
     if i % natoms == 0:
         c=0
         for j in range(1,natoms+1):
            forces_val[3*i - j - 2 - c , 0] = geom_forces_array_val[i-j , 3]
            forces_val[3*i - j - 1 - c, 0] = geom_forces_array_val[i-j , 4]
            forces_val[3*i - j - c, 0] = geom_forces_array_val[i-j , 5];
            c = 2*j
          
equil_geom_val = geoms_val[0:natoms, :];            

X_array_val =  np.zeros(shape = (3*rows_val, X_array_column_num), dtype=float)
dx = 0.0001;
dy = 0.0001;
dz = 0.0001;


for geom_number_val in range(1,ngeoms_val+1):
    i = geom_number_val*natoms
    single_geom_val = geoms_val[i-natoms:i, :]
    params_vect_single_geom_val = calc_params_vect(single_geom_val,bond_type,angle_type,dihedral_type,atom_vectors_indices,Lattice_matrix)
    
    for atom in range(1,natoms+1):
        for coord in range(3):
            if coord == 0:
                displacement_vect = np.array([dx, 0.0, 0.0])
                X_row_val = 3*i - (3*natoms-1) + 3*(atom-1)
            elif coord == 1:
                displacement_vect = np.array([0.0, dy, 0.0])
                X_row_val = 3*i - (3*natoms-2) + 3*(atom-1)
            elif coord == 2:
                displacement_vect = np.array([0.0, 0.0, dz])
                X_row_val = 3*i - (3*natoms-3) + 3*(atom-1)
            displaced_array_val = calc_displaced_array(single_geom_val, atom, displacement_vect)
            params_vect_displaced_val = calc_params_vect_displaced_atoms(affected_bonds, affected_angles, affected_dihedrals, bond_type, angle_type, dihedral_type, atom, displaced_array_val, params_vect_single_geom_val, atom_vectors_indices, Lattice_matrix)
            stretch_potential_val = calc_delta_stretch_potential(affected_bonds, params_vect_single_geom_val[1], params_vect_displaced_val[1], params_vect_equil[1], list_counting_bonds, atom)
            bend_potential_val = calc_delta_bend_potential(affected_angles,params_vect_single_geom_val[2],params_vect_displaced_val[2],params_vect_equil[2],angle_3_4_0_type,atom)
            bond_bond_potential_val = calc_delta_bond_bond_potential(affected_angles,params_vect_single_geom_val[5],params_vect_displaced_val[5],params_vect_equil[5],atom)
            bond_angle_potential_val = calc_delta_bond_angle_potential(affected_angles,params_vect_single_geom_val[5],params_vect_displaced_val[5],params_vect_equil[5],params_vect_single_geom_val[2],params_vect_displaced_val[2],params_vect_equil[2],atom)
            angle_angle_potential_val = calc_delta_angle_angle_potential(affected_angle_pairs,params_vect_single_geom_val[2],params_vect_displaced_val[2],params_vect_equil[2],n_angle_angle_potential,atom)
            torsion1_potential_val = calc_delta_torsion1_potential(affected_dihedrals,params_vect_single_geom_val[3],params_vect_displaced_val[3],params_vect_equil[3],mode1_torsion_type,list_torsion_type[0],list_counting_dihedral,params_vect_single_geom_val[4],params_vect_displaced_val[4],params_vect_equil[4],adjusted_c_list,atom)
            torsion2_potential_val = calc_delta_torsion2_potential(affected_dihedrals,params_vect_single_geom_val[3],params_vect_displaced_val[3],params_vect_equil[3],mode2_torsion_type,list_torsion_type[0],list_counting_dihedral,params_vect_single_geom_val[4],params_vect_displaced_val[4],params_vect_equil[4],atom)
            torsion3_potential_val = calc_delta_torsion3_potential(affected_dihedrals,params_vect_single_geom_val[3],params_vect_displaced_val[3],params_vect_equil[3],mode3_torsion_type,list_torsion_type[0],list_counting_dihedral,params_vect_single_geom_val[4],params_vect_displaced_val[4],params_vect_equil[4],atom)
            torsion4_potential_val = calc_delta_torsion4_potential(affected_dihedrals,params_vect_single_geom_val[3],params_vect_displaced_val[3],params_vect_equil[3],mode4_torsion_type,list_torsion_type[0],list_counting_dihedral,params_vect_single_geom_val[4],params_vect_displaced_val[4],params_vect_equil[4],atom)
            torsion5_potential_val = calc_delta_torsion5_potential(affected_dihedrals,params_vect_single_geom_val[3],params_vect_displaced_val[3],params_vect_equil[3],mode5_torsion_type,list_torsion_type[0],list_counting_dihedral,list_sign_dihedral_type,params_vect_single_geom_val[4],params_vect_displaced_val[4],params_vect_equil[4],atom)
            torsion6_potential_val = calc_delta_torsion6_potential(affected_dihedrals,params_vect_single_geom_val[3],params_vect_displaced_val[3],params_vect_equil[3],mode6_torsion_type,list_torsion_type[0],list_counting_dihedral,list_sign_dihedral_type,params_vect_single_geom_val[4],params_vect_displaced_val[4],params_vect_equil[4],atom)
            torsion7_potential_val = calc_delta_torsion7_potential(affected_dihedrals,params_vect_single_geom_val[3],params_vect_displaced_val[3],params_vect_equil[3],mode7_torsion_type,list_torsion_type[0],list_counting_dihedral,list_sign_dihedral_type,params_vect_single_geom_val[4],params_vect_displaced_val[4],params_vect_equil[4],atom)
            torsion8_potential_val = calc_delta_torsion8_potential(affected_dihedrals,params_vect_single_geom_val[3],params_vect_displaced_val[3],mode8_torsion_type,list_torsion_type[0],list_counting_dihedral,params_vect_single_geom_val[4],params_vect_displaced_val[4],atom)
            torsion9_potential_val = calc_delta_torsion9_potential(affected_dihedrals,params_vect_single_geom_val[3],params_vect_displaced_val[3],mode9_torsion_type,list_torsion_type[0],list_counting_dihedral,params_vect_single_geom_val[4],params_vect_displaced_val[4],atom)

            counter = 0
            for index in range(len(stretch_potential_val)):
                X_array_val[X_row_val-1, index] = stretch_potential_val[index].item()

            counter += len(stretch_potential_val)
            if len(bend_potential_val) > 0:
                for index in range(len(bend_potential_val)):
                    X_array_val[X_row_val-1, index + counter] = bend_potential_val[index].item()

                counter += len(bend_potential_val)
            for index in range(len(bond_bond_potential_val)):
                X_array_val[X_row_val-1, index + counter] = bond_bond_potential_val[index].item()

            counter += len(bond_bond_potential_val)
            for index in range(len(bond_angle_potential_val)):
                X_array_val[X_row_val-1, index + counter] = bond_angle_potential_val[index].item()

            counter += len(bond_angle_potential_val)
            for index in range(len(angle_angle_potential_val)):
                X_array_val[X_row_val-1, index + counter] = angle_angle_potential_val[index].item()

            counter += len(angle_angle_potential_val)
            if len(torsion1_potential_val) > 0:
                for index in range(len(torsion1_potential_val)):
                    X_array_val[X_row_val-1, index + counter] = torsion1_potential_val[index].item()

            counter += len(torsion1_potential_val)
            if len(torsion2_potential_val) > 0:
                for index in range(len(torsion2_potential_val)):
                    X_array_val[X_row_val-1, index + counter] = torsion2_potential_val[index].item()

                counter += len(torsion2_potential_val)
            if len(torsion3_potential_val) > 0:
                for index in range(len(torsion3_potential_val)):
                    X_array_val[X_row_val-1, index + counter] = torsion3_potential_val[index].item()

                counter += len(torsion3_potential_val)
            if len(torsion4_potential_val) > 0:
                for index in range(len(torsion4_potential_val)):
                    X_array_val[X_row_val-1, index + counter] = torsion4_potential_val[index].item()

                counter += len(torsion4_potential_val)
            if len(torsion5_potential_val) > 0:
                for index in range(len(torsion5_potential_val)):
                    X_array_val[X_row_val-1, index + counter] = torsion5_potential_val[index].item()

                counter += len(torsion5_potential_val)
            if len(torsion6_potential_val) > 0:
                for index in range(len(torsion6_potential_val)):
                    X_array_val[X_row_val-1, index + counter] = torsion6_potential_val[index].item()

                counter += len(torsion6_potential_val)
            if len(torsion7_potential_val) > 0:
                for index in range(len(torsion7_potential_val)):
                    X_array_val[X_row_val-1, index + counter] = torsion7_potential_val[index].item()

                counter += len(torsion7_potential_val)
            if len(torsion8_potential_val) > 0:
                for index in range(len(torsion8_potential_val)):
                    X_array_val[X_row_val-1, index + counter] = torsion8_potential_val[index].item()

                counter += len(torsion8_potential_val)
            if len(torsion9_potential_val) > 0:
                for index in range(len(torsion9_potential_val)):
                    X_array_val[X_row_val-1, index + counter] = torsion9_potential_val[index].item()

                    
pred_forces_val = X_array_val @ b_final;

file_atom_statistics_validation = open(base_dir + 'atoms_statistics_validation.txt','w+')
file_atom_statistics_validation.write(Material_name+'\n')
str_1_head = '{:^10}  {:^10}  {:^20}  {:^20} \n'.format('atom_number','element','R-squared','RMSE(eV/Angstrom)')
file_atom_statistics_validation.write(str_1_head)
for i in range(len(atomic_symbol_and_number)):
    atom_n_i = atomic_symbol_and_number[i][0]
    element_i = atomic_symbol_and_number[i][1]
    
    SSE_atom = 0;
    SST_atom = 0;
    for j in range(ngeoms_val):
        for k in range(3):
            index_jk = 3*i + k + j*3*natoms
            SSE_atom = SSE_atom + np.power(pred_forces_val[index_jk] - forces_val[index_jk],2);
            SST_atom = SST_atom + np.power((forces_val[index_jk]),2);
    
    R2_atom_i = 1 - SSE_atom/SST_atom
    RMSE_atom_i = np.sqrt(SSE_atom*natoms/len(X_array))
    str_2_values = '{:^10}  {:^10}  {:^20}  {:^20} \n'.format(str(atom_n_i),element_i,str(R2_atom_i[0]),str(RMSE_atom_i[0]))
    file_atom_statistics_validation.write(str_2_values)

file_atom_statistics_validation.close()

print('\nStatistics for validation set:')

SSE_val = 0;
SST_val = 0;
SSR_val = 0;
for i in range(len(X_array_val)):
    SSE_val = SSE_val + np.power(forces_val[i] - pred_forces_val[i],2) 
    SSR_val = SSR_val + np.power((pred_forces_val[i]),2)
    SST_val = SST_val + np.power((forces_val[i]),2)

print('\nSum of Squares Error (SSE) is in units of (eV/Angstrom)^2.')
print('Sum of Squares Regression (SSR) is in units of (eV/Angstrom)^2.')
print('Sum of Squares Total (SST) is in units of (eV/Angstrom)^2.')
print('Root Mean Square Error (RMSE) is in units of (eV/Angstrom).')
print('R-squared (R2) is unitless.\n')

print('\nSSE validation is: ', SSE_val)
print('SSR validation is: ', SSR_val)
print('SST validation is: ', SST_val)

R2_val = 1 - SSE_val/SST_val
RMSE_val = np.sqrt(SSE_val/len(X_array_val))

print('R2 validation is: ',R2_val)
print('RMSE validation is: ',RMSE_val)

pred_forces_val_0 = X_array_val @ b_final_0
SSE_val_0 = 0
SST_val_0 = 0
for i in range(len(X_array_val)):
    SSE_val_0 = SSE_val_0 + np.power(forces_val[i] - pred_forces_val_0[i],2)
    SST_val_0 = SST_val_0 + np.power((forces_val[i]),2)
R2_val_0 = 1 - SSE_val_0/SST_val_0
RMSE_val_0 = np.sqrt(SSE_val_0/len(X_array_val))
print(chr(10) + 'Validation stats using lambda -> 0 coefficients (no extra term removal beyond the box constraints):')
print('SSE validation (lambda->0) is: ', SSE_val_0)
print('SST validation (lambda->0) is: ', SST_val_0)
print('R2 validation (lambda->0) is: ', R2_val_0)
print('RMSE validation (lambda->0) is: ', RMSE_val_0)

geom_forces_val.close()
end_time = time.time()

runtime_seconds = end_time - start_time

# Convert runtime to hours
runtime_hours = runtime_seconds / 3600  # 1 hour = 3600 seconds

# Format the result to display two decimal places
formatted_runtime_hours = "{:.2f}".format(runtime_hours)

print("\nRuntime in hours:", formatted_runtime_hours)
print()