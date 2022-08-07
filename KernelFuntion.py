import numpy as np
import copy
import math

def RBF_Self_K(input_x, input_xp, signal_variance, length_scale):
    K_matrix = []
    K_Array = []
    for i in input_x:
        for j in input_xp:
            dis = abs(i[0] - j[0])
            dis = dis**2
            k_value = signal_variance * math.exp(-dis / (2 * length_scale**2))
            K_Array.append(copy.deepcopy(k_value))
        K_matrix.append(copy.deepcopy(K_Array))
        K_Array.clear()
    K_matrix = np.array(K_matrix)
    return K_matrix

def Linear_Self_K(input_x, input_xp, output_variance, offset = 0):
    K_matrix = []
    K_Array = []
    for i in input_x:
        for j in input_xp:
            k_value = (i[0] - offset) * (j[0] - offset) * output_variance
            K_Array.append(copy.deepcopy(k_value))
        K_matrix.append(copy.deepcopy(K_Array))
        K_Array.clear()
    K_matrix = np.array(K_matrix)
    return K_matrix

def RQK_Self_K (input_x, input_xp, output_variance, parameter_alpha, lengthscale):
    K_matrix = []
    K_Array = []
    for i in input_x:
        for j in input_xp:
            k_value = output_variance * ((1 + (i[0] - j[0]) ** 2 / (2 * parameter_alpha * (lengthscale ** 2))) ** (-parameter_alpha))
            K_Array.append(copy.deepcopy(k_value))
        K_matrix.append(copy.deepcopy(K_Array))
        K_Array.clear()
    K_matrix = np.array(K_matrix)
    return K_matrix

def Polynomial_Self_K(input_x, input_xp,output_variance, zeta, gamma, constant, Q):
    K_matrix = []
    K_Array = []
    for i in input_x:
        for j in input_xp:
            k_value = output_variance * (zeta + gamma * (i[0] - constant) * (j[0] - constant)) ** Q
            K_Array.append(copy.deepcopy(k_value))
        K_matrix.append(copy.deepcopy(K_Array))
        K_Array.clear()
    K_matrix = np.array(K_matrix)
    return K_matrix

def Periodic_Self_K(input_x, input_xp, output_variance, period, lengthscale):
    K_matrix = []
    K_Array = []
    for i in input_x:
        for j in input_xp:
            k_value = output_variance * math.exp(-1 * (2 * (math.sin(math.pi * abs(i[0] - j[0]) / period)) ** 2) / lengthscale)
            K_Array.append(copy.deepcopy(k_value))
        K_matrix.append(copy.deepcopy(K_Array))
        K_Array.clear()
    K_matrix = np.array(K_matrix)
    return K_matrix

def Sigmoid_Self_K(input_x, input_xp, output_variance, alpha, constant):
    K_matrix = []
    K_Array = []
    for i in input_x:
        for j in input_xp:
            k_value = output_variance * math.tanh(alpha * i[0] * j[0] + constant)
            K_Array.append(copy.deepcopy(k_value))
        K_matrix.append(copy.deepcopy(K_Array))
        K_Array.clear()
    K_matrix = np.array(K_matrix)
    return K_matrix

def WhiteNoise_Self_K(input_x, input_xp, Noise):
    K_matrix = []
    K_Array = []
    for i in input_x:
        for j in input_xp:
            if i[0] == j[0]:
                k_value = 0
            else:
                k_value = Noise
            K_Array.append(copy.deepcopy(k_value))
        K_matrix.append(copy.deepcopy(K_Array))
        K_Array.clear()
    K_matrix = np.array(K_matrix)
    return K_matrix