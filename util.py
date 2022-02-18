import numpy as np


def create_first_matrix(matrix):
    if (type(matrix) == np.ndarray and matrix[0] == 1 and  matrix[1] == 2 and  matrix[2] == 3):
        print("SUCCESS")
    else:
        print("FAILURE")

def my_new_array():
    return np.array([[0] * 5, [3] * 5])

def check_random_matrix(a1, a2, a3):
    if (a1.shape != (2, 2)): print("FAILURE")
    if (a2.shape != (4, 4)): print("FAILURE")
    if (a3.shape != (5, 2)): print("FAILURE")
    print("SUCCESS")

def check_mul(a4):
    if (a4.shape != (5, 2)): print("FAILURE")
    print("SUCCESS")
