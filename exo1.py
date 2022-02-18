import numpy as np
from util import *

my_matrix = np.array([1, 2, 3])
create_first_matrix(my_matrix)

b = my_new_array()
print(b.shape)

a1 = np.random.rand(2, 2)
a2 = np.random.rand(4, 4)
a3 = np.random.rand(5, 2)
check_random_matrix(a1, a2, a3)

a4 = np.dot(a3, a1)
check_mul(a4)
