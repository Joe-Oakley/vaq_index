import numpy as np
import math

# cells_i = 32
# for i in range(1, cells_i - 1):
#     print(i)

# j = 1
# num_vectors = 12103
# cells_for_dim = 32
# print(math.ceil(j*num_vectors / cells_for_dim))

# test_np = np.array([[1,2], [3,4]])
# print(test_np)
# print(test_np- 1)

# test_np = np.ones(10)
# test_np[9] = 5
# x = np.logical_not(test_np == 5).astype(np.int32)
# print(x)

test_np = np.array([1,2,3,4])
print(test_np)
test_np_appended = np.append(test_np, 5)
print(test_np_appended)