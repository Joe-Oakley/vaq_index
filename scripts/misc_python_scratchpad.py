import numpy as np
import math

# # 24/10/2023 Test Min heap
from heapq import heapify, heappush, heappop, heappushpop, nlargest, nsmallest

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

# test_np = np.array([1,2,3,4], dtype=np.float32)

# # print(test_np)
# # test_np_appended = np.append(test_np, 5)
# # print(test_np_appended)

# val = 2.5
# sort_idx = np.searchsorted(test_np, val)
# print(sort_idx)
# test_np = np.insert(test_np, sort_idx, val)
# print(test_np)



# heap = []
# heapify(heap)

# vals = [-243, -89, -4002, -2, -500, -8, -65, -1, -15, -18]
# print(vals)
# for val in vals:
#     heappush(heap, val)

# print()
# print('Smallest 4')
# print(nsmallest(4, heap ))
# print()
# print('Larget 4')
# print(nlargest(4, heap ))
# print()
# print()

# # heappushpop(heap,)

# Make a heap of self.UP (heapify)
# UP = np.array([3,4,5])
# UP_heap = np.multiply(UP, -1).tolist()
# heapify(UP_heap)
# vecs = [1,2,3,4,5,6,7,8,9,10]
# for i in range(len(vecs)):

#     if vecs[i]*-1 >= nsmallest(1, UP_heap)[0]:
#         heappush(UP_heap, vecs[i]*-1)
#         remove = heappop(UP_heap)


# UP = np.multiply(np.array(nlargest(3, UP_heap)), -1)
# print(UP)


arr = np.array([2,3,4,4])
arr_max = arr.max()
arr_max_idx = np.argmax(arr) # In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.

print("arr_max: " , arr_max)
print("type(arr_max) ",  type(arr_max))

print("arr_max_idx: " , arr_max_idx)
print("type(arr_max_idx) ",  type(arr_max_idx))