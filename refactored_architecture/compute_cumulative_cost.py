import math
import numpy as np

delta_c_ranked = [0.328, 0.476, 0.759, 1.087, 1.063, 1.297, 0.936, 1.085, 2.570]
delta_c_random = [0.332, 0.665, 1.204, 0.978, 0.815, 0.964, 2.409, 1.244, 1.136]

print(f'delta_c_ranked_sum = {np.sum(delta_c_ranked)} , delta_c_ranked_sum_avg = {np.average(delta_c_ranked)}')
print(f'delta_c_random_sum = {np.sum(delta_c_random)} , delta_c_random_sum_avg = {np.average(delta_c_random)}')