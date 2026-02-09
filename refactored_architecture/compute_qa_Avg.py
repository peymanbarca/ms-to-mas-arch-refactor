import math
import numpy as np

# ------------ consistency ---------
qa_avg_ranked = [0, 0, 0, 0, 5, 0, 0, 0, 1]
qa_avg_random = [0, 0, 0, 1, 8, 0, 3, 1, 2]
qa_avg_reverse = [0, 0, 0, 6, 0, 0, 2, 0, 0]


print(f'qa_avg_ranked_sum = {np.sum(qa_avg_ranked)} , qa_avg_ranked_sum_avg = {np.average(qa_avg_ranked)}')
print(f'qa_avg_random_sum = {np.sum(qa_avg_random)} , qa_avg_random_sum_avg = {np.average(qa_avg_random)}')
print(f'qa_avg_reverse_sum = {np.sum(qa_avg_reverse)} , qa_avg_reverse_sum_avg = {np.average(qa_avg_reverse)}')

# ------------ latency ---------
print('\n\n\n')

qa_avg_ranked = [0.314, 0.512, 0.651, 1.334, 1.131, 1.425, 0.853, 0.967, 1.811]
qa_avg_random = [0.657, 0.809, 1.098, 2.310, 2.025, 1.660, 1.432, 1.384, 1.519]
qa_avg_reverse = [0.473, 0.800, 1.541, 1.210, 1.089, 1.233, 2.444, 1.420, 1.636]


print(f'qa_avg_ranked_sum = {np.sum(qa_avg_ranked)} , qa_avg_ranked_sum_avg = {np.average(qa_avg_ranked)}')
print(f'qa_avg_random_sum = {np.sum(qa_avg_random)} , qa_avg_random_sum_avg = {np.average(qa_avg_random)}')
print(f'qa_avg_reverse_sum = {np.sum(qa_avg_reverse)} , qa_avg_reverse_sum_avg = {np.average(qa_avg_reverse)}')