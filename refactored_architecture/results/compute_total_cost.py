import math

c_obs_per_kb = math.pow(10, -4)

lat_p95_baseline = 1.642

c_obs_per_req_baseline = 4.4 * c_obs_per_kb

# inputs
lat_p95 = 3.278
n_llm_token = 107.6 * 1000
c_obs_per_req = 10.4 * c_obs_per_kb


c_infra = math.pow(10, -4)
c_llm_token = math.pow(10, -7)
R = 100
delta_q_lat_p95 = lat_p95 - lat_p95_baseline
delta_c = delta_q_lat_p95 * c_infra + (c_obs_per_req - c_obs_per_req_baseline + c_llm_token * n_llm_token) * R

print(f'delta_q_lat_p95 = {delta_q_lat_p95} , delta_c = {delta_c}')
