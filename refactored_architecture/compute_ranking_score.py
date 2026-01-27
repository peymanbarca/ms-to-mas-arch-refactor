import math

w_comp = 1
w_dep = 1

c_cog = [22, 14, 6, 30, 20, 5, 18, 14, 5]
c_cyc = [10, 6, 2, 14, 6, 3, 8, 5, 4]
fan_out = [1, 0, 0, 7, 1, 1, 1, 1, 1]
bc = [0.2, 0, 0, 1, 0.2, 0, 0, 0, 0]
services = ['Product_search', 'Pricing', 'Shopping_Cart', 'Order', 'Inventory', 'Payment', 'Procurement',
            'Shipment', 'Notification']

c_cog_norm = [c / max(c_cog) for c in c_cog]
c_cyc_norm = [c / max(c_cyc) for c in c_cyc]
fan_out_norm = [f / max(fan_out) for f in fan_out]
bc_norm = [b / max(bc) for b in bc]

rs = []
for i in range(len(c_cog)):
    rs.append( w_comp * (c_cog_norm[i] / (1 + c_cyc_norm[i])) - w_dep * (fan_out_norm[i] + bc_norm[i]) )

print('Final Ranking Scores: ')
print(rs)

rs_sorted = sorted(rs)
for rs_ in rs_sorted:
    i = rs.index(rs_)
    print(services[i])

