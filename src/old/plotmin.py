import matplotlib.pyplot as plt

plt.close('all')
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(MX, MY, color=color)
ax1.scatter(v, splev(v, f), color=color)

# ax1.scatter(x0,splev(x0,f), label='x0')
# ax1.scatter(v.x, v.fun, label='Local Minima')

ax1.set_ylabel('mode trace (f)', color=color)
ax1.set_xlabel('bw')
ax1.tick_params(axis='y', labelcolor=color)


ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(MX, df2(MX, f, e), color=color)
ax2.scatter(v, df2(v, f, e), color=color)
ax2.set_ylabel(r'$(df)^2$', color=color)
ax2.tick_params(axis='y', labelcolor=color)



