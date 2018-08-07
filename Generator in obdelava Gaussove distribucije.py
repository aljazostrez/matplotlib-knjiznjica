import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
from scipy import stats

gauss = []

for i in range(10000):
    gauss.append(random.gauss(30,5))

mu = np.mean(gauss)
sigma = np.var(gauss)**0.5

def sirina(sigma, N):
    return 3.49*sigma*N**(-1/3)

h = np.arange(min(gauss),max(gauss),sirina(sigma,len(gauss)))


histogram = plt.subplot(111)
n, bins, patches = plt.hist(gauss, bins=h, density=1)
plt.grid(True)

pdf =  stats.gaussian_kde(gauss)
x = np.linspace(floor(min(gauss)),ceil(max(gauss)), 10000)

plt.plot(x, pdf(x), '--')


histogram.set_xticks(np.arange(floor(min(gauss)),ceil(max(gauss)),1))

plt.show()
