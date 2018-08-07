import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil, log
from scipy import stats, integrate
from statsmodels.distributions.empirical_distribution import ECDF

gauss = []

for i in range(10000):
    gauss.append(random.gauss(30,5))

mu = np.mean(gauss)
sigma = np.var(gauss)**0.5

def sirina(sigma, N):
    return 3.49*sigma*N**(-1/3)

h = np.arange(min(gauss),max(gauss),sirina(sigma,len(gauss)))


histogram = plt.subplot(211)
n, bins, patches = plt.hist(gauss, bins=h, density=1)
plt.grid(True)

pdf =  stats.gaussian_kde(gauss)
x = np.linspace(floor(min(gauss)),ceil(max(gauss)), 10000)

plt.plot(x, pdf(x), '--')


histogram.set_xticks(np.arange(floor(min(gauss))-5,ceil(max(gauss))+5,1))

def pdf_logpdf(x):
    return pdf(x)*log(pdf(x))

def entropija(density):
    def pdf_logpdf(x):
        return density(x)*log(density(x))
    return -(integrate.quad(pdf_logpdf, min(gauss)-1, max(gauss)+1)[0])

ecdf = ECDF(gauss)

distribucija = plt.subplot(212, sharex = histogram)
d = np.arange(min(gauss), max(gauss), 0.01)
plt.plot(d, ecdf(d))
plt.grid(True)


plt.show()
