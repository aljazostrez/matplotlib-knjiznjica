import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import floor, ceil

ffa = np.fromfile('lihi_data_prbs_100hz_50000_180610_132601.bin',dtype=np.float64)

a = np.asarray(ffa)

tok = a[0::4] * 50 / 0.625
napetost = a[2::4]

zgornji_tok = []
spodnja_napetost = []
spodnji_tok = []
zgornja_napetost = []

for i in range(165000):
    if tok[i] > 32:
        zgornji_tok.append(tok[i])
        spodnja_napetost.append(napetost[i])
    elif tok[i] < 32:
        spodnji_tok.append(tok[i])
        zgornja_napetost.append(napetost[i])


tok_mu = np.mean(tok)
napetost_mu = np.mean(napetost)

fs = 50000.0
t = [i/fs for i in list(range(0, len(tok)))]

ax1 = plt.subplot(321)
plt.plot(t, tok)
plt.title('Tok v odvisnosti od časa')

ax2 = plt.subplot(322, sharex=ax1)
plt.plot(t, napetost)
plt.title('Napetost v odvisnosti od časa in toka')

h1 = np.arange(min(spodnja_napetost), max(spodnja_napetost), 0.0002)
h2 = np.arange(min(zgornja_napetost), max(zgornja_napetost), 0.0002)


histogram1 = plt.subplot(325)
n, bins, patches = plt.hist(np.array(spodnja_napetost), density=True, bins=h1)
plt.title('Spodnje napetosti, ko je tok visok (>32A)')
plt.grid(True)


histogram2 = plt.subplot(326)
plt.hist(np.array(zgornja_napetost), density=True, bins=h2)
plt.title('Zgornje napetosti, ko je tok nizek (<32A)')
plt.grid(True)



plt.show()
