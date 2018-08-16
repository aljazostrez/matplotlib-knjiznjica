import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil, log, e
from scipy import stats, integrate
from statsmodels.distributions.empirical_distribution import ECDF


##generiranje distribucije
gauss1 = []
gauss2 = []

for i in range(10000):
    gauss1.append(random.gauss(30,5))
    gauss2.append(random.gauss(30,10))

##mu = np.mean(gauss1)
##sigma = np.var(gauss1)**0.5
##
##
####najbolj primerna širina bina
##def sirina(sigma, N):
##    return 3.49*sigma*N**(-1/3)
##
##h = np.arange(min(gauss1),max(gauss1),sirina(sigma,len(gauss1)))
##
####maximum bin
##def tallest_bin(heights, values):
##    maximum_index = list(heights).index(max(list(heights)))
##    return 'Interval: {}, Vrednost: {}'.format((round(values[maximum_index],2), round(values[maximum_index+1],2)), round(y[maximum_index],4))
##            
##
####histogram in denziteta
##histogram = plt.subplot(211)
####y, x, _ = plt.hist(gauss1, bins=h, density=1)
####plt.grid(True)
####plt.title('Histogram in denziteta')
##
pdf1 =  stats.gaussian_kde(gauss1)
x1 = np.linspace(floor(min(gauss1)),ceil(max(gauss1)), 10000)
##plt.plot(x1, pdf1(x1))
##
##
####histogram.set_xticks(np.arange(floor(min(gauss1))-5,ceil(max(gauss1))+5,1))
##
##
pdf2 = stats.gaussian_kde(gauss2)
x2 = np.linspace(floor(min(gauss2)),ceil(max(gauss2)), 10000)
####plt.plot(x2, pdf2(x2), color='red')
##
##
####renyi entropija
##def disc_renyi_entropy(pk, alfa):
##    if alfa == 1:
##        return -sum([pk[i]*log(pk[i]) for i in range(len(pk))])
##    else:
##        vsota = sum([pk[i]**alfa for i in range(len(pk))])
##        return (1/(1-alfa))*log(vsota, e)
##
##def cont_renyi_entropy(pdf, alfa, minimum=None, maximum=None):
##    def pdf_na_alfa(z):
##        return (pdf(z)**alfa)
##    def pdf_logpdf(z):
##        return pdf(z)*log(pdf(z), e)
##    if alfa == 1:
##        return -integrate.quad(pdf_logpdf, minimum, maximum)[0]
##    elif alfa > 0:
##        return (1/(1-alfa))*log(integrate.quad(pdf_na_alfa, -float('inf'), float('inf'))[0], e)
##    else:
##        raise ValueError('Alfa must a positive positive number')
##
####Distribucija
##ecdf1 = ECDF(gauss1)
##
##distribucija = plt.subplot(212, sharex = histogram)
##plt.plot(x1, ecdf1(x1))
##plt.grid(True)
##plt.title('Distribucija')
##
####ecdf2 = ECDF(gauss2)
####plt.plot(x2, ecdf2(x2), color='red')
##
##plt.subplots_adjust(hspace=0.5)
##
##
####plt.show()
