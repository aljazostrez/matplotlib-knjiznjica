import numpy as np
from scipy import integrate
from math import log
from Gauss import *

## Renyi entropy for continuous sample
def cont_renyi_entropy(pdf, alpha, minimum=None, maximum=None):
    def pdf_na_alpha(z):
        return (pdf(z)**alpha)
    def pdf_logpdf(z):
        return pdf(z)*log(pdf(z), e)
    if alpha == 1:
        return -integrate.quad(pdf_logpdf, minimum, maximum)[0]
    elif alpha > 0:
        return (1/(1-alpha))*log(integrate.quad(pdf_na_alpha, -float('inf'), float('inf'))[0], e)
    else:
        raise ValueError('alpha must a positive positive number')


## Renyi divergence for continuous sample
def cont_renyi_divergence(pdf1, pdf2, alpha, minimum=None, maximum=None):
    def integrand(x):
        return ((pdf1(x))**alpha) * ((pdf2(x))**(1-alpha))
    def KL_integrand(x):
        return (pdf1(x)) * log((pdf1(x))/(pdf2(x)), e)
    if alpha == 1:
        return -integrate.quad(KL_integrand, minimum, maximum)[0]
    else:
        return (1/(1-alpha))*log(integrate.quad(integrand, minimum, maximum)[0], e)


def renyi_gauss(mu_j, sigma_j, mu_i, sigma_i, alpha):
    if alpha != 1:
        var_zvezdica = alpha*(sigma_j ** 2) + (1-alpha)*(sigma_i**2)
        rezultat = log(sigma_j/sigma_i, e) + (1/(2*(alpha-1)))*log((sigma_j**2)/var_zvezdica, e) + (alpha*((mu_i - mu_j)**2))/(2*var_zvezdica)
        return rezultat
    else:
        rezultat = (1/(2*(sigma_j**2)))*((mu_i - mu_j)**2 + sigma_i**2 - sigma_j**2) + log(sigma_j/sigma_i, e)
        return rezultat

#integracijsko območje je vredu - (min, max)

mu0 = np.mean(gauss1)
mu1 = np.mean(gauss2)
sigma0 = np.var(gauss1)
sigma1 = np.var(gauss2)

minimum = min(np.concatenate((np.array(gauss1),np.array(gauss2))))
maximum = max(np.concatenate((np.array(gauss1),np.array(gauss2))))


plot1 = plt.subplot(211)
pdf_1, = plt.plot(x2, pdf1(x2), color='red')
pdf_2, = plt.plot(x2,pdf2(x2))
plt.grid(True)
plt.legend(handles=[pdf_1, pdf_2], labels=['pdf1','pdf2'])


renyi1 = []
renyi2 = []
##renyi3 = []
renyi4 = []

for i in np.arange(0.1,2,0.1):
    renyi1.append(cont_renyi_divergence(pdf1,pdf2,i,minimum,maximum))
    renyi2.append(cont_renyi_divergence(pdf2,pdf1,i,minimum,maximum))
##    renyi4.append(renyi_gauss(mu0,sigma0,mu1,sigma1,i))
    renyi4.append(renyi_gauss(mu1,sigma1,mu0,sigma0,i))


plot2 = plt.subplot(212)
pdf1_pdf2, = plt.plot(np.arange(0.1,2,0.1), renyi1, color='green')
pdf2_pdf1, = plt.plot(np.arange(0.1,2,0.1), renyi2, color='black')
by_formula, = plt.plot(np.arange(0.1,2,0.1), renyi4, color='purple')
plt.legend(handles=[pdf1_pdf2, pdf2_pdf1, by_formula], labels=['D(pdf1, pdf2)', 'D(pdf2, pdf1)', 'D(mu0,sigma0,mu1,sigma1)'])
plt.grid(True)




plt.show()
