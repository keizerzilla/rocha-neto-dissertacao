# momentos.py
# Implementação das sugestões de Momentos segundo o arcabouço de extração
# Artur Rodrigues Rocha Neto (artur.rodrigues26@gmail.com)

import seaborn as sns
import matplotlib.pyplot as plt
from nuvem import *
from itertools import product
from math import factorial as fac

def radial_func(n, m, rhos):
    # Polinômio Radial de Zernike: função f_zernike
    
    m = abs(m)
    poly = 0
    for s in np.arange(((n - m) / 2) + 1):
        num = pow(-1, s) * fac(n - s) * pow(rhos, n - 2*s)
        den = fac(s) * fac(((n + m) / 2) - s) * fac(((n - m) / 2) - s)
        poly = poly + (num / den)
    
    return poly

def angular_func(m, angles):
    # Função Angular de Zernike: função g_zernike
    
    return np.prod(np.cos(m*angles), axis=1) + np.prod(np.sin(m*angles), axis=1)

def zernike_args(order, repetition):
    # Função utilitária
    # Calcula pares de argumentos para momentos de zernike
    
    orders = list(range(order+1))
    repetitions = list(range(repetition+1))
    parans = [t for t in product(orders, repetitions)
              if (t[0] - t[1]) % 2 == 0 and t[0] >= abs(t[1])]
    
    return parans

def zernike_moments(cloud, order=10, repetition=10):
    # Sugestão de Momentos de Zernike
    
    matmom = np.zeros((order+1, repetition+1))
    parans = zernike_args(order, repetition)
    
    for n, m in parans:
        ans = radial_func(n, m, cloud[:, 0]) * angular_func(m, cloud[:, 1:])
        matmom[n, m] = (n + 1 / np.pi) * ans.sum()
    
    return matmom[matmom != 0]

def orthogonal_args(p, q, r):
    # Função utilitária
    # Calcula pares de argumentos para Momentos polinomiais
    
    o1 = list(range(p+1))
    o2 = list(range(q+1))
    o3 = list(range(r+1))
    
    return product(o1, o2, o3)

def orthogonal_moments(poly_func, cloud, p, q, r):
    # Função que extrái Momentos polinomais
    
    args = orthogonal_args(p, q, r)
    ans = []
    
    for a, b, c in args:
        mu = poly_func(a, cloud[:, 1]) * \
             poly_func(b, cloud[:, 2]) * \
             poly_func(c, cloud[:, 3]) * \
             cloud[:, 0]
        ans.append(mu.sum())
    
    return ans

def legendre_poly(n, x):
    # Polinômio recursivo de Legendre: função g_legendre
    
    if n < 0:
        return np.zeros(x.shape)
    elif n == 0:
        return np.ones(x.shape)
    elif n == 1:
        return x
    else:
        num1 = ((2 * n) - 1) * x * legendre_poly(n - 1, x)
        num2 = (n - 1) * legendre_poly(n - 2, x)
        den = n
        return (num1 - num2) / den

def chebyshev_poly(n, x):
    # Polinômio recursivo de Chebyshev: função g_chebyshev
    
    if n < 0:
        return np.zeros(x.shape)
    elif n == 0:
        return np.ones(x.shape)
    elif n == 1:
        return x
    else:
        return 2 * x * chebyshev_poly(n - 1, x) - chebyshev_poly(n - 2, x)

def legendre_moments(cloud, p, q, r):
    # Sugestão de Momentos de Legendre
    
    return orthogonal_moments(legendre_poly, cloud, p, q, r)

def chebyshev_moments(cloud, p, q, r):
    # Sugestão de Momentos de Chebyshev
    
    return orthogonal_moments(chebyshev_poly, cloud, p, q, r)

def plot_zernike():
    # Plota os Polinômios Radiais de Zernike
    
    sns.set_theme()
    
    x = np.linspace(0, 1, 1000)
    args = [(0, 0), (1, 1), (2, 0), (2, 2), (3, 1), (4, 0), (4, 2), (5, 1)]
    for arg in args:
        y = radial_func(arg[0], arg[1], x)
        label = "R_({},{})".format(arg[0], arg[1])
        plt.plot(x, y, label=label)
    
    title = "Polinômio Radial de Zernike"
    plt.title(title)
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$R_{nm}(\rho)$")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
    plt.tight_layout()
    plt.show()

def plot_polynomial(poly_func, poly_name, poly_letter):
    # Plota uma função polinomial
    
    sns.set_theme()
    
    x = np.linspace(-1, 1, 1000)
    args = [0, 1, 2, 3, 4, 5]
    for arg in args:
        y = poly_func(arg, x)
        label = "k = {}".format(arg)
        plt.plot(x, y, label=label)
    
    title = "Polinômio de {}".format(poly_name)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("{}_k(x)".format(poly_letter))
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
    plt.tight_layout()
    plt.show()
