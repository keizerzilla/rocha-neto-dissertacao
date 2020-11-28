# nuvem.py
# Arquivos de manipulação de nuvens: carregamento, salvamento, conversão
# Artur Rodrigues Rocha Neto (artur.rodrigues26@gmail.com)

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier as KNN

def load_xyz(path):
    # Carrega nuvem de pontos do arquivo 'path' em formato XYZ
    
    return np.array(pd.read_csv(path, sep=" ", header=None))

def save_xyz(cloud, path):
    # Salva nuvem 'cloud' no arquivo 'path'
    
    df = pd.DataFrame(cloud, columns=None)
    df.to_csv(path, sep=" ", header=None, index=None)

def angle_norm(n, x):
    # Ângulos que os pontos de uma nuvem 'x' fazem com um plano de normal 'n'
    
    prod = n * x
    num = np.abs(prod.sum(axis=1))
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(n)
    return np.arcsin(num / den)

def cloud_preproc(cloud):
    # Converte nuvem cartesiana 'cloud' em representação RABG
    
    cloud = cloud - np.mean(cloud, axis=0)
    _, _, v = np.linalg.svd(cloud, full_matrices=False)
    angles = np.apply_along_axis(angle_norm, 1, v, cloud).T
    distances = np.linalg.norm(cloud, axis=1)
    rhos = distances / np.amax(distances)
    rhos = rhos.reshape((-1, 1))
    return np.concatenate((rhos, angles), axis=1)
