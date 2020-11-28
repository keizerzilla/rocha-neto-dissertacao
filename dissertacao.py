# dissertacao.py
# Arquivos gerais da dissertação: códigos geradores de resultado e figuras
# Artur Rodrigues Rocha Neto (artur.rodrigues26@gmail.com)

import os
import open3d as o3d
import seaborn as sn
from nuvem import *
from momentos import *
from bosphorus import *
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from itertools import product, permutations
from scipy.special import sph_harm as spharm

def numpy2latex(A, filename="table.txt"):
    # Função utilitária
    # Gera uma tabela formato LaTeX a partir de um array numpy
    
    f = open(filename, "w")
    cols = A.shape[1]
    
    tabformat = "%.2f"
    tabalign = "c"*cols
    
    header = "".join(["H{} & ".format(h) for h in range(cols)])
    header = header[:-3] + "\\\\\n"
    
    f.write("\n\\begin{table}[h]\n")
    f.write("\\captionsetup{width=16cm} % MUDAR AQUI!!!\n")
    f.write("\\caption{\\label{tab:exemplo-1} LEGENDA AQUI!!!}\n")
    f.write("\\IBGEtab{}{\n")
    f.write("\\begin{tabular}{%s}\n" % tabalign)
    f.write("\\toprule\n")
    f.write(header)
    f.write("\\midrule \\midrule\n")
    
    np.savetxt(f, A, fmt=tabformat, delimiter="\t&\t", newline="\t \\\\ \n")
    
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("}{\n")
    f.write("\\Fonte{o autor.}\n")
    f.write("}\n")
    f.write("\\end{table}\n")
    
    f.flush()
    f.close()

def plots_spharms():
    # Função utilitária
    # Plota alguns esféricos harmônicos (Figura 3)
    
    for m, n in args:
        name = "spharm_{}_{}.png".format(n, m)
        title = fr"$Y_{n},{m}$"

        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        phi, theta = np.meshgrid(phi, theta)
        xyz = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        Y = spharm(abs(m), n, theta, phi)
        
        if m < 0:
            Y = np.sqrt(2) * (-1)**m * Y.imag
        elif m > 0:
            Y = np.sqrt(2) * (-1)**m * Y.real
        
        Yx, Yy, Yz = np.abs(Y) * xyz
        
        cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("Spectral"))
        cmap.set_clim(-0.5, 0.5)
        
        fig = plt.figure(figsize=plt.figaspect(1.))
        ax = fig.gca(projection="3d")
        ax.plot_surface(Yx, Yy, Yz, rstride=2, cstride=2, facecolors=cmap.to_rgba(Y.real))
        
        ax_lim = 0.5
        ax.plot([-ax_lim, ax_lim], [0,0], [0,0], c="0.5", lw=1, zorder=10)
        ax.plot([0,0], [-ax_lim, ax_lim], [0,0], c="0.5", lw=1, zorder=10)
        ax.plot([0,0], [0,0], [-ax_lim, ax_lim], c="0.5", lw=1, zorder=10)
        ax.set_xlim(-ax_lim, ax_lim)
        ax.set_ylim(-ax_lim, ax_lim)
        ax.set_zlim(-ax_lim, ax_lim)
        ax.set_title(r"$Y_{{{},{}}}$".format(n, m))
        
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig("figures/spharm/" + name)
        plt.close()

def exemplos_nuvens():
    # Plotando algumas nuvens legais
    
    pasta = "exemplos-nuvens/"
    nuvens = {
        "coelho" : os.path.join(pasta, "coelho.xyz"),
        "dragon" : os.path.join(pasta, "dragon.ply"),
        "armadillo" : os.path.join(pasta, "armadillo.ply"),
        "buddha2" : os.path.join(pasta, "buddha2.ply"),
    }
    
    for nuvem, cfile in nuvens.items():
        pcd = o3d.io.read_point_cloud(cfile)
        o3d.visualization.draw_geometries([pcd])

def cortes_nasais_plot():
    # Plotando nuvens cortadas ao redor do nariz
    # Requer acesso à base de dados!
    
    pasta = "cortes-nasais/"
    nuvens = {
        "100" : os.path.join(pasta, "100.xyz"),
        "90" : os.path.join(pasta, "90.xyz"),
        "80" : os.path.join(pasta, "80.xyz"),
        "70" : os.path.join(pasta, "70.xyz"),
        "60" : os.path.join(pasta, "60.xyz"),
    }
    
    #geometries = []
    for nuvem, cfile in nuvens.items():
        pcd = o3d.io.read_point_cloud(cfile)
        pcd.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([pcd])
    

def bunny_visualization():
    # Cria imagem com os coelhos transformados
    
    pasta = "coelhos/transformacoes/"
    bunnies = {
        "R000" : (os.path.join(pasta, "bunny.xyz"), [1, 0, 0]),
        "S555" : (os.path.join(pasta, "bunny_all.xyz"), [0, 1, 0]),
        "T100" : (os.path.join(pasta, "bunny_rotX-45.xyz"), [0, 0, 1]),
        "RX60" : (os.path.join(pasta, "bunny_rotY-60.xyz"), [1, 1, 0]),
        "RY60" : (os.path.join(pasta, "bunny_rotZ-90.xyz"), [1, 0, 1]),
        "RZ60" : (os.path.join(pasta, "bunny_scale.xyz"), [0, 1, 1]),
        "SUPR" : (os.path.join(pasta, "bunny_translate.xyz"), [0, 0, 0]),
    }
    
    geometries = []
    for bunny, cfile in bunnies.items():
        pcd = o3d.io.read_point_cloud(cfile[0])
        pcd.paint_uniform_color(cfile[1])
        geometries.append(pcd)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0.1, 0.5, 0])
    geometries.append(mesh_frame)
    
    o3d.visualization.draw_geometries(geometries)

def draw_cloud_svd(cloud_path):
    # Desenha uma nuvem e o eixo local gerado por sua DVS
    
    cloud = load_xyz(cloud_path)
    cloud = cloud - np.mean(cloud, axis=0)
    U, S, V = np.linalg.svd(cloud, full_matrices=False)
    V = -1/S[0] * V
    
    sn.set_theme()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], marker=".", color=(0, 0, 0), s=0.05)
    
    for v, l, c in zip(V,  [r"$v_0$", r"$v_1$", r"$v_2$"],  [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], label=l, color=c)
    
    plt.axis("off")
    plt.title("Eixo local gerado pela DVS")
    plt.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig("cloud_svd.png")
    plt.close()
    
    for v, l, c, e, a in zip(V,  [r"$v_0$", r"$v_1$", r"$v_2$"],  [(1, 0, 0), (0, 1, 0), (0, 0, 1)], ["v0", "v1", "v2"], [r"$\alpha$", r"$\beta$", r"$\gamma$"]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
        ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], marker=".", color=(0, 0, 0), s=0.05)
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], label=l, color=c)
        
        point = np.array([0, 0, 0])
        normal = v
        
        d = -point.dot(normal)
        xx, yy = np.meshgrid(np.linspace(-0.05, 0.05, 3), np.linspace(-0.05, 0.05, 3))
        z = (-normal[0]*xx - normal[1]*yy - d) * 1. /normal[2]
        ax.plot_surface(xx, yy, z, alpha=0.2, color=c)
        
        plt.axis("off")
        plt.title("Ângulo singular: " + a)
        plt.legend(loc="lower center")
        plt.tight_layout()
        plt.savefig("cloud_svd_{}.png".format(e))
        plt.close()

def bunny_invariance(func, args, name="moment"):
    # Avalia invariância de momentos usando diversos coelhos transformados
    
    pasta = "coelhos/transformacoes/"
    bunnies = {
        "R000" : os.path.join(pasta, "bunny.xyz"),
        "S555" : os.path.join(pasta, "bunny_all.xyz"),
        "T100" : os.path.join(pasta, "bunny_rotX-45.xyz"),
        "RX60" : os.path.join(pasta, "bunny_rotY-60.xyz"),
        "RY60" : os.path.join(pasta, "bunny_rotZ-90.xyz"),
        "RZ60" : os.path.join(pasta, "bunny_scale.xyz"),
        "SUPR" : os.path.join(pasta, "bunny_translate.xyz"),
    }
    
    all_moments = []
    
    for bunny, cfile in bunnies.items():
        x = load_xyz(cfile)
        x = cloud_preproc(x)
        moments = func(x, *args)
        all_moments.append(moments)
    
    all_moments = np.array(all_moments).T
    variances = np.var(all_moments, axis=1).reshape(-1, 1)
    
    all_moments = np.append(all_moments, variances, axis=1)
    numpy2latex(all_moments, "tabelas/invariancias/" + name + ".txt")
    
    print(name)
    print(variances)
    print()
