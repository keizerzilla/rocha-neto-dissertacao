# bosphorus.py
# Manipulação da base de dados Bosphorus
# Artur Rodrigues Rocha Neto (artur.rodrigues26@gmail.com)

import os
import parse
import seaborn as sns
from nuvem import *
from math import ceil
from momentos import *
from itertools import accumulate

def bosphorus_extraction(folder, func, args, test="test"):
    # Realiza extração de Momentos em conjunto de nuvens da base Bosphorus
    # Necessário acesso à base de dados!
    
    os.makedirs("results/bosphorus/", exist_ok=True)
    resfile = "results/bosphorus/{}.csv".format(test)
    filefmt = "bs{:d}_{:w}_{:w}_{:d}.xyz"
    ans = []
    
    for cfile in os.listdir(folder):
        if not cfile.endswith(".xyz"):
            continue
        
        match = parse.parse(filefmt, cfile)
        subject = str(match[0])
        typexpr = str(match[1])
        express = str(match[2])
        samplen = str(match[3])
        x = load_xyz(os.path.join(folder, cfile))
        
        x = cloud_preproc(x)
        moments = func(x, *args)
        moments = list(moments)
        
        ans.append(moments + [samplen, typexpr, express, subject])
        print(cfile, "ok!")
    
    header = ["m{}".format(p) for p in range(len(ans[0]) - 4)]
    header = header + ["samplen", "typexpr", "express", "subject"]
    df = pd.DataFrame(ans, columns=header)
    df.to_csv(resfile, index=False)

def run_classification(X_train, y_train, X_test, y_test):
    # Executa classificação segundo framework da biblioteca scikit-learn
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    pt = PowerTransformer(method="yeo-johnson")
    pt.fit(X_train)
    X_train = pt.transform(X_train)
    X_test = pt.transform(X_test)
    
    result = dict()
    
    ping = time.time()
    classifier = KNN(n_neighbors=1, p=2)
    classifier.fit(X_train, y_train)    
    result["accu"] = round(classifier.score(X_test, y_test)*100, 4)
    result["true"] = y_test
    result["pred"] = classifier.predict(X_test)
    result["time"] = round(time.time() - ping, 4)
    
    return result

def bosphorus_classification(dataset):
    # Abre arquivo de Momentos pré-extraídos e executa classificação
    
    df = pd.read_csv(dataset)
    
    condtrain = (df["typexpr"] == "N") & (df["samplen"] == 0)
    trainset = df.loc[condtrain].drop(["samplen", "typexpr", "express"], axis=1)
    X_train = np.array(trainset.drop(["subject"], axis=1))
    y_train = np.ravel(trainset[["subject"]])
    
    cond_test = (df["typexpr"] == "N") & (df["samplen"] != 0)
    testset = df.loc[cond_test].drop(["samplen", "typexpr", "express"], axis=1)
    X_test = np.array(testset.drop(["subject"], axis=1))
    y_test = np.ravel(testset[["subject"]])
    
    return run_classification(X_train, y_train, X_test, y_test)

def bosphorus_classification_subset(dataset, features):
    # Abre arquivo de Momentos pré-extraídos e executa classificação apenas num
    # subconjunto de valores do vetor de atributos (útil para análise)
    
    df = pd.read_csv(dataset)
    
    condtrain = (df["typexpr"] == "N") & (df["samplen"] == 0)
    trainset = df.loc[condtrain].drop(["samplen", "typexpr", "express"], axis=1)
    X_train = np.array(trainset[features])
    y_train = np.ravel(trainset[["subject"]])
    
    cond_test = (df["typexpr"] == "N") & (df["samplen"] != 0)
    testset = df.loc[cond_test].drop(["samplen", "typexpr", "express"], axis=1)
    X_test = np.array(testset[features])
    y_test = np.ravel(testset[["subject"]])
    
    return run_classification(X_train, y_train, X_test, y_test)

def bosphorus_pipeline(func, args, cut=80, test="temp"):
    # Pipeline de teste na Bosphorus como mostrado na Figura 10 da dissertação
    # Requer acesso à base de dados!
    
    folder = "D:/mestrado/bosphorus/bs-out"
    noses = "D:/mestrado/bosphorus/nose.csv"
    resfile = "results/bosphorus/pipeline/{}.csv".format(test)
    filefmt = "bs{:d}_{:w}_{:w}_{:d}.xyz"
    
    df = pd.read_csv(noses)
    ans = []
    
    print("Calculando", test, "...")
    
    for index, row in df.iterrows():
        cloud = folder + "/" + row["cloud"]
        coord = row["groundtruth"].replace("[", "").replace("]", "")
        coord = coord.replace(" ", "")
        x, y, z = coord.split(",")
        nose = np.array([float(x), float(y), float(z)])
        face = load_xyz(cloud)
        x = face[np.linalg.norm(face - nose, axis=1) <= cut]
        
        x = cloud_preproc(x)
        moments = func(x, *args)
        moments = list(moments)
        
        match = parse.parse(filefmt, row["cloud"])
        subject = str(match[0])
        typexpr = str(match[1])
        express = str(match[2])
        samplen = str(match[3])
        
        ans.append(moments + [samplen, typexpr, express, subject])
    
    header = ["m{}".format(p) for p in range(len(ans[0]) - 4)]
    header = header + ["samplen", "typexpr", "express", "subject"]
    df = pd.DataFrame(ans, columns=header)
    df.to_csv(resfile, index=False)
    
    return bosphorus_classification(resfile)

def bosphorus_spread(sequence, num):
    # Amostragens igualmente espaçadas de um vetor de Momentos
    
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(ceil(i * length / num))]

def bosphorus_plots(folder):
    # Gera gráficos com o desempenho dos Momentos usados no teste de
    # classificação (Figuras 11, 12 e 13)
    
    dpi = 160
    
    moments = {
        "Legendre" : (orthogonal_args, [4, 4, 4]),
        "Chebyshev" : (orthogonal_args, [4, 4, 4]),
        "Zernike" : (zernike_args, [20, 20]),
    }
    
    for moment, args in moments.items():
        sns.set_theme()
        fig = plt.figure(figsize=(1024/dpi, 768/dpi), dpi=dpi)
        ax = plt.gca()
        num = 10
        
        parans = list(args[0](*args[1]))
        header = ["m{}".format(x) for x in range(len(parans))]
        
        pairs = list(accumulate([[el] for el in header]))
        last = pairs[-1]
        pairs = list(bosphorus_spread(pairs, num))
        pairs = pairs + [last]
        del pairs[0]
        
        pairs2 = list(accumulate([[el] for el in parans]))
        last2 = pairs2[-1]
        pairs2 = list(bosphorus_spread(pairs2, num))
        pairs2 = pairs2 + [last2]
        del pairs2[0]
        
        xticks = [p[-1] for p in pairs2]
        x2 = []
        for p in xticks:
            t = []
            for e in p:
                t.append(str(e))
            x2.append(t)
        xticks = [r"$\Lambda_{("+",".join(p)+")}^{(%s)}$" % moment for p in x2]
        
        for cut in range(100, 50, -10):
            scenario = "-".join([str(a) for a in args[1]])
            test = folder + "{}-{}_{}".format(moment, scenario, cut) + ".csv"
            accuracies = []
            
            for index, pair in enumerate(pairs):
                ans = bosphorus_classification_subset(test, pair)
                accuracies.append(ans["accu"])
                print(moment, cut, len(pair), "features ok! ->", ans["accu"])
                
                accu = round(ans["accu"], 2)
                px = index * (len(pairs) + 1)
                py = accu + 1
                plt.text(px, py, str(accu), fontsize=8)
            
            label = str(cut)
            x = np.linspace(0, 100, len(accuracies))
            plt.plot(x, accuracies, label=label, marker="o")
            plt.xticks(x, xticks, rotation=45)
        
        msg = "Momentos de {}: acurácia versus vetor de atributo e corte"
        title = msg.format(moment)
        figpath = "results/bosphorus/pipeline/figures/"
        figfile = figpath + "acc-vetor-corte-{}.png".format(moment)
        
        plt.title(title)
        plt.legend(title="Corte (mm)", loc="lower center", ncol=5)
        ax.set_ylim(40, 100)
        plt.xlabel("Vetores de atributo")
        plt.ylabel("Acurácia (%)")
        plt.tight_layout()
        plt.savefig(figfile, dpi=dpi)
