# Librerias y Dependencias
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from sklearn.metrics import f1_score, accuracy_score
import sys
import math
from math import e




print("Numero Aleatorio Flotante:", sys.float_info.max)
print("Euler: ", e)


# Funcion para construir el dataset - Asociado a los respectivos grupos
def build_groups_quality():
    GC1 = pd.read_excel("../Archivos Generados/PipelineResults/GruposCalidad-Fase2/Grupo1.xlsx",index_col=0)
    GC2 = pd.read_excel("../Archivos Generados/PipelineResults/GruposCalidad-Fase2/Grupo2.xlsx",index_col=0)
    GC3 = pd.read_excel("../Archivos Generados/PipelineResults/GruposCalidad-Fase2/Grupo3.xlsx",index_col=0)


    GC1["Grupo"] = 1
    GC2["Grupo"] = 2
    GC3["Grupo"] = 3

    # Se Unen los diferentes Grupos en un unico dataset
    df = pd.concat([GC1, GC2,GC3], ignore_index=True)

    # Elimino las variables no relevantes
    df = df.drop(["ID_LOTE", "RDT_AJUSTADO"], axis=1)

    return df



# Funcion para Normalizar la Vista minable a exepción de la etiqueta(Variable Objetivo)
def normalize_view_minable(df):
    norm = MinMaxScaler()
    df_norm = norm.fit_transform(df.values[:,:-1])
    return df_norm






'''
w: Longitud del vector de pesos, igual al numero de caracteristicas normalizadas.
nc: Nunero de ceros que debe contener el vector de pesos [10,20,30,40,50]
'''
def generar_vector_pesos(w, nc):
    # Se seleccionan diferentes posiciones(50) para hacerce cero.
    posceros = np.random.choice(len(w), nc, replace=False)
    w[posceros] = 0
    #print(w)
    s  = np.sum(w)
    # Normalizo los pesos - Solo obtengo los 3 decimales de c/d peso
    wf = np.round(w/s, 3)
    sc = 1- np.sum(wf)
    pos = np.random.randint(len(wf))
    if sc != 0:
        wf[pos]= wf[pos]+sc

    return wf





'''
df_norm: Dataset Normalizado.
wi: Vector de Pesos.
'''

# Dependiendo del vector de pesos me extrae el acuracy - F1 score
def quality_function(df_norm, wi):
    # Vector Distancias Euclideanas Ponderadas
    y_pred = []
    minDep = sys.float_info.max
    posMinDep = 0
    for i in range(len(df_norm)):
        vrf = df_norm[i] 
        for j in range(len(df_norm)):
            if i != j:
                ri = wi* np.power((df_norm[j] - vrf), 2)
                dE = np.sum(ri)
                if dE < minDep:
                    posMinDep=j
                    minDep = dE
              
        y_pred.append(df.values[posMinDep][-1])
    qs = accuracy_score(df.values[:,-1], y_pred)
    return qs



'''
Recocido Simulado (Simulated Annealing, SA)

'''

# Se contruyen los grupos de calidad
df = build_groups_quality()
df_norm = normalize_view_minable(df)

temperature = 10
nc = 50
P= 174
pm= 0.1
bw = 1/((P-nc)/10)

# Generar pesos aleatorios 
p = np.random.rand(P)
s= generar_vector_pesos(p,nc)
qs= quality_function(df_norm, s)
best = np.copy(s)
qbest = qs
print(f"Calidad s: {qs}")
print(f"Calidad qbest: {qbest}")

for t in range(temperature):
    #------- r = tweks(s) ------------
    temp = temperature-t
    rv = np.copy(s)
    for d in range(P):
        aleatorio2 = random.random()
        if aleatorio2 < pm:
            aleatorio2 = random.random()
            if aleatorio2 < nc/P:
                rv[0] = 0
            else:
                rv[d]=rv[d]+ np.random.uniform(-bw,bw)
                if rv[d] < 0:
                    rv[d] = 0



    r= generar_vector_pesos(rv,0)
    qr= quality_function(df_norm, r)
    print("Calidad de r: ", qr)
    aleatorio = np.random.rand()
    print((aleatorio < e**((qr-qs)/temp)))
    print(" Termino 2: ",  e**((qr-qs)/temp))
    if (qr >= qs) or (aleatorio < e**((qr-qs)/temp)):
        s=r
        qs = qr
    
    # Verficamos el remplazo
    if qs < qbest:
        best = np.copy(s)
        qbest=qs
    

    print("Best Final: ", qbest)


