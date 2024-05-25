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
def BuildGroupsQuality():
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
# df: es la matriz df.values [] , no incluye la etiqueta del grupo
def NormalizeViewMinable(df,valMin, dataRange):
    dataset_normalizado = np.empty((df.shape[0], df.shape[1]))
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            dataset_normalizado[i][j]= (df[i][j] - valMin[j])/dataRange[j]

    return dataset_normalizado



'''
Funcion para generar el vector de pesos aleatorio.
Entradas:
w: Vector de pesos w [0,1], igual al numero de caracteristicas normalizadas.
nc: Nunero de ceros que debe contener el vector de pesos [10,20,30,40,50]
'''

def GenerateWeightVector(w, nc):
    posceros = np.random.choice(len(w), nc, replace=False)
    w[posceros] = 0
    s  = np.sum(w)
    wf = np.round(w/s, 4)


    return wf



'''
-Función de calidad, que retorna la metrica de calidad asociada a ese vector w especifico
-Se debe tener en cunata la seleccion de la metrica de calidad asociada, para evaluar
el desempeño del algoritmo (Problema de Clasificacion)
Entradas:
    df_norm: Dataset Normalizado.
    wi: Vector de Pesos.
'''

# Dependiendo del vector de pesos me extrae el acuracy - F1 score
def qualityFunction(df_norm, wi,df):
    y_pred = []
    for i in range(len(df_norm)):
        minDep = sys.float_info.max
        posMinDep = 0
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
    #qs = f1_score(df.values[:,-1], y_pred, average='micro')
    return qs





'''
....Recocido Simulado (Simulated Annealing, SA)

'''


#   PARAMETROS GENERALES
# ==============================================================================
#temp = 100
P= 174
max_iterations = 100                                                       
nc_array = [50,55,60,65]
#nc_array = [50]          
pm= 0.1                            

#  SA
# ==============================================================================

#df = BuildGroupsQuality()
#df_norm = NormalizeViewMinable(df)

df = pd.read_csv("FASE2/Final_dataset_join.csv")
print("Longitud df: ", df.shape)
#2. Lectura de los Encoders
valMin = np.loadtxt('FASE2/Encoder_ValMin.txt')
dataRange = np.loadtxt('FASE2/Encoder_dataRange.txt')
df_matriz = df.values[:,:-1]

df_norm = NormalizeViewMinable(df_matriz, valMin, dataRange)





for nceros in range(len(nc_array)):
    print(f"Numero de ceros {nc_array[nceros]} vector solucion")
    
    p = np.random.rand(P)
    nc = nc_array[nceros]
    bw = 1/((P-nc)/10)

    s= GenerateWeightVector(p,nc)
    qs= qualityFunction(df_norm, s,df)
    best = np.copy(s)
    qbest = qs
    curvaSA = []
    vectorBest = []
    temp=100
    print("Reinicio la Temperatura: ", temp)
    for t in range(temp):
        rv = np.copy(s)
        # Generación del Twick
        for d in range(P):
            aleatorio2 = random.random()
            if aleatorio2 < pm:
                aleatorio2 = random.random()
                if aleatorio2 < nc/P:
                    rv[d] = 0
                else:
                    rv[d]= rv[d]+ np.random.uniform(-bw,bw)
                    if rv[d] < 0:
                        rv[d] = 0

        # Vuelvo a normmalizar el vector de pesos r sumado el twick
        rv = GenerateWeightVector(rv,0)
        qr= qualityFunction(df_norm, rv,df)
        aleatorio = np.random.rand()
        if (qr >= qs) or (aleatorio < e**((qr-qs)/temp)):
            s=rv
            qs = qr
        
        # Disminuyo la termperatura
        temp = temp-t
        # se verifica el remplazo
        if qs > qbest:
            best = np.copy(s)
            qbest=qs
        
        curvaSA.append(qbest)
        vectorBest.append(best)
    
    #Guardo el Best y Qbest despeus de reducir al minimo la temepratura
    print("Curva SA: ", curvaSA)
    dicc = {"vector":vectorBest,
            "curvaSA": curvaSA
            }
    
    df_new = pd.DataFrame(data=dicc)
    df_new.to_csv(f"ResultadosImprovisacion/SA/Training2/acc_nc_{nc}.csv")
    dicc = dict()





