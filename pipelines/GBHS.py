# Librerias y Dependencias
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from sklearn.metrics import f1_score, accuracy_score
import sys
import math




print("Numero Aleatorio Flotante:", sys.float_info.max)

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



# Funcion para Normlizar la vista minable a exepción de la etiqueta objetivo
def Normalize_view_minable():
    norm = MinMaxScaler()
    df_norm = norm.fit_transform(df.values[:,:-1])
    return df_norm






'''
w: Longitud del vector de pesos, igual al numero de caracteristicas normalizadas.
'''
def generarVectorPesos(w):
    # Se seleccionan diferentes posiciones(50) para hacerce cero.
    posceros = np.random.choice(len(w), 50, replace=False)
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

# Dependeindo del vector de pesos me extrae el acuracy - F1 score
def qualityFunction(df_norm, wi):
    # Vector Distancias Euclideanas Ponderadas
    y_pred = []
    minDep = sys.float_info.max
    posMinDep = 0
    for i in range(len(df_norm)):
        vrf = df_norm[i] 
        for j in range(len(df_norm)):
            if i != j:
                ri = wi* np.power((df_norm[j] - vrf), 2)
                dE = np.sqrt(np.sum(ri))
                if dE < minDep:
                    posMinDep=j
                    minDep = dE
              
        y_pred.append(df.values[posMinDep][-1])
    qs = accuracy_score(df.values[:,-1], y_pred)
    return qs




'''
Función para generar la memoria Armonica
MAC: Memoria Armonica
wi: Vector de Pesos.
'''
def GenerateArmonyMemory(df_norm, MAC):
    Lw = []
    for i in range (MAC):
        # 1. Generar pesos aleatorios | dim = cantidad de caracteristicas
        vp = np.random.rand(df_norm.shape[1])
        wi = generarVectorPesos(vp)
        # 2. Se obtiene el fitnes asociados a esos pesos.
        Qs = qualityFunction(df_norm, wi)
        wiq = np.append(wi, Qs)
        Lw.append(wiq)

    
    # Se obtiene la memoria armonica ordenada de  Mayor- Menor
    MA = Lw.sort(key=lambda x: x[-1], reverse=True)
    return MA


df = build_groups_quality()


############################## PROCESO DE IMPROVISACIÓN ######################################
'''
lmp: Numero de improvisaciones (iteraciones) que realiza GBHS.
P: Numero de Atributos
HMRC: Tasa de Consideración de la memoria Armonica
ParMin: Tasa de ajuste de tono mínima.
ParMax: Tasa de ajuste de tono máxima.
hmns: Tamaño de la memoria armonica.
'''
lmp = 10
P= len(MA[0])
HMRC = 0.85
PAR = 0.35              #[0.1, 0.12, 0.13 , 0.40]
hmns = 5                #Tamaño de la memoria armonica