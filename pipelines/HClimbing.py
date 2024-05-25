# Librerias y Dependencias
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import distance
from sklearn.metrics import f1_score, accuracy_score
import sys
import math
from math import e





# Funcion para Normalizar la Vista minable a exepción de la etiqueta(Variable Objetivo)
# df: es la matriz df.values [] , no incluye la etiqueta del grupo
def NormalizeViewMinable(df,valMin, dataRange):
    dataset_normalizado = np.empty((df.shape[0], df.shape[1]))
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            dataset_normalizado[i][j]= (df[i][j] - valMin[j])/dataRange[j]

    return dataset_normalizado


#df: no es un dataframe, es la  matriz de valores asociados al DF df.values[.,:-1]
def NomralizacionZscore(df):
    scaler = StandardScaler()
    ajus = scaler.fit(df)
    #print(ajus.mean_)
    #print(ajus.var_)
    df_normalizado = ajus.transform(df)
    return df_normalizado


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

def qualityFunctionOriginal(df_norm, wi,df):
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

        #print("Min Dep: ", minDep)   
        y_pred.append(df.values[posMinDep][-1])
    qs = accuracy_score(df.values[:,-1], y_pred)
    #qs = f1_score(df.values[:,-1], y_pred, average='micro')
    return qs




def qualityFunctionVp(df_norm, wi,df,k):
    #print("longitud df_norm: ",len(df_norm))
    #print("Longitud wi: ", len(wi))
    y_pred = []
    for i in range(len(df_norm)):
        vrf = df_norm[i]
        #print(f"Vector {i} :", vrf) 
        ListaPesosPonderados= [[sys.float_info.max,0] for a in range(k)]
        for j in range(len(df_norm)):
            if i != j:
                #print(ListaPesosPonderados)
                ri = wi* np.power((df_norm[j] - vrf), 2)
                dE = np.sum(ri)
                #print("Distancia Euclideana: ", dE)
                if dE < ListaPesosPonderados[k-1][0]:
                    ListaPesosPonderados[k-1][0]=dE
                    ListaPesosPonderados[k-1][1]=j
                    ListaPesosPonderados.sort(key=lambda x: x[0], reverse=False)
       #print(f"vector {i} es muy similar a los vectores en la posición {ListaPesosPonderados}")

        dEG0=0
        dEG1=0
        dEG2=0
        for i in range(len(ListaPesosPonderados)):
            index=ListaPesosPonderados[i][1]
            g = int(df.values[index][-1])
            if g==0:
                dEG0= np.power((1/ListaPesosPonderados[i][0]),2) + dEG0
            if g==1:
                dEG1= np.power((1/ListaPesosPonderados[i][0]),2)  + dEG1
            if g==2:
                dEG2= np.power((1/ListaPesosPonderados[i][0]),2)  + dEG2

            
        dis = [dEG0, dEG1,dEG2]
        #print("Distancias Inversas Aociadas: ", dis)
        grupoSelected = int(dis.index(max(dis)))
        #print("Grupo seleccionado: ", grupoSelected)
        y_pred.append(grupoSelected)
    qs = accuracy_score(df.values[:,-1], y_pred)
    #mc = confusion_matrix(df.values[:,-1], y_pred)
    #print("Matriz de Confusión: ", mc)

    return qs


def qualityFunctionVs(df_norm, wi,df,k):
    #print("longitud df_norm: ",len(df_norm))
    #print("Longitud wi: ", len(wi))
    y_pred = []
    for i in range(len(df_norm)):
        vrf = df_norm[i]
        #print(f"Vector {i} :", vrf) 
        ListaPesosPonderados= [[sys.float_info.max,0] for a in range(k)]
        for j in range(len(df_norm)):
            if i != j:
                #print(ListaPesosPonderados)
                ri = wi* np.power((df_norm[j] - vrf), 2)
                dE = np.sum(ri)
                #print("Distancia Euclideana: ", dE)
                if dE < ListaPesosPonderados[k-1][0]:
                    ListaPesosPonderados[k-1][0]=dE
                    ListaPesosPonderados[k-1][1]=j
                    ListaPesosPonderados.sort(key=lambda x: x[0], reverse=False)
        #print(f"vector {i} es muy similar a los vectores en la posición {ListaPesosPonderados}")

        grupos = []
        for i in range(len(ListaPesosPonderados)):
            index=ListaPesosPonderados[i][1]
            g = df.values[index][-1]
            grupos.append(g)
        
        #print("Grupos asociados: ", grupos)
        grupoSelected = int(pd.Series(grupos).value_counts().index[0])
        #print("Grupo seleccionado: ", grupoSelected)
        y_pred.append(grupoSelected)
    qs = accuracy_score(df.values[:,-1], y_pred)
    #mc = confusion_matrix(df.values[:,-1], y_pred)
    #print("Matriz de Confusión: ", mc)

    return qs

'''
....Hill Climbing
'''


#   PARAMETROS GENERALES
# ==============================================================================
#temp = 100
#P= 174
P=170
max_iterations = 500                                                       
nc_array = [50,55,60,65]
array_vecinos = [1,3,5]
#nc_array = [50]          
pm= 0.1                            

#  SA
# ==============================================================================

#df = BuildGroupsQuality()
#df_norm = NormalizeViewMinable(df)

df = pd.read_csv("FASE2/Prueba/Final_dataset_join.csv")
print("Longitud df: ", df.shape)
#2. Lectura de los Encoders
valMin = np.loadtxt('FASE2/Prueba/Encoder_ValMin.txt')
dataRange = np.loadtxt('FASE2/Prueba/Encoder_dataRange.txt')
df_matriz = df.values[:,:-1]

# Normalización min-maz
#df_norm = NormalizeViewMinable(df_matriz, valMin, dataRange)
df_norm = NomralizacionZscore(df_matriz)




for nceros in range(len(nc_array)):
    for vecino in range(len(array_vecinos)):
        print(f"Numero de ceros {nc_array[nceros]} vector solucion")
        print(f"Numero de vecinos {array_vecinos[vecino]}")
        
        p = np.random.rand(P)
        nc = nc_array[nceros]
        bw = 1/((P-nc)/10)
        k=array_vecinos[vecino]
        s= GenerateWeightVector(p,nc)
        qs= qualityFunctionVp(df_norm, s,df,k)
        print("QS: ", qs)
        curvaHC = []
        #vectorHC = []

        for i in range(max_iterations):
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
            qr= qualityFunctionVp(df_norm, rv,df,k)
            
            if (qr >= qs):
                s=rv
                qs = qr
            
            
            
            curvaHC.append(qs)
            #vectorHC.append(s)
        
        #Guardo el Best y Qbest despeus de reducir al minimo la temepratura
        print("Curva SA: ", curvaHC)

        '''
        dicc = {"vector":vectorBest,
                "curvaSA": curvaSA
                }
        
        df_new = pd.DataFrame(data=dicc)
        df_new.to_csv(f"ResultadosImprovisacion/SA/Training2/acc_nc_{nc}.csv")
        dicc = dict()
        '''
