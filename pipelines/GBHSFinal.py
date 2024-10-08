
# Librerias y Dependencias
# ==============================================================================
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from sklearn.metrics import f1_score, accuracy_score
import sys
import math





'''
# Funcion para Normalizar la Vista minable a exepción de la etiqueta(Variable Objetivo)
# df: es la matriz df.values [] , no incluye la etiqueta del grupo
# valMin: Valores minimos de cada columna , utilizadas para hacer la normalización
# dataRange: Rango de datos [valmax - val min] para cada columna
'''

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
    nc: Nunero de ceros que debe contener el vector de pesos [10,20,30,40,50..]
salidas:
    wf: vector de pesos normalizado
'''
def GenerateWeightVector(w, nc):
    posceros = np.random.choice(len(w), nc, replace=False)
    w[posceros] = 0
    s  = np.sum(w)
    wf = w/s
    return wf





'''
-Función de calidad, que retorna la metrica de calidad asociada a ese vector w especifico
-Se debe tener en cuenta la seleccion de la metrica de calidad asociada, para evaluar
el desempeño del algoritmo (Problema de Clasificacion)
Entradas:
    df_norm: Dataset Normalizado.
    wi: Vector de Pesos.
    df: Dataframe Original con grupos armados
'''


def qualityFunction(df_norm, wi,df):
    y_pred = []
    for i in range(len(df_norm)):
        posMinDep = 0
        minDep = sys.float_info.max 
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
Implmentación de la funcion voto simple (K valores)
def qualityFunction(df_norm, wi,df,k):
    y_pred = []
    for i in range(len(df_norm)):
        vrf = df_norm[i]
        ListaPesosPonderados= [[sys.float_info.max,0] for a in range(k)]
        for j in range(len(df_norm)):
            if i != j:
                ri = wi* np.power((df_norm[j] - vrf), 2)
                dE = np.sum(ri)
                if dE < ListaPesosPonderados[k-1][0]:
                    ListaPesosPonderados[k-1][0]=dE
                    ListaPesosPonderados[k-1][1]=j
                    ListaPesosPonderados.sort(key=lambda x: x[0], reverse=False)
        grupos = []
        for i in range(len(ListaPesosPonderados)):
            index=ListaPesosPonderados[i][1]
            g = df.values[index][-1]
            grupos.append(g)
        
        grupoSelected = int(pd.Series(grupos).value_counts().index[0])
        y_pred.append(grupoSelected)
    qs = accuracy_score(df.values[:,-1], y_pred)
    return qs
'''





'''
Función para generar la memoria Armonica
Entradas:
    df_norm: datataframe normalizado. [Matriz]
    MAC: Tamaño de la  Memoria Armonica
    nc: Numero de ceros (Selección de atributos)
    df: Dataframe original , para ubicar el grupo asociado a la menor distancia.

Salidas:
    Memoria Armonica [wi | fitnes] ordenada de mayor a menor decuerdo al fitnes.
'''

def GenerateArmonyMemory(df_norm, MAC,nc,df):
    Lw = []
    for i in range (MAC):
        vp = np.random.rand(df_norm.shape[1])
        # Normalizacion vector de pesos.
        wi = GenerateWeightVector(vp, nc)
        Qs = qualityFunction(df_norm, wi,df)
        wiq = np.append(wi, Qs)
        Lw.append(wiq)

    
    Lw.sort(key=lambda x: x[-1], reverse=True)
    return Lw




# GBHS
# ==============================================================================

'''
Entradas
    lmp: Numero de improvisaciones (iteraciones) que realiza GBHS.
    P: Numero de Atributos
    HMRC: Tasa de Consideración de la memoria Armonica
    ParMin: Tasa de ajuste de tono mínima.
    ParMax: Tasa de ajuste de tono máxima.
    PAR: realación(ParMin, ParMax) : [0.1,0.25, 0.35, 0.40]
    hmns: Tamaño de la memoria armonica  [5,10,15,20].
    nc: Numero de ceros [10,20,30,40,50]    [10,20,30,40,50,60,70]  
'''


lmp = 100                                                                               
HMRC_array = [0.85,0.90]                                          
PAR_array=[0.3, 0.35,0.40]                                                         
hmns_array = [10,5,15,20]  
#nc_array = [30,40,50,60]
nc_array = [0]
#k_array=[1,3,5,7]   
k_array=[1]                                       


#1. Se contruyen los grupos de calidad
df = pd.read_csv("FASE2/Final_dataset_join.csv")
print("Longitud df: ", df.shape)
# Lectura de los Encoders | Normalización del dataframe
valMin = np.loadtxt('FASE2/Encoder_ValMin.txt')
print("Encoder: ", len(valMin))
dataRange = np.loadtxt('FASE2/Encoder_dataRange.txt')
df_matriz = df.values[:,:-1]
print(df_matriz.shape)
df_norm = NormalizeViewMinable(df_matriz,valMin,dataRange)
#np.savetxt('FASE2/datset_normalizado_join.csv',df_norm, delimiter=',')


cont = 1



# GBHS - IMPROVISACIÓN
# ============================================================================================
for ik in range(len(k_array)):
    for incero in range(len(nc_array)):
        for ihmn in range(len(hmns_array)):
            for ipar in range(len(PAR_array)):
                for ihmrc in range(len(HMRC_array)):
                    print(f"k values: {k_array[ik]} , PAR: {PAR_array[ipar]}, Longitud de la Memoria Armonica: {hmns_array[ihmn]}, Numero de ceros: {nc_array[incero]}, HMRC: {HMRC_array[ihmrc]}")
                    print(f"Iteracion {cont}")
                    # Parametros Bucle
                    PAR = PAR_array[ipar]
                    hmn = hmns_array[ihmn]
                    nc= nc_array[incero]
                    HMRC = HMRC_array[ihmrc]
                    k= k_array[ik]

                    # Se genera la memoria Armonica - diferentes tamaños
                    #np.random.seed(43)
                    MA = GenerateArmonyMemory(df_norm,hmn,nc,df)
                    # (P-1): Numero de atributos y/o dimensiones
                    P=len(MA[0]) 

                    curvaFitnes = []
                    vectorIteration= []
                    for i in range (lmp):
                        pesosAleatorios = np.random.rand(P-1)
                        for j in range(P-1):
                            Aleatorio1 = random.random() 
                            if (Aleatorio1 < HMRC):
                                pma = random.randint(0, hmn-1)
                                pesosAleatorios[j]= MA[pma][j]

                                Aleatorio2 = random.random()
                                if Aleatorio2 < PAR:
                                    pesosAleatorios[j] = MA[0][j]
                            
                            else:
                                Aleatorio3 = random.random()
                                if Aleatorio3 < nc/P:
                                    Aleatorio4 = 0
                                else:
                                    Aleatorio4 = Aleatorio3/(P-nc)
                                
                                pesosAleatorios[j] = Aleatorio4
                        

                        # Normalización de los pesos
                        wf = GenerateWeightVector(pesosAleatorios, 0)
                        fitnes = qualityFunction(df_norm, wf,df)



                        # Remplazo
                        if MA[hmn -1][P-1] < fitnes:
                            new_register = np.append(wf, fitnes)
                            MA[hmn-1] = new_register
                            MA.sort(key=lambda x: x[-1], reverse=True)
                        

                        
                        curvaFitnes.append(MA[0][P-1])
                        vectorIteration.append(MA[0])
                        
        
                    
                    print("Valor de la curva en la ultima poisción: ",curvaFitnes[-1])

                    dicc = {"vector":vectorIteration,
                                "Fitnes": curvaFitnes}
                    
            
                    df_new = pd.DataFrame(data=dicc)
                    df_new.to_csv(f"ResultadosImprovisacion/GBHS/Training4/accuracy_PAR_{PAR}_Tmem_{hmn}_nc_{nc}.csv")
                    cont=cont+1
                    dicc = dict()
                    

           