
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




print("Numero Aleatorio Flotante:", sys.float_info.max)

# Funcion para construir el dataset - Asociado a los respectivos grupos
def BuildGroupsQuality():
    GC1 = pd.read_excel("../Archivos Generados/PipelineResults/GruposCalidad-Fase2/Grupo1.xlsx",index_col=0)
    GC2 = pd.read_excel("../Archivos Generados/PipelineResults/GruposCalidad-Fase2/Grupo2.xlsx",index_col=0)
    GC3 = pd.read_excel("../Archivos Generados/PipelineResults/GruposCalidad-Fase2/Grupo3.xlsx",index_col=0)


    GC1["Grupo"] = 1
    GC2["Grupo"] = 2
    GC3["Grupo"] = 3


    df = pd.concat([GC1, GC2,GC3], ignore_index=True)
    df = df.drop(["ID_LOTE", "RDT_AJUSTADO"], axis=1)
    return df



# Funcion para Normalizar la Vista minable a exepción de la etiqueta(Variable Objetivo)
def NormalizeViewMinable(df):
    norm = MinMaxScaler()
    df_norm = norm.fit_transform(df.values[:,:-1])
    return df_norm





'''
Funcion para generar el vector de pesos aleatorio.
Entradas:
w: Vector de pesos w [0,1], igual al numero de caracteristicas normalizadas.
nc: Nunero de ceros que debe contener el vector de pesos [10,20,30,40,50]
'''
def GenerateWeightVector(w, nc):
    posceros = np.random.choice(len(w), nc, replace=False)
    w[posceros] = 0
    w = np.round(w,3)
    s  = np.sum(w)
    wf = np.round(w/s, 3)
    sc = 1- np.sum(wf)
    pos = np.random.randint(len(wf))
    if sc != 0:
        wf[pos]= wf[pos]+sc

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
def qualityFunction(df_norm, wi):
    #print("longitud df_norm: ",len(df_norm))
    #print("Longitud wi: ", len(wi))
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
        #print(posMinDep)
        y_pred.append(df.values[posMinDep][-1])
    qs = accuracy_score(df.values[:,-1], y_pred)
    return qs




'''
Función para generar la memoria Armonica
Entradas:
    MAC: Tamaño de la  Memoria Armonica
    wi: Vector de Pesos.
    nc: Numero de ceros (Selección de atributos)
'''

def GenerateArmonyMemory(df_norm, MAC,nc):
    Lw = []
    for i in range (MAC):
        # Creo la semilla
        np.random.seed(2)
        vp = np.random.rand(df_norm.shape[1])
        wi = GenerateWeightVector(vp, nc)
        Qs = qualityFunction(df_norm, wi)
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

excel = "ResultadosImprovisacion/resultados_GBHS.xlsx"
lmp = 10                                                                                
HMRC = 0.85                                          
#PAR_array=[0.25,0.3, 0.35,0.40]                                                         
#hmns_array = [5,10,15,20]  
#nc_array = [50,55,60,65]
PAR_array=[0.35]                                                         
hmns_array = [5,10,15,20]  
nc_array = [50,55,60,65]                                            


#1. Se contruyen los grupos de calidad
df = BuildGroupsQuality()
df_norm = NormalizeViewMinable(df)


cont = 1

# 
# GBHS - IMPROVISACIÓN
# ============================================================================================
for incero in range(len(nc_array)):
    for ihmn in range(len(hmns_array)):
        for ipar in range(len(PAR_array)):
            print(f"PAR: {PAR_array[ipar]}, Longitud de la Memoria Armonica: {hmns_array[ihmn]}, Numero de ceros: {nc_array[incero]} ")
            print(f"Iteracion {cont}")
            # Parametros Bucle
            PAR = PAR_array[ipar]
            hmn = hmns_array[ihmn]
            nc= nc_array[incero]

            # Se genera la memoria Armonica - diferentes tamaños
            MA = GenerateArmonyMemory(df_norm,hmn,nc)
            P=len(MA[0]) 

            curvaFitnes = []
            vectorIteration= []
            for i in range (lmp):
                # seed
                np.random.seed(123)
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
                fitnes = qualityFunction(df_norm, wf)



                # Remplazo
                if MA[hmn -1][P-1] < fitnes:
                    new_register = np.append(wf, fitnes)
                    MA[hmn-1] = new_register
                    #print("------------------------------------")
                    MA.sort(key=lambda x: x[-1], reverse=True)
                

                
                curvaFitnes.append(MA[0][P-1])
                vectorIteration.append(MA[0])
                
 
            
            print("Valor de la curva en la ultima poisción: ",curvaFitnes[-1])

            dicc = {"vector":vectorIteration,
                        "Fitnes": curvaFitnes}
            
    
            df_new = pd.DataFrame(data=dicc)
            df_new.to_csv(f"ResultadosImprovisacion/GBHS/accuracy_PAR_{PAR}_Tmem_{hmn}_nc_{nc}.csv")
            cont=cont+1
            dicc = dict()
            

            

           