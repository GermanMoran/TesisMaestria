
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

    # Se Unen los diferentes Grupos en un unico dataset
    df = pd.concat([GC1, GC2,GC3], ignore_index=True)

    # Elimino las variables no relevantes
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
    # Se seleccionan diferentes posiciones| atributos a establecer  en cero
    posceros = np.random.choice(len(w), nc, replace=False)
    w[posceros] = 0
    w = np.round(w,3)
    s  = np.sum(w)
    # Normalizo los pesos - Solo obtengo los 3 decimales de c/d peso
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
    # Vector Distancias Euclideanas Ponderadas
    y_pred = []
    minDep = sys.float_info.max
    posMinDep = 0
    for i in range(len(df_norm)):
        vrf = df_norm[i] 
        for j in range(len(df_norm)):
            if i != j:
                ri = wi* np.power((df_norm[j] - vrf), 2)
                #dE = np.sqrt(np.sum(ri))
                dE = np.sum(ri)
                if dE < minDep:
                    posMinDep=j
                    minDep = dE
              
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
        # 1. Generar pesos aleatorios | dim = cantidad de caracteristicas
        vp = np.random.rand(df_norm.shape[1])
        wi = GenerateWeightVector(vp, nc)
        # 2. Se obtiene el fitnes asociados a esos pesos.
        Qs = qualityFunction(df_norm, wi)
        wiq = np.append(wi, Qs)
        Lw.append(wiq)

    
    # Se obtiene la memoria armonica ordenada de  Mayor- Menor (Deacuerdo al fitnes)
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
    nc: Numero de ceros [10,20,30,40,50]
'''

lmp = 20                                                                                
HMRC = 0.85                                        
PAR =  0.35                                                         
hmns = [5]                                          
nc= 50

#1. Se contruyen los grupos de calidad
df = BuildGroupsQuality()
#2. Se Normaliza el df
df_norm = NormalizeViewMinable(df)

# 3. Implementacion GBHS

for hmn in hmns:

   # Se genera la memoria Armonica - diferentes tamaños
    MA = GenerateArmonyMemory(df_norm,hmn,nc)
    P=len(MA[0]) 

    
    #for i in range(hmn):
    #    print("Memoria Armonica: ", MA[i][P-1])

    
    curva = []
    for i in range (lmp):
        print(f"Iteracion {i}")
        pesosAleatorios = np.random.rand(P-1)
        # print("Vector de Pesos: ", pesosAleatorios)
        # Donde P es el numero de variables
        for j in range(P-1):
            Aleatorio1 = random.random() 
            if (Aleatorio1 < HMRC):
                # pma: Numero entrero entre 0 y (hmns -1)
                # hmns: Tamaño de la memoria armonica
                pma = random.randint(0, hmn-1)
                pesosAleatorios[j]= MA[pma][j]

                Aleatorio2 = random.random()
                if Aleatorio2 < PAR:
                    # Se toma el valor de la mejor armonia
                    pesosAleatorios[j] = MA[0][j]
            
            else:
                Aleatorio3 = random.random()
                if Aleatorio3 < nc/P:
                    Aleatorio3 = 0
                else:
                    Aleatorio3 = random.random()/(P-nc)
                    #Aleatorio3 = random.random()/(P/2)
                
                pesosAleatorios[j] = Aleatorio3
        
        # Fin primer for - Conformación vector de pesos
        # Normalización de los pesos
        wf = GenerateWeightVector(pesosAleatorios, 0)
        fitnes = qualityFunction(df_norm, wf)
        print("Funcion de calidad: ", fitnes)


        # Remplazo
        if MA[hmn -1][P-1] < fitnes:
            # Creo el nuevo registro y  remplazo por el peor
            new_register = np.append(wf, fitnes)
            # Remplazo
            MA[hmn-1] = new_register
            # Ordeno la Armonia
            print("------------------------------------")
            MA.sort(key=lambda x: x[-1], reverse=True)
        
        else:
            print("La funcion de calidad del ultimo armonico  es mejor que el fitnes")
        
        curva.append(MA[0][P-1])




    # Curva de las iteraciones 
    print(curva)
