# Librerias y Dependencias
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
    sc = abs(1- np.sum(wf))
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
======Funcion para calcular la población de lobos==============
df_norm: Dataset Normalizado para calcular el fitnes asociado a cada agente
searchAgents: Numero elementos de la población (Omegas)

'''    


def GenerateWolfPopulation(df_norm,SearchAgents,nc):
    GWO = []
    for i in range (SearchAgents):
        # 1. Generar pesos aleatorios | dim = cantidad de caracteristicas
        np.random.seed(i)
        vp = np.random.rand(df_norm.shape[1]) # 50
        wi = GenerateWeightVector(vp, nc)
        # 2. Contruyo la Población de lobos
        GWO.append(wi)
      
         
    return GWO



'''

# GWO
# ==============================================================================

Entradas: 
    # Max_iter=1000
    # Los limite UB, lb depnde de los limites ede los cuales tomara valores mi población..
    # lb=-100
    # ub=100
    # dim=30
    # SearchAgents_no=5  | pop size
'''

# Variables Generales
Max_iter=15
#SearchAgents=5
nc_array = [50,55,60,65]   
popzize_array= [5,10,15,20]


df = BuildGroupsQuality()
df_norm = NormalizeViewMinable(df)
dim= df_norm.shape[1]





for inc in range(len(nc_array)):
    for ipoz in range(len(popzize_array)):
        
        print(f"=======Pruebas nc {nc_array[inc]} y popZize {popzize_array[ipoz]} ============")

        
        # GWO- Variables iniciales
        # ============================================================================================
        SearchAgents = popzize_array[ipoz]
        nc= nc_array[inc]
        Alpha_pos = np.zeros(dim)
        Alpha_fitnes = float("-inf")

        Beta_pos = np.zeros(dim)
        Beta_fitnes = float("-inf")

        Delta_pos = np.zeros(dim)
        Delta_fitnes = float("-inf")


        GWO = GenerateWolfPopulation(df_norm, SearchAgents,nc)


        Convergence_curve = np.zeros(Max_iter)
        vector_array = []
        # l: Numero de iteraciones
        for l in range(Max_iter):

            # Etapa 1: Definición Alfa, Beta y delta de la Población 

            for i in range(SearchAgents):
                # Obtengo el fitnes asociado a cada elemento de la población 
                fitness = qualityFunction(df_norm,GWO[i])
                print("fitenes: ", fitness)


                # Se actualizan las posiciones de Alfa,Beta y delta

                # 1. Actualizo Alfa
                if fitness > Alpha_fitnes:
                    Beta_fitnes = Alpha_fitnes
                    Beta_pos = Alpha_pos.copy()
                    Delta_fitnes = Beta_fitnes
                    Delta_pos = Beta_pos.copy()
                    Alpha_fitnes = fitness
                    Alpha_pos = GWO[i][:].copy()
                
                # 2. Actualizo Beta y Delta
                if fitness < Alpha_fitnes and fitness > Beta_fitnes:
                    Delta_fitnes = Beta_fitnes
                    Delta_pos = Beta_pos.copy()
                    Beta_fitnes = fitness
                    Beta_pos = GWO[i][:].copy()

                # 3. Actualizo delta
                
                if (fitness < Alpha_fitnes) and (fitness < Beta_fitnes) and (fitness > Delta_fitnes):
                    Delta_fitnes = fitness
                    Delta_pos = GWO[i][:].copy()



            print("Alfa: Fitenes:  ", Alpha_fitnes) 
            print("Beta: Fitenes:  ", Beta_fitnes) 
            print("Delta: Fitenes:  ", Delta_fitnes) 
            
            a = 2 - l * ((2) / Max_iter)

            # Etapa 2: Actualización de las posiciones de los lobos (Agentes) incluyendo los Omegas

            # Actualizo la posición de los agentes incluyendo Omegas
            for i in range(SearchAgents):
                for j in range(dim):

                    r1 = random.random()  # r1 es numero aleatorio entre 0 y 1
                    r2 = random.random()  # r2 es numero aleatorio entre 0 y 1

                    # Defino los valores de las constantes A y C para cada agente (Alfa , Beta, delta) , me guian el moviento de la manada

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2


                    D_alpha = abs(C1 * Alpha_pos[j] - GWO[i][j])
                    X1 = Alpha_pos[j] - A1 * D_alpha
            

                    r1 = random.random()
                    r2 = random.random()

                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
            

                    D_beta = abs(C2 * Beta_pos[j] - GWO[i][j])
                    X2 = Beta_pos[j] - A2 * D_beta


                    r1 = random.random()
                    r2 = random.random()

                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
            

                    D_delta = abs(C3 * Delta_pos[j] - GWO[i][j])
                    X3 = Delta_pos[j] - A3 * D_delta
                    
                    sumdis = (X1 + X2 + X3) / 3 
                    if sumdis <=  0:
                        GWO[i][j] = 0
                    else:
                        GWO[i][j] = sumdis
                    
                    #print(GWO[i][j])

                # Nomralizar
                GWO[i] = GenerateWeightVector(GWO[i],0)
                #print("Suma del nuevo vector de la población : ", sum(GWO[i]))
            
            Convergence_curve[l] = Alpha_fitnes
            vector_array.append(Alpha_pos)

            #if l % 1 == 0:
                #print(["At iteration " + str(l) + " the best fitness is " + str(Alpha_fitnes)])


        # Se muestra la curva de convergencia una vez termiandas las iteraciones
        print("Curva de Convergencia: ", Convergence_curve)    
        
        dicc = {"vector":vector_array,
            "Fitnes": Convergence_curve}

    
        df_new = pd.DataFrame(data=dicc)
        df_new.to_csv(f"ResultadosImprovisacion/GWO/accuracy_nc_{nc}_PopZize_{SearchAgents}.csv")
        dicc = dict()

        

