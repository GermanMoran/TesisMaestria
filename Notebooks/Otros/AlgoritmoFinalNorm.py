# Librerias y dependencias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
from numpy import mean
from numpy import std
from numpy import arange
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,max_error,mean_squared_error
import sys
import random

'''
    Esta clase permite realizar la codificación en caliente especificameente variables categoricas
'''

class OneHotCoding():
    def __init__(self, df, bin_features):
        self.bin_features = bin_features
        self.df = df
    
    # Metodo para realizar la codificacion dummy a las variables categoricas

    def dummyCodification(self):
        cat_features = self.df.select_dtypes(include = ["object", "category"]).columns
        bin_dataset = self.df[self.bin_features].replace({'SI': 1, 'NO': 0})
        categorical_features = [x for x in cat_features if x not in self.bin_features]
        df_cat = pd.get_dummies(df[categorical_features])
        self.df.drop(cat_features, axis = 1, inplace = True)
        df_final = pd.concat([self.df,df_cat,bin_dataset ], axis = 1)
        df_final.to_excel("../Archivos Generados/PipelineResults/dasetOneHot.xlsx")
        print("Ejeción Terminada")
        return df_final


#categorical_transformer = Pipeline(
#    steps=[("OneHotCoding",  OneHotCoding(df,bin_features).dummyCodification())]
#)


class LinearRegession():
    def __init__(self, df, alpha, l1_ratio):
        self.df = df
        self.alpha = alpha
        self.l1_ratio = l1_ratio
    

    def CalcularModeloLR(self):
        # alpha=0.1, l1_ratio=0.97
        Y = self.df.RDT_AJUSTADO.values
        X = self.df.drop(["RDT_AJUSTADO","ID_LOTE"], axis=1).values 
        modelElasticNet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=123)
        model = modelElasticNet.fit(X,Y)
        #r_2 = model.score(X,Y)
        # Pedicciones
        yhat = model.predict(X)
        r_2 = r2_score(Y, yhat)
        
        return [model, r_2, yhat]


class CLR():

    def __init__(self, df, yhat):
        self.df = df
        self.yhat = yhat

    
    def calcularMAE(self):
        contador=0
        EPA = 0
        Acumulador = 0
        accepted_average_error= []
        self.df["yhat"]= pd.Series(self.yhat)
        self.df["EA"] = abs(self.df.RDT_AJUSTADO - self.df.yhat)
        self.df_ordely = self.df.sort_values(by=['EA'],ascending=True).reset_index()

        # Calculamos el MAE
        for i in range(len(self.df_ordely)):
            Acumulador = Acumulador + self.df_ordely.loc[i].EA
            EPA = Acumulador/(i+1)
            accepted_average_error.append(EPA)
        
        # Agregamos el promedio al dataset Ordenado.
        self.df_ordely["MAE"] = pd.Series(accepted_average_error)
        #self.df_ordely.to_excel(f"FASE1/DatasetOrdenadoIteraciónesss{contador +1}.xlsx")
        #self.df.to_excel("../Archivos Generados/PipelineResults/DatasetOriginal.xlsx")
        contador=contador+1
        return self.df_ordely
    


def DeleteRecordsGroup(Group, df):
    indexEliminar = list(Group["index"])
    df_new = df[df.index.isin(indexEliminar)== False]
    return df_new


# Retorna el grupo a modificar
def compareCorrelation(CorelacionGruposCalidad, nuevaCorrelation, listaGrupos):
    lista_grupos = listaGrupos
    #print("lista grupos: ",lista_grupos)
    arr =  np.array(CorelacionGruposCalidad) - np.array(nuevaCorrelation)
    position = np.where(arr == np.amin(arr))
    #print("Posision grupo: ",position)
    indexGrupoModificar = position[0][0]
    #print("index: ", indexGrupoModificar)
    lista_grupos.pop(indexGrupoModificar)
    return lista_grupos, indexGrupoModificar


def dataframeNormalized(dataset):
    Y = dataset.RDT_AJUSTADO
    X = dataset.drop(["RDT_AJUSTADO"], axis=1)
    # Nombre columnas de X[Variables independientes]
    X_name_columns= X.columns
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X.values)
    df_X_normalized = pd.DataFrame(X_normalized, columns= X_name_columns)
    df_normalized = pd.concat([df_X_normalized, Y], axis=1)
    print(df_normalized.shape)
    return df_normalized



# Funcion para Normalizar la Vista minable a exepción de la etiqueta(Variable Objetivo)
def EncoderViewMinable(df):
    new_dataset = df
    norm = MinMaxScaler()
    norm = norm.fit(new_dataset.values[:,:])
    valMin = norm.data_min_
    valMax = norm.data_max_
    dataRange = norm.data_range_
    #df_norm = norm.fit_transform(df.values[:,:-1])

    return [valMin, valMax, dataRange]


def fase1(dataset,Minimum_records,minimum_correlation, MAE_Allowed,additional_average_error):
    group_acepted = []
    correlation_model =[]
    model_acepted = []
    contador = 0
    while (len(dataset) !=0):
        contador = contador+ 1
        print("Tamaño del dataset: ", dataset.shape)
        modellr, r_2, yhat = LinearRegession(dataset, 0.1, 0.97).CalcularModeloLR()
        print("Ajuste del Modelo dataset Completo: ", r_2)
        DatasetOrdely = CLR(dataset,yhat).calcularMAE()
        DatasetOrdely.to_excel(f"FASE1/DatasetOrdenado{contador}.xlsx")

        try:
            group = DatasetOrdely.loc[DatasetOrdely.MAE < MAE_Allowed]
        except:
             print(f"No se cumple con el criterio de selección, verificar variable MAE_ALLOWED: {MAE_Allowed}")
             # Retornar variables vacias
             break

        print(f"Longitud de grupo {contador}: ", len(group))
        if (len(group) >= Minimum_records):
                print("El grupo cumple minimo de registros")
                group = group.drop(["yhat","EA","MAE"], axis=1)
                dataset = dataset .drop(["yhat","EA"], axis=1)
                group_model, r2_group_mode, yhat_group = LinearRegession(group,0.1,0.97).CalcularModeloLR()
                print(f"R2 del grupo {contador}: ",r2_group_mode)
                if(r2_group_mode >= minimum_correlation):
                    # Elimino los registros para la siguiente iteración
                    dataset = DeleteRecordsGroup(group, dataset)
                    group = group.drop(['index'], axis=1)
                    group_acepted.append(group)
                    model_acepted.append(group_model)
                    correlation_model.append(r2_group_mode)
                    group.to_excel(f"FASE1/GrupoN_{contador}.xlsx")
                    MAE_Allowed = MAE_Allowed + additional_average_error
                else:
                    print("No cumple con la condición de  la correlacion")
                    Orphans = dataset
                    # Se eliminan las variables de trtamiento [yhat, EA]
                    Orphans = Orphans.drop(["yhat", "EA"],axis=1).reset_index(drop=True)
                    print("Logitud Huerfanos: ", Orphans.shape)
                    break  
        else:

            Orphans = dataset
            # Se eliminan las variables de trtamiento [yhat, EA]
            Orphans = Orphans.drop(["yhat", "EA"],axis=1).reset_index(drop=True)
            print("Logitud Huerfanos: ", Orphans.shape)
            # Se guardan los huerfanos en un archivo.
            Orphans.to_excel("FASE1/HuerfanosN.xlsx")
            break
    
    return [group_acepted, model_acepted, correlation_model, Orphans]



def fase2(group_acepted, correlation_model, Orphans):
    # Variables de entrada
    quality_groups = group_acepted.copy()
    correlation_quality_groups = correlation_model.copy()
    new_correlation = []

    group_list = list(range(len(quality_groups)))
    for register in range(len(Orphans)):
        for group in range(len(quality_groups)):
            quality_groups[group] = quality_groups[group].append(Orphans.loc[0], ignore_index = True)        
            new_model, new_r2, new_yhat = LinearRegession(quality_groups[group],0.1, 0.97).CalcularModeloLR()
            new_correlation.append(new_r2)

        # Compración de las correlaciones
        list_groups_remove, index = compareCorrelation(correlation_quality_groups, new_correlation, group_list)
        correlation_quality_groups[index] = new_correlation[index]
        new_correlation = []
        group_list = list(range(len(quality_groups)))

        # Eliminacion de Regristro en los demas grupos
        for i in list_groups_remove:
            quality_groups[i].drop([len(quality_groups[i]) -1 ],axis=0, inplace=True)

        Orphans.drop([0],axis=0, inplace=True)
        Orphans = Orphans.reset_index(drop=True)

    return [quality_groups,correlation_quality_groups, Orphans]




#  Funciones Improvisación
#===================================================================================================

# Construir grupos de Calidad
def BuildGroupsQuality(definitive_groups):
    for i in range(len(definitive_groups)):
        definitive_groups[i]["Grupo"]= i

    df = pd.concat(definitive_groups, ignore_index=True)
    df = df.drop(["ID_LOTE", "RDT_AJUSTADO"], axis=1)
    return df



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
def qualityFunction(df_norm, wi,df):
    #print("longitud df_norm: ",len(df_norm))
    #print("Longitud wi: ", len(wi))
    y_pred = []
    minDep = sys.float_info.max
    posMinDep = 0
    for i in range(len(df_norm)):
        vrf = df_norm[i]
        minDep = sys.float_info.max
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

def GenerateArmonyMemory(df_norm, MAC,nc,df):
    Lw = []
    for i in range (MAC):
        # Creo la semilla
        
        vp = np.random.rand(df_norm.shape[1])
        wi = GenerateWeightVector(vp, nc)
        Qs = qualityFunction(df_norm, wi,df)
        wiq = np.append(wi, Qs)
        Lw.append(wiq)

    
    Lw.sort(key=lambda x: x[-1], reverse=True)
    return Lw



def ImprovisationGBHS(PAR, hmn, nc,dataset_normalizado , lmp, HMRC,df):

    df_norm = dataset_normalizado
    np.random.seed(123)
    # Se genera la memoria Armonica - diferentes tamaños
    MA = GenerateArmonyMemory(df_norm,hmn,nc,df)
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
            #print("------------------------------------")
            MA.sort(key=lambda x: x[-1], reverse=True)
        

        
        curvaFitnes.append(MA[0][P-1])
        vectorIteration.append(MA[0])
        

    
    '''
    print("Valor de la curva en la ultima poisción: ",curvaFitnes[-1])

    dicc = {"vector":vectorIteration,
                "Fitnes": curvaFitnes}
    

    df_new = pd.DataFrame(data=dicc)
    df_new.to_csv(f"ResultadosImprovisacion/GBHS.csv")
    dicc = dict()
    '''
    return [curvaFitnes[-1], vectorIteration[-1]]


# Funcion para Normalizar la Vista minable a exepción de la etiqueta(Variable Objetivo)
# df: es la matriz df.values [] , no incluye la etiqueta del grupo
def NormalizeViewMinable(df,valMin, dataRange):
    dataset_normalizado = np.empty((df.shape[0], df.shape[1]))
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            dataset_normalizado[i][j]= (df[i][j] - valMin[j])/dataRange[j]

    return dataset_normalizado




def predictionTest(dataset_test_norm, df_norm, vector_pesos_optmizacion, final_datset_join, list_final_models,dataset_validation, y_true):
    df_groups_finally = df_norm.copy()
    minDep = sys.float_info.max
    print("Longitud dataset norm: ",dataset_test_norm.shape[0])
    print("Vista minable : ",df_groups_finally.shape[0])
    posMinDep = 0
    MAE = 0
    PSI15 =0 
    y_pred_test = []
    for z in range(int(dataset_test_norm.shape[0])):
        minDep = sys.float_info.max
        for k in range(int(df_groups_finally.shape[0])):
            ri = vector_pesos_optmizacion * np.power((dataset_test_norm[z]- df_groups_finally[k]),2) 
            dE = np.sum(ri)
            if dE < minDep:
                posMinDep = k
                minDep=dE
        
        #print("zz, " , z)
        # Grupo seleccionado
        grupoSelected = int(final_datset_join.values[posMinDep][-1])
        #print(grupoSelected)
        #model_final, r2_final, yhat_final = LinearRegession(definitive_groups[i],0.1, 0.97).CalcularModeloLR() 
        psi_predicho = list_final_models[grupoSelected].predict(dataset_validation.values[z].reshape(1,-1))
        #print(psi_predicho)

        MAE = MAE + np.power((y_true[z][0] - psi_predicho),2)
        y_pred_test.append(psi_predicho[0])

    return [y_pred_test , MAE]



def metricasPerformanceCLR(y_true, yhat_test):

    R2 = r2_score(y_true, yhat_test)
    MAE =  mean_absolute_error(y_true, yhat_test)
    ME = max_error(y_true, yhat_test)
    MSE = mean_squared_error(y_true, yhat_test)
    print("Coeficiente de determinación : ", R2)
    print("Error absoluto medio: ", MAE)
    print("Maximo error residual: ", ME)
    print("RMSE: ", MSE)





# Variables Globales
# ==============================================================================
Minimum_records = 64
minimum_correlation= 0.88
MAE_Allowed = 143                                         #Aumentamos el MAE DE
additional_average_error = 98
contador = 0
definitive_groups = []

#1. Lectura del Dataset Principal
# ==============================================================================

df = pd.read_excel("../Archivos Generados/DatasetFinal.xlsx")
#print(df.head())
# Ornial variables list
bin_features =['SEM_TRATADAS','DRENAJE','ALMACENAMIENTO_FINCA','CAP_ENDURE_RASTA','MOTEADOS_RASTA','MOTEADOS_MAS70cm._RASTA',
               'OBSERVA_EROSION_RASTA','OBSERVA_MOHO_RASTA','OBSERVA_RAICES_VIVAS_RASTA','OBSERVA_HOJARASCA_MO_RASTA',
                'SUELO_NEGRO_BLANDO_RASTA','CUCHILLO_PRIMER_HTE_RASTA','CERCA_RIOS_QUEBRADAS_RASTA',
               ]


#2. Codificación de variables categoricas - Dummy
# ============================================================
dataset = OneHotCoding(df,bin_features).dummyCodification()
print("Dimension dataset Original: ", dataset.shape)

dataset_original = dataset.copy()
dataset_vindep = dataset.copy()
dataset_vindep= dataset_vindep.drop(["RDT_AJUSTADO","ID_LOTE"],axis=1)



#2. Encoders [Min, Max, data Range] para aplicar data Normalization
# =================================================================
'''
    Entradas: dataset: vista minable de caracteristicas indepenedientes, exepto la V objetivo
'''
valMin, valMax, dataRange = EncoderViewMinable(dataset_vindep)
print("Longitudes : ", len(valMin), len(valMax), len(dataRange))
# Guardamos los encoders apra posteriores usos
np.savetxt('FASE2/Encoder_ValMin.txt', valMin)
np.savetxt('FASE2/Encoder_dataRange.txt', dataRange)


#3. División de datset Training and Test seed 123
# ============================================================

dataset_train, dataset_test = train_test_split(dataset, test_size = 0.05, random_state=43)
print("Longitud Dataset Entrenamiento:",  dataset_train.shape)
#Guardamose l dataset de testeo.
dataset_test.to_csv('FASE2/dataset_test.csv',index=False)
dataset_training = dataset_train.copy()


# 4. Construcción de grupos de CalidaD FASE 1
# ============================================================


group_acepted, model_acepted, correlation_model, Orphans = fase1(dataset_training, Minimum_records,minimum_correlation, MAE_Allowed,additional_average_error)
print("---------------- FASE 1---------------------")
print("Correlaciones Iniciales: ", correlation_model)
#print(f"Grupo 1 {len(group_acepted[0])}, Grupo 2: {len(group_acepted[1])}, Grupo 3: {len(group_acepted[2])}")
print("Huerfanos: ", Orphans.shape)


# 5. Construccción de grupos definitivos 
# ============================================================

if len(group_acepted) > 1:
    if len(Orphans) == 0:
        definitive_groups = group_acepted
    
    else:
        # Fase 2, incluir los huerfanos
        print("FASE 2")
        quality_groups, correlation_quality_groups, orphans = fase2(group_acepted,correlation_model, Orphans)
        print("------------------ FASE 2 -------------------")
        print("Correlaciones Finales: ", correlation_quality_groups)
        #print(f"Grupo 1 {len(quality_groups[0])}, Grupo 2: {len(quality_groups[1])}, Grupo 3: {len(quality_groups[2])}")
        print("Huerfanos: ", orphans.shape)
        # Guardamos los grupos Finales
        for c, g in enumerate (quality_groups):
            g.to_excel(f"FASE2/grupo_N{c}.xlsx")
   
        
        for c, value in enumerate (correlation_quality_groups):
            if value < 0.88:
                dataset_group = quality_groups[c]
                # Se aplica el mismo proceso que la fase 1
                group_acepted2, model_acepted2, correlation_model2, Orphans2 = fase1(dataset_group ,Minimum_records,minimum_correlation, MAE_Allowed,additional_average_error)
            else:
                definitive_groups.append(quality_groups[c])


        # Aqui va el proceso de Afinamieno e improvisación 
        
else:
    definitive_groups = Orphans


print("Grupos Definitivos: ", len(definitive_groups))

# Modelos Finales
list_final_models = []
for i in range(len(definitive_groups)):
    model_final, r2_final, yhat_final = LinearRegession(definitive_groups[i],0.1, 0.97).CalcularModeloLR()
    print("R2: ", r2_final)
    list_final_models.append(model_final)
    


grupos_finales = definitive_groups.copy()



# Definición de Pesos
# =================================================================================================

#1.Union grupos | unico dataset
final_datset_join = BuildGroupsQuality(grupos_finales)
final_datset_join.to_csv('FASE2/Final_dataset_join.csv',index=False)
print(final_datset_join.shape)

'''
#2.Normalizacion dataset
df_norm = NormalizeViewMinable(final_datset_join.values[:,:-1],valMin, dataRange)
print(df_norm.shape)



#3.Optimización GBHS
# Se jecuta la mejor metahurtistica de acuerdo a los resultados obtenidos
metric,vector_pesos =  ImprovisationGBHS(0.35, 5, 40, df_norm, 20, 0.85, final_datset_join)
#metric,vector_pesos =  ImprovisationGBHS(0.4, 20, 50, df_norm, 30, 0.85, final_datset_join)
vector_pesos_optmizacion = vector_pesos[0: -1]
print("-------------------------- RESULTADOS OPTMIZACIÓN-------------------------")
print("Mejor Fitnes: ",metric)
print("Vector de pesos: ", vector_pesos_optmizacion)
print("Longitud del vector de pesos: ", len(vector_pesos_optmizacion))


'''
# Cargar un array desde una rchivo excel
vp = np.loadtxt('FASE2/bestFitnes.txt')
vector_pesos_optmizacion = vp.flatten()
'''

#  Proceso Para el conjunto de Testeo.
#===================================================================================================


# 1. Normalización conjunto de test con los encoders
dataset_validation = dataset_test.copy()
dataset_validation = dataset_validation.drop(["ID_LOTE", "RDT_AJUSTADO"], axis=1)
dataset_test_norm = NormalizeViewMinable(dataset_validation.values[:,:], valMin, dataRange)
print(dataset_test_norm.shape)


# 2. y hat Modelo CLR
y_true = dataset_test.RDT_AJUSTADO.values.reshape(-1,1)
yhat_test, MAE_ = predictionTest(dataset_test_norm, df_norm, vector_pesos_optmizacion, final_datset_join, list_final_models,dataset_validation, y_true)
print("Longitud----------: ", yhat_test )
print("MAE_: ", MAE_/len(dataset_test_norm))

np.savetxt('ResultadosImprovisacion/Comparaciones/ytest.csv', yhat_test, delimiter=',', fmt='%s')
# Verificación metricas de mesempñeo
y_true = dataset_test.RDT_AJUSTADO.values.reshape(-1,1)
np.savetxt('ResultadosImprovisacion/Comparaciones/ytrue.csv', y_true, delimiter=',', fmt='%s')
#3. Verificación metricas de mesempñeo

metrics = metricasPerformanceCLR(y_true, yhat_test)

'''