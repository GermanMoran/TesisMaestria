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


# Read to dataset
df = pd.read_excel("../Archivos Generados/DatasetFinal.xlsx")
print(df.head())
# Ornial variables list
bin_features =['SEM_TRATADAS','DRENAJE','ALMACENAMIENTO_FINCA','CAP_ENDURE_RASTA','MOTEADOS_RASTA','MOTEADOS_MAS70cm._RASTA',
               'OBSERVA_EROSION_RASTA','OBSERVA_MOHO_RASTA','OBSERVA_RAICES_VIVAS_RASTA','OBSERVA_HOJARASCA_MO_RASTA',
                'SUELO_NEGRO_BLANDO_RASTA','CUCHILLO_PRIMER_HTE_RASTA','CERCA_RIOS_QUEBRADAS_RASTA',
               ]



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
        Y = self.df.RDT_AJUSTADO
        X = self.df.drop(["RDT_AJUSTADO","ID_LOTE"], axis=1) 
        modelElasticNet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=123)
        model = modelElasticNet.fit(X,Y)
        r_2 = model.score(X,Y)
        # Pedicciones
        yhat = model.predict(X)

        return [model, r_2, yhat]


class CLR():

    def __init__(self, df, yhat):
        self.df = df
        self.yhat = yhat

    
    def calcularMAE(self):
        EPA = 0
        Acumulador = 0
        accepted_average_error= []
        self.df["yhat"]= pd.Series(yhat)
        self.df["EA"] = abs(self.df.RDT_AJUSTADO - self.df.yhat)
        self.df_ordely = self.df.sort_values(by=['EA'],ascending=True).reset_index()

        # We calculated MAE
        for i in range(len(self.df_ordely)):
            Acumulador = Acumulador + self.df_ordely.loc[i].EA
            EPA = Acumulador/(i+1)
            accepted_average_error.append(EPA)
        
        # we agregate the average to dataset ordely
        self.df_ordely["MAE"] = pd.Series(accepted_average_error)
        #self.df_ordely.to_excel("../Archivos Generados/PipelineResults/DatasetOrdenadoMAE.xlsx")
        #self.df.to_excel("../Archivos Generados/PipelineResults/DatasetOriginal.xlsx")
        return self.df_ordely
    


def DeleteRecordsGroup(Group, df):
    indexEliminar = list(Group["index"])
    df_new = df[df.index.isin(indexEliminar)== False]
    return df_new



def compareCorrelation(CorelacionGruposCalidad, nuevaCorrelation, listaGrupos):
    lista_grupos = listaGrupos
    arr =  np.array(CorelacionGruposCalidad) - np.array(nuevaCorrelation)
    position = np.where(arr == np.amin(arr))
    indexGrupoModificar = position[0][0]
    lista_grupos.pop(indexGrupoModificar)
    return lista_grupos, indexGrupoModificar


# Variables Globales
# =========================================================================================
Minimum_records = 80
minimum_correlation= 0.88
MAE_Allowed = 45
group_acepted = []
correlation_model =[]
model_acepted = []
additional_average_error = 50
contador = 0



#  FASE 1
# =========================================================================================

dataset = OneHotCoding(df,bin_features).dummyCodification()
#print(dataset.RDT_AJUSTADO)
while (len(dataset) !=0):
    contador = contador+1
    print("Tamaño del dataset: ", dataset.shape)
    modellr, r_2, yhat = LinearRegession(dataset, 0.1, 0.97).CalcularModeloLR()
    print("Ajuste del Modelo dataset Completo: ", r_2)
    DatasetOrdely = CLR(dataset,yhat).calcularMAE()
    group = DatasetOrdely.loc[DatasetOrdely.MAE < MAE_Allowed]
    if (len(group) >= Minimum_records):
            print("El grupo cumple minimo de registros")
            group = group.drop(["yhat","EA","MAE"], axis=1)
            dataset = dataset .drop(["yhat","EA"], axis=1)
            group_model, r2_group_mode, yhat_group = LinearRegession(group,0.1,0.97).CalcularModeloLR()
            print(f"R2 del grupo {contador}: ",r2_group_mode)
            if(r2_group_mode >= 0.88):
                # Elimino los registros para la siguiente iteración
                dataset = DeleteRecordsGroup(group, dataset)
                group = group.drop(['index'], axis=1)
                group_acepted.append(group)
                model_acepted.append(group_model)
                correlation_model.append(r2_group_mode)
                group.to_excel(f"../Archivos Generados/PipelineResults/Grupo{contador}.xlsx")
                MAE_Allowed = MAE_Allowed + additional_average_error
            else:
                break  
    else:
        Orphans = dataset
        Orphans = Orphans.drop(["yhat", "EA"],axis=1).reset_index(drop=True)
        print("Logitud Huerfanos: ", Orphans.shape)
        # Aqui hay que eliminar columnas EA , yhat
        Orphans.to_excel("../Archivos Generados/PipelineResults/Huerfanos.xlsx")
        break


#  FASE 2
# ==================================================================================================

# Control Variables
GruposCalidad = group_acepted
CorelacionGruposCalidad = correlation_model
print(f"Grupo Calidad 1: {len(GruposCalidad[0])} , Grupo de Calidad 2: {len(GruposCalidad[1])}, Grupo de Calidad 3: {len(GruposCalidad[2])}")
print("Correlaciones Grupos de calidad: ", CorelacionGruposCalidad)

listaGrupos = [0, 1, 2]
nuevaCorrelation= []

for register in range(len(Orphans)):
    for group in range(len(GruposCalidad)):
        GruposCalidad[group] = GruposCalidad[group].append(Orphans.loc[0], ignore_index = True)
        new_mode, new_r2, new_that = LinearRegession(GruposCalidad[group],0.1, 0.97).CalcularModeloLR()
  
        nuevaCorrelation.append(new_r2)
        


    lista_id_grupos_retirar, index = compareCorrelation(CorelacionGruposCalidad, nuevaCorrelation, listaGrupos)
    CorelacionGruposCalidad[index] = nuevaCorrelation[index]
    nuevaCorrelation = []
    listaGrupos = [0, 1, 2]
 
    for i in lista_id_grupos_retirar:
        GruposCalidad[i].drop([len(GruposCalidad[i]) -1 ],axis=0, inplace=True)

    Orphans.drop([0],axis=0, inplace=True)
    Orphans = Orphans.reset_index(drop=True)




print(f"Grupo Calidad 1: {len(GruposCalidad[0])} , Grupo de Calidad 2: {len(GruposCalidad[1])}, Grupo de Calidad 3: {len(GruposCalidad[2])}")
print("Correlacion grupos finales: ", CorelacionGruposCalidad)
print("Longitud de los huerfanos: ", len(Orphans))

# Guardamos los grupos de calidad definitivos
GruposCalidad[0].to_excel("../Archivos Generados/PipelineResults/GruposCalidad-Fase2/Grupo1.xlsx")
GruposCalidad[1].to_excel("../Archivos Generados/PipelineResults/GruposCalidad-Fase2/Grupo2.xlsx")
GruposCalidad[2].to_excel("../Archivos Generados/PipelineResults/GruposCalidad-Fase2/Grupo3.xlsx")
