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
        #self.df_ordely.to_excel("../Archivos Generados/PipelineResults/DatasetOrdenadoMAE.xlsx")
        #self.df.to_excel("../Archivos Generados/PipelineResults/DatasetOriginal.xlsx")
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



df = pd.read_excel("../Archivos Generados/DatasetFinal.xlsx")
print(df.head())
# Ornial variables list
bin_features =['SEM_TRATADAS','DRENAJE','ALMACENAMIENTO_FINCA','CAP_ENDURE_RASTA','MOTEADOS_RASTA','MOTEADOS_MAS70cm._RASTA',
               'OBSERVA_EROSION_RASTA','OBSERVA_MOHO_RASTA','OBSERVA_RAICES_VIVAS_RASTA','OBSERVA_HOJARASCA_MO_RASTA',
                'SUELO_NEGRO_BLANDO_RASTA','CUCHILLO_PRIMER_HTE_RASTA','CERCA_RIOS_QUEBRADAS_RASTA',
               ]





# Variables Globales
# ==============================================================================
Minimum_records = 80
minimum_correlation= 0.88
MAE_Allowed = 45
additional_average_error = 50
contador = 0
definitive_groups = []


dataset = OneHotCoding(df,bin_features).dummyCodification()
# ============================================================

df_name_columns = dataset.columns
scaler = MinMaxScaler()
dataset_normalizado = scaler.fit_transform(dataset)
dataset_n = pd.DataFrame(dataset_normalizado, columns = df_name_columns)

# ==============================================================
group_acepted, model_acepted, correlation_model, Orphans = fase1(dataset_n,Minimum_records,minimum_correlation, MAE_Allowed,additional_average_error)
print("---------------- FASE 1---------------------")
print("Correlaciones Iniciales: ", correlation_model)
print(f"Grupo 1 {len(group_acepted[0])}, Grupo 2: {len(group_acepted[1])}, Grupo 3: {len(group_acepted[2])}")
print("Huerfanos: ", Orphans.shape)




if len(group_acepted) > 1:
    if len(Orphans) == 0:
        definitive_groups = group_acepted
    
    else:
        # Fase 2, incluir los huerfanos
        print("FASE 2")
        quality_groups, correlation_quality_groups, orphans = fase2(group_acepted,correlation_model, Orphans)
        print("------------------ FASE 2 -------------------")
        print("Correlaciones Finales: ", correlation_quality_groups)
        print(f"Grupo 1 {len(quality_groups[0])}, Grupo 2: {len(quality_groups[1])}, Grupo 3: {len(quality_groups[2])}")
        print("Huerfanos: ", orphans.shape)
        # Guardamos los grupos Finales
        for c, g in enumerate (quality_groups):
            g.to_excel(f"FASE2/grupo{c}.xlsx")
   
        
        for c, value in enumerate (correlation_quality_groups):
            if value < 0.88:
                dataset_group = quality_groups[c]
                # Se aplica el mismo proceso que la fase 1
                group_acepted2, model_acepted2, correlation_model2, Orphans2 = fase1(dataset_group ,Minimum_records,minimum_correlation, MAE_Allowed,additional_average_error)
            else:
                definitive_groups.append(quality_groups[c])
       
else:
    definitive_groups = Orphans


print("Grupos Definitivos: ", len(definitive_groups))