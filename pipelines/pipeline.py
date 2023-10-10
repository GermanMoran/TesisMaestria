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
        print(self.alpha)
        print(self.l1_ratio)
        Y = self.df.RDT_AJUSTADO
        X = self.df.drop(["RDT_AJUSTADO","ID_LOTE"], axis=1) 
        modelElasticNet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        model = modelElasticNet.fit(X,Y)
        r_2 = model.score(X,Y)
        return [model, r_2]


class CLR():
    def __init__(self, df, alpha, l1_ratio):
        self.df = df
        self.alpha = alpha
        self.l1_ratio = l1_ratio
    



# Dataset a trabajar
dataset = OneHotCoding(df,bin_features).dummyCodification()
print(dataset.RDT_AJUSTADO)
modellr, r_2 = LinearRegession(dataset, 0.1, 0.97).CalcularModeloLR()
print("Ajuste del Modelo dataset Completo: ", r_2)



