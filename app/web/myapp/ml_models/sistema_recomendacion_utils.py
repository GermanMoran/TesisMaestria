#  Librerias y Dependencias
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import random
import joblib

import os

from myapp.models import CultivoMaiz

# Plot convergence curve
def plot_convergence_curve(fitness_history, alg):
  efos = np.arange(len(fitness_history))
  plt.title("Convergence curve")
  plt.xlabel("EFOs")
  plt.ylabel("Fitness (Ganancia[$])")
  plt.plot(efos, fitness_history, label= str(alg))
  plt.legend()
  #plt.savefig("Convergence curve for " + str(f) + "" + str(alg) + ".png")
  plt.show()


  #Plot convergence curve comparison for two or more algorithms
def plot_convergence_curve_comparison(fitness_history, f, alg):
  efos = np.arange(len(fitness_history[0]))
  plt.title("Convergence curve for HC, SA y GWO")
  plt.xlabel("EFOs")
  plt.ylabel("Fitness")
  algorithms = len(alg)
  for a in range(algorithms):
    plt.plot(efos, fitness_history[a], label=str(alg[a]))
  plt.legend()
  #plt.savefig("Convergence curve for " + str(f) + "" + str(alg) + ".png")
  plt.show()

def print_alorithms_with_avg_fitness(alg, avg_fitness):
  rows = len(alg)
  for r in range(rows):
    print(alg[r] + " {0:12.6f}".format(avg_fitness[r]))

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
        df_cat = pd.get_dummies(self.df[categorical_features])
        self.df.drop(cat_features, axis = 1, inplace = True)
        df_final = pd.concat([self.df,df_cat,bin_dataset ], axis = 1)
        #df_final.to_csv("daset_codificado.csv")
        #print("Ejeción Terminada")
        return df_final, categorical_features

'''
    Esta es la función Objetivo a maximizar (Prediccion Generada Algortimo ClR)
'''

class CLR():
  def __init__(self, modelo_clasificacion, modelo_regresion):
        self.modelo_clasificacion = modelo_clasificacion
        self.modelo_regresion = modelo_regresion

  def etapaClasficacion(self):
    grupo_asignado = self.modelo_clasificacion.predict(self.registro)
    return grupo_asignado[0]

  def etapaRegresion(self, ga):
    prediccion = self.modelo_regresion[ga].predict(self.registro.values)
    return prediccion[0]

  def evaluate(self,registro):
    grupo_asignado = self.modelo_clasificacion.predict(registro)[0]
    prediccion = self.modelo_regresion[grupo_asignado].predict(registro.values)
    return prediccion[0]





'''
    Esta clase nos permite genera:
    - Solucioción Inicial con su respectivo ajuste
    - Solucion modificada Tweck Update deacuerdo rango de Variables de Optimización
'''

class Solution():
  def __init__(self, variablesOptimizar, f,f_cost, reg_opt, col_train_model,cf):
    self.variablesOptimizar = variablesOptimizar
    self.cf = cf
    self.function = f
    self.fitness = 0
    self.reg_opt = reg_opt
    self.col_train_model = col_train_model
    self.f_cost = f_cost



  def initialization(self):
    df_opt= self.reg_opt.copy()
    df_opt_dm = pd.get_dummies(df_opt, columns=self.cf,dtype=np.int64)
    df_opt_dm_m = df_opt_dm.reindex(columns=self.col_train_model, fill_value=0)
    #print(df_opt_dm.shape)
    self.fitness= self.function.evaluate(df_opt_dm_m)
    return self.fitness, df_opt, df_opt_dm_m

  def twekUpdate(self,bandwidth):
    dff= self.reg_opt.copy()
    #Proceso twick [variables categoricas y continuas]
    #========================================================================
    bandwidths = np.random.uniform(low=-bandwidth, high=bandwidth)
    #print("Bandwitdt Generado: ", bandwidths)
    var_select = list(np.random.choice(self.variablesOptimizar, 3, replace=False))
    #print("var selected: ", var_select)
    list_vac = []
    for variable in var_select:
        match variable:
            case 'TIPO_SIEMBRA':
              if dff[variable].values[0] == "Manual":
                  list_vac.append("Mecanizado")
              else:
                  list_vac.append("Manual")

            case 'SEM_TRATADAS':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'MATERIAL_GENETICO':
              MG =  ['Otro', 'P3966 (Pioneer)', 'P4082 (Pioneer)', 'DK 234', 'PAC 105',
              'NK254 (Syngenta)', 'PIONEER 30F35', 'PIONEER 30F35 HRR',
              'DK 234 YGRR', 'Impacto (Syngenta)', 'ICA V 305',
              'PIONEER 30F35 H', 'ADV 9339 (Syngenta)', 'DK7088',
              'PIONEER 30F32', 'CORPOICA V 114', 'Sinko (Syngenta)',
              'ADV 9293 (Syngenta)', 'DK 1596', 'Status (Syngenta)',
              'Cerato (Syngenta)', 'ICA V 156', 'ICA V 109', 'PIONEER 30F32HW',
              'FNC 3056', 'FNC 114', 'DK 1040']
              filter_MG = [v for v in MG if v != dff[variable].values[0]]
              MG_Selected = np.random.choice(filter_MG)
              list_vac.append(MG_Selected)

            case 'DRENAJE':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'METODO_COSECHA':
              if dff[variable].values[0] == "Manual":
                  list_vac.append("Mecanizada")
              else:
                  list_vac.append("Manual")

            case 'ALMACENAMIENTO_FINCA':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'ContEnfQui_Emer_Flor':
              r = np.random.choice([x for x in range(4) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContEnfQui_Flor_Cose':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'ContMalMec_Siem_Emer':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'ContMalMec_Emer_Flor':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'ContMalMec_Flor_Cose':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'ContMalQui_Antes_Siem':
              r = np.random.choice([x for x in range(3) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContMalQui_Siem_Emer':
              r = np.random.choice([x for x in range(4) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContMalQui_Emer_Flor':
              r = np.random.choice([x for x in range(5) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContMalQui_Flor_Cose':
              r = np.random.choice([x for x in range(2) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContPlaQui_Antes_Siem':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'ContPlaQui_Siem_Emer':
              r = np.random.choice([x for x in range(3) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContPlaQui_Emer_Flor':
              r = np.random.choice([x for x in range(10) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContPlaQui_Flor_Cose':
              r = np.random.choice([x for x in range(3) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'TotN_Antes_Siem':
              valor = dff[variable].values[0] + bandwidths
              valor = round(np.clip(valor, 0, 105),2)
              list_vac.append(valor)

            case 'TotN_Siem_Emer':
              valor = dff[variable].values[0] + bandwidths
              valor = round(np.clip(valor, 0, 82.5),2)
              list_vac.append(valor)

            case 'TotN_Emer_Flor':
              valor = dff[variable].values[0] + bandwidths
              valor = round(np.clip(valor, 0, 276),2)
              list_vac.append(valor)

            case 'TotP_Antes_Siem':
              valor = dff[variable].values[0] + bandwidths
              valor = round(np.clip(valor, 0, 24),2)
              list_vac.append(valor)

            case 'TotP_Siem_Emer':
              valor = dff[variable].values[0] + bandwidths
              valor = round(np.clip(valor, 0, 40),2)
              list_vac.append(valor)

            case 'TotP_Emer_Flor':
              valor = dff[variable].values[0] + bandwidths
              valor = round(np.clip(valor, 0, 47.5),2)
              list_vac.append(valor)

            case 'TotK_Antes_Siem':
              valor = dff[variable].values[0] + bandwidths
              valor = round(np.clip(valor, 0, 30),2)
              list_vac.append(valor)

            case 'TotK_Siem_Emer':
              valor = dff[variable].values[0] + bandwidths
              valor = round(np.clip(valor, 0, 75),2)
              list_vac.append(valor)

            case 'TotK_Emer_Flor':
              valor = dff[variable].values[0] + bandwidths
              valor = round(np.clip(valor, 0, 180),2)
              list_vac.append(valor)

            case 'FerOrg_Emer_Flor':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'FerQui_Antes_Siem':
              r = np.random.choice([x for x in range(4) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'FerQui_Siem_Emer':
              r = np.random.choice([x for x in range(4) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'FerQui_Emer_Flor':
              r = np.random.choice([x for x in range(8) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)


            case _:
                print("Error")

    # Actualizacion Registro a Optimizar
    # =====================================================================================
    #print("Valore var_ selected: ", list_vac)
    dff.loc[:,var_select]= list_vac
    nrd = pd.get_dummies(dff, columns=self.cf,dtype=np.int64)
    nrd_m = nrd.reindex(columns=self.col_train_model, fill_value=0)
    self.fitness= self.function.evaluate(nrd_m)

    # Calculo de la ganacia y costo asociado al registro
    self.costo = self.f_cost.evaluate(dff)
    # Aqui se define el presupuesto asociado para mi LOTE.
    if self.costo < 4000000:
      self.Ganancia = (self.fitness * 2500) - self.costo
    else:
      self.Ganancia = float('-inf')

    return self.fitness, dff, nrd_m, np.round(self.Ganancia,2), np.round(self.costo,2)

  def calculateGanancia(self, reg_opt):
    self.Ganancia = (self.fitness * 2500) -(self.f_cost.evaluate(reg_opt))
    return np.round(self.Ganancia,2)

'''
  Heuristica Inicial: Hill Climbing
'''
class HC:
    def __init__(self, max_efos: int, bandwidth):
        self.max_efos = max_efos
        self.bandwidth = bandwidth

    def evolve(self,variablesOptimizar, f1,fc, df_opti, columns_train_models,cf):
        no_improvement_count = 0
        best_fitness_history = np.zeros(self.max_efos, float)
        Sol = Solution(variablesOptimizar, f1,fc,df_opti, columns_train_models,cf)
        Qs, S, S_d = Sol.initialization()
        Qsg = Sol.calculateGanancia(S)
        #print("Calidad Inicial: ",Qs)
        #print("Ganancia Inicial de la  Solucion: ",Qsg)
        best_fitness_history[0] = Qsg
        for iteration in range(1, self.max_efos):
            Sol1 = Solution(variablesOptimizar, f1,fc, S, columns_train_models,cf)
            Qr, R, R_d, Qrg, Qcc = Sol1.twekUpdate(self.bandwidth)
            #print(f"Calidad R: {Qr} y Ganacia  de R {Qrg} y Costo Asociado Solucion es : {Qcc}")
            # El presupuesto presupuesto apra la siembra es 1 millon.
            if (Qrg > Qsg):
              #print("Entra Aquiiiiiii")
              Qsg = Qrg
              S = R
              no_improvement_count = 0
            else:
              no_improvement_count += 1
              if no_improvement_count >= 50:
                best_fitness_history[iteration:] = Qsg
                break
            best_fitness_history[iteration] = Qsg

        # Retorno el historial, la mejor ganancia y la mejor solucion.
        return best_fitness_history, Qsg,S

    def __str__(self):
        result = "Hill-Climbing"
        return result


'''
Heuristica SA: Simuling Anneling
'''

class SA:
    def __init__(self, max_efos, bandwidth):
        self.max_efos = max_efos
        self.bandwidth = bandwidth

    def evolve(self,variablesOptimizar, f1, fc,  df_opti, columns_train_models,cf):
        no_improvement_count = 0
        to = 100
        best_fitness_history = np.zeros(self.max_efos, float)
        # [Sol es  inicialmente, el registro Original]
        Sol = Solution(variablesOptimizar, f1,fc,df_opti, columns_train_models,cf)
        Qs, s, S_d = Sol.initialization()
        #print("Calidad Registro Original: ", Qs)
        Qsg = Sol.calculateGanancia(S)
        #print("Ganancia del registro Original: ",  Qsg)
        best_fitness_history[0] = Qsg

        # Best  (El best Inicial el (Ganancia Inicial))
        # =============
        qbest = Qsg
        best = s.copy()
        # =============
        t= to
        for iteration in range(1, self.max_efos):
            Sol1 = Solution(variablesOptimizar, f1, fc, s, columns_train_models,cf)
            Qr, R, R_d, Qrg, Qcc = Sol1.twekUpdate(self.bandwidth)
            #print(f"Calidad R: {Qr} y Ganacia  de R {Qrg} y Costo Asociado Solucion es : {Qcc}")
            #print("Calidad R: ", Qr)
            t = t - to/(self.max_efos + 1)
            #print("Temperatura asociada: ", t)
            ale = np.random.uniform()
            prob = np.exp((Qrg-Qsg) / t) # Maximizando (Ganancias)
            #print(f"El numero aletorio es: {ale} y la probabilida  es  {prob}")
            if Qrg > Qsg or ale < prob:
                Qsg = Qrg
                s = R

            if Qsg > qbest:
                #print("Entra al bucle, remplazo")
                best = s
                qbest = Qsg
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                #print("Contador",no_improvement_count)
                if no_improvement_count >= 50:
                  best_fitness_history[iteration:] = Qsg
                  break

            best_fitness_history[iteration] = qbest

        return best_fitness_history, qbest,best

    def __str__(self):
        result = "SA"
        return result


'''
  Funcion Objetivo Real - Calculo de los costos a optimizar.
'''
class CostVariables():
    def __init__(self, lista_costos_unitarios, variables_optimizar):
      self.lista_costos_unitarios = lista_costos_unitarios
      self.variables_optimizar = variables_optimizar

    def evaluate(self, reg_opt):
      cost_reg_opt = reg_opt[self.variables_optimizar] *(self.lista_costos_unitarios)
      a = np.round(cost_reg_opt.values[0].sum(),2)
      return a


'''
 Definición de clases algortimo de Lobo Gris (GWO)
'''

class SolutionWolf():
  def __init__(self, variablesOptimizar, f,f_cost, reg_opt, col_train_model,cf,pop_size, presupuesto, precio_venta):
    self.variablesOptimizar = variablesOptimizar
    self.cf = cf
    self.function = f
    self.f_cost = f_cost
    self.fitness = 0
    self.reg_opt = reg_opt
    self.col_train_model = col_train_model
    self.pop_size = pop_size
    self.presupuesto= presupuesto
    self.precio_venta = precio_venta



  def initialization(self):
    df_opt= self.reg_opt.copy()
    df_opt_dm = pd.get_dummies(df_opt, columns=self.cf,dtype=np.int64)
    df_opt_dm_m = df_opt_dm.reindex(columns=self.col_train_model, fill_value=0)
    self.fitness= self.function.evaluate(df_opt_dm_m)

    self.costo = self.f_cost.evaluate(df_opt)
    # Aqui se define el presupuesto asociado para mi LOTE.
    if self.costo < self.presupuesto:
      self.Ganancia = (self.fitness * self.precio_venta) - self.costo
    else:
      self.Ganancia = float('-inf')
    return self.fitness, df_opt, df_opt_dm_m,np.round(self.Ganancia,2), np.round(self.costo,2)


  def generateNewWolf(self, reg_opt):
      df_opt= reg_opt.copy()
      # Rangos Varaibles Categoricas.
      #===============================================================
      TIPO_SIEMBRA = ["Manual","Mecanizado"]
      SEM_TRATADAS =[1,0]
      MATERIAL_GENETICO = ['Otro', 'P3966 (Pioneer)', 'P4082 (Pioneer)', 'DK 234', 'PAC 105',
            'NK254 (Syngenta)', 'PIONEER 30F35', 'PIONEER 30F35 HRR',
            'DK 234 YGRR', 'Impacto (Syngenta)', 'ICA V 305',
            'PIONEER 30F35 H', 'ADV 9339 (Syngenta)', 'DK7088',
            'PIONEER 30F32', 'CORPOICA V 114', 'Sinko (Syngenta)',
            'ADV 9293 (Syngenta)', 'DK 1596', 'Status (Syngenta)',
            'Cerato (Syngenta)', 'ICA V 156', 'ICA V 109', 'PIONEER 30F32HW',
            'FNC 3056', 'FNC 114', 'DK 1040']
      METODO_COSECHA = ['Manual', 'Mecanizada']
      #===============================================================
      # Actualizacion Registro a Optimizar.
      df_opt.loc[:,self.variablesOptimizar]= [#np.random.choice(TIPO_SIEMBRA),
                                              #np.random.choice(SEM_TRATADAS),
                                              #np.random.choice(MATERIAL_GENETICO),
                                              #np.random.randint(0,1),
                                              #np.random.choice(METODO_COSECHA),
                                              #np.random.randint(0,1),
                                              np.random.randint(0,4),
                                              np.random.randint(0,2),
                                              np.random.randint(0,2),
                                              np.random.randint(0,2),     # Final 3 Linea
                                              np.random.randint(0,2),
                                              np.random.randint(0,3),
                                              np.random.randint(0,4),
                                              np.random.randint(0,5),
                                              np.random.randint(0,3),
                                              np.random.randint(0,2),
                                              np.random.randint(0,3),
                                              np.random.randint(0,10),
                                              np.random.randint(0,3),
                                              round(np.random.uniform(0,115),2),
                                              round(np.random.uniform(0,82.5),2),
                                              round(np.random.uniform(0,276),2),
                                              np.random.randint(0,25),
                                              round(np.random.uniform(0,40),2),
                                              round(np.random.uniform(0,47.5),2),
                                              np.random.randint(0,31),
                                              round(np.random.uniform(0,75),2),
                                              round(np.random.uniform(0,180),2)]
                                              #np.random.randint(0,1),
                                              #np.random.randint(0,3),
                                              #np.random.randint(0,3),
                                              #np.random.randint(0,7)]

      #Proceso Dumificacion
      nrd = pd.get_dummies(df_opt, columns=self.cf,dtype=np.int64)
      nrd = nrd.reindex(columns=self.col_train_model, fill_value=0)
      self.fitness= self.function.evaluate(nrd)

      # Aqui incluimos el tema de la Ganancia.
      # Calculo de la ganacia y costo asociado al registro
      self.costo = self.f_cost.evaluate(df_opt)
      # Aqui se define el presupuesto asociado para mi LOTE.
      if self.costo < self.presupuesto:
        self.Ganancia = (self.fitness * self.precio_venta) - self.costo
      else:
        self.Ganancia = float('-inf')

      return self.fitness, df_opt, nrd, np.round(self.Ganancia,2), np.round(self.costo,2)

  def generatePopulationWolfs(self,reg_opt):
    # Generamos las Poblacion de lobos
    lista_population = []
    for i in range(self.pop_size):
      fit,d,d_d,g,c= self.generateNewWolf(reg_opt)             # C/d elemento de la población esta conformado por una ganancia y registro
      lista_population.append((g,d))

    return lista_population

  def changeCategoricalFeatures(self,lista_var_cat,dff):
    list_vac = []
    for variable in lista_var_cat:
        match variable:
            case 'TIPO_SIEMBRA':
              if dff[variable].values[0] == "Manual":
                  list_vac.append("Mecanizado")
              else:
                  list_vac.append("Manual")

            case 'SEM_TRATADAS':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'MATERIAL_GENETICO':
              MG =  ['Otro', 'P3966 (Pioneer)', 'P4082 (Pioneer)', 'DK 234', 'PAC 105',
              'NK254 (Syngenta)', 'PIONEER 30F35', 'PIONEER 30F35 HRR',
              'DK 234 YGRR', 'Impacto (Syngenta)', 'ICA V 305',
              'PIONEER 30F35 H', 'ADV 9339 (Syngenta)', 'DK7088',
              'PIONEER 30F32', 'CORPOICA V 114', 'Sinko (Syngenta)',
              'ADV 9293 (Syngenta)', 'DK 1596', 'Status (Syngenta)',
              'Cerato (Syngenta)', 'ICA V 156', 'ICA V 109', 'PIONEER 30F32HW',
              'FNC 3056', 'FNC 114', 'DK 1040']
              filter_MG = [v for v in MG if v != dff[variable].values[0]]
              MG_Selected = np.random.choice(filter_MG)
              list_vac.append(MG_Selected)

            case 'DRENAJE':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'METODO_COSECHA':
              if dff[variable].values[0] == "Manual":
                  list_vac.append("Mecanizada")
              else:
                  list_vac.append("Manual")

            case 'ALMACENAMIENTO_FINCA':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'ContEnfQui_Emer_Flor':
              r = np.random.choice([x for x in range(4) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContEnfQui_Flor_Cose':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'ContMalMec_Siem_Emer':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'ContMalMec_Emer_Flor':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'ContMalMec_Flor_Cose':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'ContMalQui_Antes_Siem':
              r = np.random.choice([x for x in range(3) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContMalQui_Siem_Emer':
              r = np.random.choice([x for x in range(4) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContMalQui_Emer_Flor':
              r = np.random.choice([x for x in range(5) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContMalQui_Flor_Cose':
              r = np.random.choice([x for x in range(2) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContPlaQui_Antes_Siem':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'ContPlaQui_Siem_Emer':
              r = np.random.choice([x for x in range(3) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContPlaQui_Emer_Flor':
              r = np.random.choice([x for x in range(10) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'ContPlaQui_Flor_Cose':
              r = np.random.choice([x for x in range(3) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'FerOrg_Emer_Flor':
              list_vac.append(abs(dff[variable].values[0]-1))

            case 'FerQui_Antes_Siem':
              r = np.random.choice([x for x in range(4) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'FerQui_Siem_Emer':
              r = np.random.choice([x for x in range(4) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)

            case 'FerQui_Emer_Flor':
              r = np.random.choice([x for x in range(8) if x != dff[variable].values[0]] , 1)[0]
              list_vac.append(r)


            case _:
                print("Error")

    # Actualizacion Registro a Optimizar
    # =====================================================================================
    #print("Valores Remplazar: ", list_vac)
    return list_vac


class GWO():
   def __init__(self, pop_size, max_efos):
        self.pop_size = pop_size
        self.max_efos = max_efos

   def envolve(self,variablesOptimizar, f1,fc, df_opti, columns_train_models,cf,pop_size,presupuesto,precio_venta):
    no_improvement_count = 0
    best_fitness_history = np.zeros(self.max_efos, float)
    # Generamos las poblacion de lobos
    # Pop (Lista de Tuplas [fitnes, df])
    # ====================================================================
    Sol = SolutionWolf(variablesOptimizar, f1, fc, df_opti, columns_train_models,cf,pop_size,presupuesto,precio_venta)
    pop = Sol.generatePopulationWolfs(df_opti)

    # Ordenamos la poblacion de acuerdo al Fitness
    # ===================================================================
    pop_ord = sorted(pop, key=lambda x: x[0], reverse=True)

    for k in range(self.max_efos):

      # Definicion de la jerarquia de Lobos (Alfa, Beta, Delta)
      #=================================================================
      alpha_fitnes, beta_fitnes, delta_fitnes = pop_ord[0][0], pop_ord[1][0], pop_ord[2][0]
      alpha_pos, beta_pos, delta_pos = pop_ord[0][1], pop_ord[1][1], pop_ord[2][1]
      #print(f"Los fitnes asociados son alpha: {alpha_fitnes}, beta: {beta_fitnes}, delta: {delta_fitnes}")

      # Proceso de casa y busqueda de Lobos
      # =================================================================
      a = 2*(1 - k/EFOS)
      A1, A2, A3 = a * (2 * np.random.random() - 1), a * (2 * np.random.random() - 1), a * (2 * np.random.random() - 1)
      C1, C2, C3 = 2 * np.random.random(), 2*np.random.random(), 2*np.random.random()

      # Definicion de variables continuas y categoricas
      #=====================================================================
      varContinuas = ['TotN_Antes_Siem','TotN_Siem_Emer','TotN_Emer_Flor','TotP_Antes_Siem','TotP_Siem_Emer','TotP_Emer_Flor','TotK_Antes_Siem','TotK_Siem_Emer','TotK_Emer_Flor']
      '''varCategoricas = ['TIPO_SIEMBRA','SEM_TRATADAS','MATERIAL_GENETICO','DRENAJE','METODO_COSECHA','ALMACENAMIENTO_FINCA',
                      'ContEnfQui_Emer_Flor','ContEnfQui_Flor_Cose','ContMalMec_Siem_Emer','ContMalMec_Emer_Flor',
                      'ContMalMec_Flor_Cose','ContMalQui_Antes_Siem','ContMalQui_Siem_Emer','ContMalQui_Emer_Flor',
                      'ContMalQui_Flor_Cose','ContPlaQui_Antes_Siem','ContPlaQui_Siem_Emer','ContPlaQui_Emer_Flor',
                      'ContPlaQui_Flor_Cose','FerOrg_Emer_Flor', 'FerQui_Antes_Siem','FerQui_Siem_Emer','FerQui_Emer_Flor'] '''
      varCategoricas = ['ContEnfQui_Emer_Flor','ContEnfQui_Flor_Cose','ContMalMec_Siem_Emer','ContMalMec_Emer_Flor','ContMalMec_Flor_Cose',
                        'ContMalQui_Antes_Siem','ContMalQui_Siem_Emer','ContMalQui_Emer_Flor','ContMalQui_Flor_Cose','ContPlaQui_Antes_Siem',
                        'ContPlaQui_Siem_Emer','ContPlaQui_Emer_Flor','ContPlaQui_Flor_Cose']

      #Limite | Tupas con los valore minimos y maximos
      # ========================================================================================
      limites_continuas = [(0,105),(0,82.5),(0,276),(0,24),(0,40),(0,47.5),(0,30),(0,75),(0,180)]
      for z in range(self.pop_size):
        #print(f"Elemento Poblacion {z}")
        X1 = [0 for i in range(len(varContinuas))]
        X2 = [0 for i in range(len(varContinuas))]
        X3 = [0 for i in range(len(varContinuas))]
        new_values_continious = [0 for i in range(len(varContinuas))]

        for i in range(len(varContinuas)):
          X1[i] = alpha_pos[varContinuas[i]].values[0] - A1 * abs(C1 - alpha_pos[varContinuas[i]].values[0] -pop_ord[z][1][varContinuas[i]].values[0])
          X2[i] = beta_pos[varContinuas[i]].values[0] - A2 * abs(C2 - beta_pos[varContinuas[i]].values[0] -pop_ord[z][1][varContinuas[i]].values[0])
          X3[i] = delta_pos[varContinuas[i]].values[0] - A3 * abs(C3 - delta_pos[varContinuas[i]].values[0] -pop_ord[z][1][varContinuas[i]].values[0])
          x_nuevo = (X1[i] + X2[i] + X3[i])/3

          #print("Xnuevo: ",x_nuevo)
          new_values_continious[i] = round(np.clip(x_nuevo,limites_continuas[i][0],limites_continuas[i][1]),2)


        #print(f"Elemento  de la poblacion {z}: ", new_values_continious)

        dff_mut = df_opti.copy()
        dff_mut.loc[:,varContinuas]= new_values_continious

        # Proceso con las variables categoricas (Seleccion Aleatoria del rango de Posibildiades)
        # =====================================================================================

        #var_cat = selectionCategoricalVariables(var_cat_select,dff_mut)
        var_cat= Sol.changeCategoricalFeatures(varCategoricas,dff_mut)
        #print("variables categoricas cambiadas", var_cat)
        dff_mut.loc[:,varCategoricas]= var_cat


        # Calculo de la funcion Objetivo (Fitenes)
        #print(dff_mut[variablesOptimizar])
        a1,a2,a3,a4,a5 = SolutionWolf(variablesOptimizar, f1, fc, dff_mut,columns_train_models,cf,pop_size,presupuesto,precio_venta).initialization()
        #print("Fitenes  Nueva Solucion: ", a1)
        #print("Fitenes (Ganancia) Nueva Solucion: ", a4)
        #print("Costo Nueva Solucion: ", a5)


        if a4 > pop_ord[z][0]:
          #print("Mejor solucion Actualizada")
          pop_ord[z] = (a4,dff_mut)


          #print(pop_ord[z])


      # Ordeno la Poblacion de los lobos para la proxima iteracion
      # =========================================================================================
      pop_ord = sorted(pop_ord, key=lambda x: x[0], reverse=True)
      best_fitness_history[k] = alpha_fitnes

      #print(f"==================== Iteracion {k} tiene un Alfa Fitenes de {alpha_fitnes}")


      if k > 0 and alpha_fitnes == best_fitness_history[k-1]:
        no_improvement_count += 1
        if no_improvement_count >= 50:
          best_fitness_history[k:] = alpha_fitnes
          break
      else:
        no_improvement_count = 0



    return best_fitness_history, alpha_fitnes,alpha_pos

   def __str__(self):
     result = "GWO"
     return result

EFOS=1000
def optimizar_por_id_lote(id_lote, precioVenta, presupuesto, costos_unitarios):
    #1. Cargue de los conjuntos de datos
    #===============================================================================================================
    #df_original = pd.read_csv("DatasetFinal.csv")
    #df =  df_original.copy()
    queryset = CultivoMaiz.objects.all()
    df = pd.DataFrame(list(queryset.values()))

    df_target = df.RDT_AJUSTADO
    df_features = df.drop(["RDT_AJUSTADO", "ID_LOTE"], axis=1)

    df_opti = df[df['ID_LOTE'] == int(id_lote)]
    #print("ID_LOTE.....................................................", id_lote)
    df_opti.drop(["RDT_AJUSTADO", "ID_LOTE"], axis=1)
    #print("Longitud Dataset Total: ", df.shape)
    df.head(5)

    #2. Codificación de variables categoricas - Dummy (variables Binarias  0-1)
    # =====================================================================================================================
    bin_features =['SEM_TRATADAS','DRENAJE','ALMACENAMIENTO_FINCA','CAP_ENDURE_RASTA','MOTEADOS_RASTA','MOTEADOS_MAS70cm_RASTA',
               'OBSERVA_EROSION_RASTA','OBSERVA_MOHO_RASTA','OBSERVA_RAICES_VIVAS_RASTA','OBSERVA_HOJARASCA_MO_RASTA',
                'SUELO_NEGRO_BLANDO_RASTA','CUCHILLO_PRIMER_HTE_RASTA','CERCA_RIOS_QUEBRADAS_RASTA',
               ]


    df_features[bin_features] = df_features[bin_features].replace({'SI': 1, 'NO': 0})
    df_opti[bin_features] = df_opti[bin_features].replace({'SI': 1, 'NO': 0})

    #print(df_features)

    # Artefactos (Algortimo CLR - Best Model Logrado en entrenamiento)
    # ===============================================================================
    #modelo_clasificacion = joblib.load('model_clasf.pkl')
    #modelo_reg1 = joblib.load('modelo_0.pkl')
    #modelo_reg2 = joblib.load('modelo_1.pkl')
    #lista_models= [modelo_reg1,modelo_reg2]
    script_dir = os.path.dirname(__file__)
    model_clasf_path = os.path.join(script_dir, 'model_clasf.pkl')
    model_0_path = os.path.join(script_dir, 'modelo_0.pkl')
    model_1_path = os.path.join(script_dir, 'modelo_1.pkl')

    modelo_clasificacion = joblib.load(model_clasf_path)
    modelo_reg1 = joblib.load(model_0_path)
    modelo_reg2 = joblib.load(model_1_path)
    lista_models= [modelo_reg1,modelo_reg2]

    #============== Variables a Optimizar para lograr el mayor rendimiento=============
    # Variables Fijas: Clima, Suelo, Creciemineto Plantas.
    # Variable Mejorar: Practicas de Manejo.
    # ================================================================================
    '''variablesOptimizar=['TIPO_SIEMBRA','SEM_TRATADAS','MATERIAL_GENETICO','DRENAJE','METODO_COSECHA','ALMACENAMIENTO_FINCA',
                    'ContEnfQui_Emer_Flor','ContEnfQui_Flor_Cose','ContMalMec_Siem_Emer','ContMalMec_Emer_Flor',
                    'ContMalMec_Flor_Cose','ContMalQui_Antes_Siem','ContMalQui_Siem_Emer','ContMalQui_Emer_Flor',
                    'ContMalQui_Flor_Cose','ContPlaQui_Antes_Siem','ContPlaQui_Siem_Emer','ContPlaQui_Emer_Flor',
                    'ContPlaQui_Flor_Cose','TotN_Antes_Siem','TotN_Siem_Emer','TotN_Emer_Flor','TotP_Antes_Siem',
                    'TotP_Siem_Emer','TotP_Emer_Flor','TotK_Antes_Siem','TotK_Siem_Emer','TotK_Emer_Flor','FerOrg_Emer_Flor',
                    'FerQui_Antes_Siem','FerQui_Siem_Emer','FerQui_Emer_Flor']'''

    variablesOptimizar=['ContEnfQui_Emer_Flor','ContEnfQui_Flor_Cose','ContMalMec_Siem_Emer','ContMalMec_Emer_Flor',
                        'ContMalMec_Flor_Cose','ContMalQui_Antes_Siem','ContMalQui_Siem_Emer','ContMalQui_Emer_Flor',
                        'ContMalQui_Flor_Cose','ContPlaQui_Antes_Siem','ContPlaQui_Siem_Emer','ContPlaQui_Emer_Flor',
                        'ContPlaQui_Flor_Cose','TotN_Antes_Siem','TotN_Siem_Emer','TotN_Emer_Flor','TotP_Antes_Siem',
                        'TotP_Siem_Emer','TotP_Emer_Flor','TotK_Antes_Siem','TotK_Siem_Emer','TotK_Emer_Flor']

    #print("Longitud variables Optimizar: ", len(variablesOptimizar))
    #CostosUnitariosVarOpimizar = [35000,35000, 400000,400000,
    #                              400000,64000,64000, 35000,
    #                              64000, 80000,130000,130000,
    #                              130000,3200,3200,3200,4000,
    #                              4000, 4000,3000,3000,3000]

    #CostosUnitariosVarOpimizar = [
    #    int(costos_unitarios.get(var, 0)) for var in variablesOptimizar
    #]

    print("Costos unitarios:", costos_unitarios)

    CostosUnitariosVarOpimizar = []
    for var in variablesOptimizar:
        valor_original = costos_unitarios.get(var, 0)
        try:
            valor_convertido = int(float(str(valor_original).replace(',', '.')))
            print(f"Variable: {var}, Valor original: {valor_original}, Convertido: {valor_convertido}")
            CostosUnitariosVarOpimizar.append(valor_convertido)
        except Exception as e:
            print(f"Error con variable {var}, valor {valor_original}: {e}")
            CostosUnitariosVarOpimizar.append(0)

    print("Lista final:", CostosUnitariosVarOpimizar)


    #print("Longitud costos Unitarios: ", len(CostosUnitariosVarOpimizar))
    # Variable categoricas Dumificar dentro DF
    # ==================================================================
    cf = ['TIPO_SIEMBRA','MATERIAL_GENETICO','CULT_ANT','METODO_COSECHA','TERRENO_CIRCUN_RASTA','POSICION_PERFIL_RASTA','PEDREG_PERFIL_ROCAS',
    'ESTRUCTURA_RASTA','OBSERVA_COSTRAS_DURAS_RASTA','SITIO_EXPUESTO_SOL_RASTA','OBSERVA_COSTRAS_BLANCAS_RASTA','OBSERVA_COSTAS_NEGRAS_RASTA',
    'REGION_SECA_ARIDA_RASTA','OBSERVA_PLANTAS_PEQUENAS_RASTA','RECUBRIMIENTO_VEGETAL_SUELO_RASTA','d_interno','drenaje_externo']

    # Columnas (Esquema) Utilizado por los modelos para lograr Predicción
    # ===================================================================
    #columns_train_models = pd.read_csv("columns_train_model.csv")['0'].values
    columns_train_path = os.path.join(script_dir, 'columns_train_model.csv')
    columns_train_models = pd.read_csv(columns_train_path)['0'].values

    # Registro Optimizar
    # ===============================================================
    #df_opti =df_features.loc[80:80,:]
    #print(df_opti)

    #print(df_opti[variablesOptimizar])

    # Variables Generales - Algortimo GWO
    # ================================================================================================================
    np.random.seed(6)
    PopZize=20
    #Presupuesto = 4000000
    #precioVenta = 2500
    f3 = CLR(modelo_clasificacion, lista_models)
    fc3 = CostVariables(CostosUnitariosVarOpimizar, variablesOptimizar)
    # Instanaciamos las clase lobo
    gwo = GWO(PopZize,EFOS)
    best_fitness_history_grey_wolfs, fitness_alfa,alpha_pos = gwo.envolve(variablesOptimizar,f3,fc3,df_opti,columns_train_models,cf,PopZize, presupuesto, precioVenta)

    #print("fitness alpha (ganancia): ", fitness_alfa)

    #  Mejor solucion
    #  La idea es exportar ese DF con esas variables a Optimizar (Archivo.CSV)
    # ==================================================================================
    mejor_solucion = alpha_pos[variablesOptimizar]
    #print("mejor solucion: ", mejor_solucion)
    
    gwo1 = SolutionWolf(variablesOptimizar, f3,fc3, alpha_pos, columns_train_models,cf,PopZize, presupuesto, precioVenta)
    rendimiento_esperado, reg , reg_d, ganancia , costo = gwo1.initialization()
    #print("costo............................", costo)

    return f"{float(rendimiento_esperado):,.2f}", f"{float(ganancia):,.2f}", f"{float(costo):,.2f}", mejor_solucion
   
