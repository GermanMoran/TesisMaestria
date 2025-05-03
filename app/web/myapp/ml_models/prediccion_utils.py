import pandas as pd
import pickle
from constants import CONSTANTES

# Cargar el modelo completo al iniciar el módulo (para evitar cargarlo en cada llamada)

with open(CONSTANTES.MODEL_PATH, 'rb') as f:
    modelo_completo = pickle.load(f)

# Extraer los objetos guardados
scaler = modelo_completo["scaler"]
bin_features = modelo_completo["bin_features"]
model_clasf = modelo_completo["model_clasf"]
models_reg = modelo_completo["models_reg"]

def preprocesar_nuevo_dato(nuevo_dato, bin_features, scaler, columnas_entrenamiento):
    """
    Preprocesa un nuevo dato para que tenga el mismo formato que los datos de entrenamiento.
    """
    columnas_a_eliminar = ["ID_LOTE", "RDT_AJUSTADO"]
    nuevo_dato = nuevo_dato.drop(columns=[col for col in columnas_a_eliminar if col in nuevo_dato.columns], errors='ignore')

    cat_features = nuevo_dato.select_dtypes(include=["object", "category"]).columns
    bin_dataset = nuevo_dato[bin_features].replace({'SI': 1, 'NO': 0})
    categorical_features = [x for x in cat_features if x not in bin_features]
    df_cat = pd.get_dummies(nuevo_dato[categorical_features])

    nuevo_dato.drop(cat_features, axis=1, inplace=True)
    nuevo_dato_final = pd.concat([nuevo_dato, df_cat, bin_dataset], axis=1)

    nuevo_dato_final = nuevo_dato_final.reindex(columns=columnas_entrenamiento, fill_value=0)
    nuevo_dato_final = pd.DataFrame(scaler.transform(nuevo_dato_final), columns=columnas_entrenamiento)

    return nuevo_dato_final

def predictionYield(x_dataset_test, lista_modelos, cluster_asignado):
    """
    Realiza la predicción del rendimiento utilizando el modelo adecuado según el cluster asignado.
    """
    y_pred_list = []
    for z in range(len(x_dataset_test)):
        y_pred = lista_modelos[cluster_asignado[z]].predict(x_dataset_test.values[z].reshape(1, -1))
        y_pred_list.append(y_pred[0])
    return y_pred_list

def obtener_prediccion():
    """
    Carga el archivo nuevo_dato.csv, lo preprocesa y devuelve la predicción junto con el valor real.
    """
    df_nuevo_dato = pd.read_csv(CONSTANTES.DATASET_PATH).iloc[0:1]  # Tomamos solo la primera fila
    columnas_entrenamiento = scaler.feature_names_in_

    x_sample = preprocesar_nuevo_dato(df_nuevo_dato, bin_features, scaler, columnas_entrenamiento)

    x_sample = x_sample.reindex(columns=model_clasf.feature_names_in_, fill_value=0)
    grupo_sample = model_clasf.predict(x_sample)

    y_pred_sample = predictionYield(x_sample, models_reg, grupo_sample)

    prediccion_valor = y_pred_sample[0]
    valor_real = df_nuevo_dato["RDT_AJUSTADO"].iloc[0] if "RDT_AJUSTADO" in df_nuevo_dato.columns else "No disponible"

    return prediccion_valor, valor_real

def preprocesar_nuevo_dato_formulario(nuevo_dato, bin_features, scaler):
    """
    Preprocesa un nuevo dato para que tenga el mismo formato que los datos de entrenamiento.
    """
    # 1. Eliminar columnas innecesarias
    nuevo_dato = nuevo_dato.drop(columns="ID_LOTE")
    nuevo_dato = nuevo_dato.drop(columns="RDT_AJUSTADO")

    # 2. Codificación One-Hot para variables categóricas
    cat_features = nuevo_dato.select_dtypes(include=["object", "category"]).columns
    bin_dataset = nuevo_dato[bin_features].replace({'SI': 1, 'NO': 0})
    categorical_features = [x for x in cat_features if x not in bin_features]
    df_cat = pd.get_dummies(nuevo_dato[categorical_features])

    # 3. Unir el dataset codificado con las variables binarias
    nuevo_dato.drop(cat_features, axis=1, inplace=True)
    nuevo_dato_final = pd.concat([nuevo_dato, df_cat, bin_dataset], axis=1)
    
    # 4. Obtener las columnas de entrenamiento directamente del scaler
    columnas_entrenamiento = scaler.feature_names_in_

    # 5. Asegurar que las columnas coincidan exactamente con las del entrenamiento
    nuevo_dato_final = nuevo_dato_final.reindex(columns=columnas_entrenamiento, fill_value=0)

    # 6. Aplicar normalización
    nuevo_dato_final = pd.DataFrame(scaler.transform(nuevo_dato_final), columns=columnas_entrenamiento)

    return nuevo_dato_final


def obtener_prediccion_formulario(datos_usuario=None):
    """
    Si recibe datos_usuario, los usa para hacer la predicción.
    Si no recibe datos_usuario, carga la primera fila de nuevo_dato.csv.
    """

    if datos_usuario is None:
        datos_usuario = pd.read_csv(CONSTANTES.DATASET_PATH).iloc[0:1]  # Tomamos solo la primera fila
    
    x_sample = preprocesar_nuevo_dato_formulario(datos_usuario, bin_features, scaler)

    x_sample = x_sample.reindex(columns=model_clasf.feature_names_in_, fill_value=0)
    grupo_sample = model_clasf.predict(x_sample)

    y_pred_sample = predictionYield(x_sample, models_reg, grupo_sample)

    prediccion_valor = y_pred_sample[0]
    valor_real = datos_usuario["RDT_AJUSTADO"].iloc[0] if "RDT_AJUSTADO" in datos_usuario.columns else "No disponible"

    return prediccion_valor, valor_real