class Constants:
    DATASET_PATH = "myapp/ml_models/nuevo_dato.csv"
    MODEL_PATH = "myapp/ml_models/modelo_completo.pkl"
    ROW_DATA = 0
    # variables generales del cultivo---------------------------------------------
    SI_NO_CHOICES = [("SI", "SI"), ("NO", "NO")]
    SI_CHOICES = "SI"
    NO_CHOICES = "NO"
    ID_LOTE_LABEL = "ID Lote"
    TIPO_SIEMBRA_LABEL = "Tipo de siembra"
    TIPO_SIEMBRA_CHOICES = [("Manual", "Manual"), ("Mecanizado", "Mecanizado")]
    SEM_TRATADAS_LABEL = "Semillas tratadas"
    MATERIAL_GENETICO_LABEL = "Material genético"
    MATERIAL_GENETICO_CHOICES = [
        ("ADV 9293 (Syngenta)", "ADV 9293 (Syngenta)"),
        ("ADV 9339 (Syngenta)", "ADV 9339 (Syngenta)"),
        ("Cerato (Syngenta)", "Cerato (Syngenta)"),
        ("CORPOICA V 114", "CORPOICA V 114"),
        ("DK 1040", "DK 1040"),
        ("DK 1596", "DK 1596"),
        ("DK 234", "DK 234"),
        ("DK 234 YGRR", "DK 234 YGRR"),
        ("DK7088", "DK7088"),
        ("FNC 114", "FNC 114"),
        ("FNC 3056", "FNC 3056"),
        ("ICA V 109", "ICA V 109"),
        ("ICA V 156", "ICA V 156"),
        ("ICA V 305", "ICA V 305"),
        ("Impacto (Syngenta)", "Impacto (Syngenta)"),
        ("NK254 (Syngenta)", "NK254 (Syngenta)"),
        ("Otro", "Otro"),
        ("P3966 (Pioneer)", "P3966 (Pioneer)"),
        ("P4082 (Pioneer)", "P4082 (Pioneer)"),
        ("PAC 105", "PAC 105"),
        ("PIONEER 30F32", "PIONEER 30F32"),
        ("PIONEER 30F32HW", "PIONEER 30F32HW"),
        ("PIONEER 30F35", "PIONEER 30F35"),
        ("PIONEER 30F35 H", "PIONEER 30F35 H"),
        ("PIONEER 30F35 HRR", "PIONEER 30F35 HRR"),
        ("Sinko (Syngenta)", "Sinko (Syngenta)"),
    ]
    CULT_ANT_LABEL = "Cultivo anterior"
    CULT_ANT_CHOICES = [
        ("Algodon", "Algodon"),
        ("Maiz", "Maiz"),
        ("Pastos", "Pastos"),
        ("Frijol", "Frijol"),
        ("Yuca", "Yuca"),
    ]
    DRENAJE_LABEL = "Drenaje"
    METODO_COSECHA_LABEL = "Método de cosecha"
    METODO_COSECHA_CHOICES=[("Manual", "Manual"), ("Mecanizada", "Mecanizada")]
    ALMACENAMIENTO_FINCA_LABEL="Almacenamiento en finca" 
    DIAS_EN_EMERGER_LABEL="Dias en emerger" 
    DIAS_EN_EMERGER_A_FLORECER_LABEL="Días en emerger a florecer"
    DIAS_EN_FLORECER_A_COSECHAR_LABEL="Dias en florecer a cosechar"
    POBLACION_20DIAS_AJT_LABEL="Población de plantas 20 días después de la siembra"
    ALTURA_LOT_LABEL="Altura del lote (m)"
    #variables de manejo del cultivo-------------------------------------------------
    CONT_ENF_QUI_EMER_FLOR_LABEL = "Numero de controles enfermedades mediante quimicos etapas de Emergencia- Floracion"
    CONT_ENF_QUI_FLOR_COSE_LABEL = "Numero de controles enfermedades mediante quimicos etapas de Floración a Cosecha"
    CONT_MAL_MEC_SIEM_EMER_LABEL = "Numero de controles de malezas mediante herramientas mecanizada etapas de Siembra a Emergencia"
    CONT_MAL_MEC_EMER_FLOR_LABEL = "Numero de controles de malezas mediante herramientas mecanizada etapas de Emergencia a Floración"
    CONT_MAL_MEC_FLOR_COSE_LABEL = "Numero de controles de malezas mediante herramientas mecanizada etapas de Floración a Cosecha"
    CONT_MAL_QUI_ANTES_SIEM_LABEL = "Numero de controles de malezas mediante Quimicos antes de la siembra"
    CONT_MAL_QUI_SIEM_EMER_LABEL = "Numero de controles de malezas mediante Quimicos en las etapas de siembra a emergencia"
    CONT_MAL_QUI_EMER_FLOR_LABEL = "Numero de controles de malezas mediante Quimicos en las etapas de emergencia a Floración"
    CONT_MAL_QUI_FLOR_COSE_LABEL = "Numero de controles de malezas mediante Quimicos en las etapas de Floración a Cosecha"
    CONT_PLA_QUI_ANTES_SIEM_LABEL = "Numero de controles de Plagas mediante Quimicos Antes de la siembra"
    CONT_PLA_QUI_SIEM_EMER_LABEL = "Numero de controles de Plagas mediante Quimicos en las etpas de siembra a emergecia"
    CONT_PLA_QUI_EMER_FLOR_LABEL = "Numero de controles de Plagas mediante Quimicos en las etpas Emergencia a Floracion"
    CONT_PLA_QUI_FLOR_COSE_LABEL = "Numero de controles de Plagas mediante Quimicos en las etpas Floracion a Cosecha"
    TOT_N_ANTES_SIEM_LABEL = "Total de Nitrogeno antes de la siembra (kg)"
    TOT_N_SIEM_EMER_LABEL = "Total de Nitrogeno de Siembra a Emergencia(kg)"
    TOT_N_EMER_FLOR_LABEL = "Total de Nitrogeno de Emergencia a Floración (kg)"
    TOT_P_ANTES_SIEM_LABEL = "Total de Fosforo antes de la siembra (kg)"
    TOT_P_SIEM_EMER_LABEL = "Total de Fosforo de Siembra a Emergencia(kg)"
    TOT_P_EMER_FLOR_LABEL = "Total de Fosforo de Emergencia a Floración (kg)"
    TOT_K_ANTES_SIEM_LABEL = "Total de Potasio antes de la siembra (kg)"
    TOT_K_SIEM_EMER_LABEL = "Total de Potasio de Siembra a Emergencia(kg)"
    TOT_K_EMER_FLOR_LABEL = "Total de Potasio de Emergencia a Floración (kg)"
    FER_ORG_EMER_FLOR_LABEL = "Cantidad de Fertilizantes Organicos aplicados entre las etapas de Emergencia a Floración"
    FER_QUI_ANTES_SIEM_LABEL = "Cantidad de Fertilizantes Quimicos aplicados antes de la siembra"
    FER_QUI_SIEM_EMER_LABEL = "Cantidad de Fertilizantes Quimicos aplicados entre siembra y emergencia"
    FER_QUI_EMER_FLOR_LABEL = "Cantidad de Fertilizantes Quimicos aplicados entre emergencia y Floración"
        
    # -----------------------------------------------------------------------------
    PENDIENTE_RASTA_LABEL = "Pendiente Rasta"
    TERRENO_CIRCUN_RASTA_LABEL = "Nivel del Terreno"
    TERRENO_CIRCUN_RASTA_CHOICES = [
        ("ONDULADO", "ONDULADO"),
        ("ONDULADO Y MONTANIOSO", "ONDULADO Y MONTANIOSO"),
        ("PLANO O LLANO", "PLANO O LLANO"),
    ]
    POSICION_PERFIL_RASTA_LABEL = "Posicion del Terreno"
    POSICION_PERFIL_RASTA_CHOICES = [
        ("LADERA CONCAVA", "LADERA CONCAVA"),
        ("LADERA CONVEXA", "LADERA CONVEXA"),
        ("LADERA PLANA", "LADERA PLANA"),
        ("PIE DE UNA ELEVACION", "PIE DE UNA ELEVACION"),
        ("PLANO", "PLANO"),
        ("PLANO CON ONDULACIONES", "PLANO CON ONDULACIONES"),
    ]
    NO_CAPAS_RASTA_LABEL = "Numero Capas Internas Suelo"
    PH_RASTA_LABEL = "PH Suelo"
    PEDREG_PERFIL_ROCAS_LABEL = "Presencia Rocas Terreno"
    PEDREG_PERFIL_ROCAS_CHOICES = [("MUY ROCOSO", "MUY ROCOSO"), ("SIN ROCAS", "SIN ROCAS")]
    CAP_ENDURE_RASTA_LABEL = "Presencia Capas Duras Terreno"
    CAP_ENDURE_RASTA_CHOICES = [("SI", "SI"), ("NO", "NO")]
    PROFUND_CAP_ENDURE_RASTA_LABEL = "Profundidad de Capa dura en el Terreno (cm)"
    ESPESOR_CAP_ENDURE_RASTA_LABEL = "Espresor de Capas(cm)"
    MOTEADOS_RASTA_LABEL = "Presencia de Moteados"
    MOTEADOS_RASTA_CHOICES = [("SI", "SI"), ("NO", "NO")]
    PROFUND_MOTEADOS_RASTA_LABEL = "Profundidad Moteados Capa del Terreno (cm)"
    MOTEADOS_MAS70CM_RASTA_LABEL = "Profundidad de Moteados superior a 70 cm"
    MOTEADOS_MAS70CM_RASTA_CHOICES = [("SI", "SI"), ("NO", "NO")]
    ESTRUCTURA_RASTA_LABEL = "Estructura del terreno"
    ESTRUCTURA_RASTA_CHOICES = [
        ("ATERRONADA", "ATERRONADA"),
        ("GRANULAR", "GRANULAR"),
        ("MASIVA", "MASIVA"),
        ("SUELTA O POLVOSA", "SUELTA O POLVOSA"),
    ]
    OBSERVA_EROSION_RASTA_LABEL = "Se observa Eroción terreno?"
    OBSERVA_EROSION_RASTA_CHOICES = [("SI", "SI"), ("NO", "NO")]
    OBSERVA_MOHO_RASTA_LABEL = "Se observa Hongos en el terreno?"
    OBSERVA_MOHO_RASTA_CHOICES = [("SI", "SI"), ("NO", "NO")]
    OBSERVA_COSTRAS_DURAS_RASTA_LABEL = "Se Observa Costras duras en el terreno?"
    OBSERVA_COSTRAS_DURAS_RASTA_CHOICES = [("MUY MARCADAS", "MUY MARCADAS"), ("NO HAY", "NO HAY"), ("POCO MARCADAS", "POCO MARCADAS")]
    SITIO_EXPUESTO_SOL_RASTA_LABEL = "La Capa del terreno esta expuesta al sol?"
    SITIO_EXPUESTO_SOL_RASTA_CHOICES = [
        ("LA MANIANA", "LA MANIANA"),
        ("LA MANIANA Y LA TARDE", "LA MANIANA Y LA TARDE"),
        ("LA TARDE", "LA TARDE"),
    ]
    OBSERVA_COSTRAS_BLANCAS_RASTA_LABEL = "Se observa Presencia de Costras Blancas en el Terreno?"
    OBSERVA_COSTRAS_BLANCAS_RASTA_CHOICES = [("NO HAY", "NO HAY"), ("POCO MARCADAS", "POCO MARCADAS")]
    OBSERVA_COSTAS_NEGRAS_RASTA_LABEL = "Se Observa costras negras en el terreno?"
    OBSERVA_COSTAS_NEGRAS_RASTA_CHOICES = [("NO HAY", "NO HAY"), ("POCO MARCADAS", "POCO MARCADAS")]
    REGION_SECA_ARIDA_RASTA_LABEL = "La Region es seca y arida?"
    REGION_SECA_ARIDA_RASTA_CHOICES = [("SI", "SI"), ("NO", "NO")]
    OBSERVA_RAICES_VIVAS_RASTA_LABEL = "Se observa la presencia de raices vivas en el terreno?"
    OBSERVA_RAICES_VIVAS_RASTA_CHOICES = [("SI", "SI"), ("NO", "NO")]
    PROFUND_RAICES_VIVAS_RASTA_LABEL = "Profundidad de las raices vivas en el terreno (cm)"
    OBSERVA_PLANTAS_PEQUENAS_RASTA_LABEL = "Se Observa plantas pequeñas en el terreno?"
    OBSERVA_PLANTAS_PEQUENAS_RASTA_CHOICES = [
        ("MUY AFECTADAS", "MUY AFECTADAS"),
        ("NO HAY CULTIVO", "NO HAY CULTIVO"),
        ("PLANTAS NORMALES", "PLANTAS NORMALES"),
        ("POCO AFECTADAS", "POCO AFECTADAS"),
    ]
    OBSERVA_HOJARASCA_MO_RASTA_LABEL = "Se Observa Hojarasca y MOHO en el terreno?"
    OBSERVA_HOJARASCA_MO_RASTA_CHOICES = [("SI", "SI"), ("NO", "NO")]
    SUELO_NEGRO_BLANDO_RASTA_LABEL = "El suelo es negro y blando?"
    SUELO_NEGRO_BLANDO_RASTA_CHOICES = [("SI", "SI"), ("NO", "NO")]
    CUCHILLO_PRIMER_HTE_RASTA_LABEL = "Primera rastrillada con cuchillo tiene el terreno?"
    CUCHILLO_PRIMER_HTE_RASTA_CHOICES = [("SI", "SI"), ("NO", "NO")]
    CERCA_RIOS_QUEBRADAS_RASTA_LABEL = "El terreno esta cerca de rios y quebradas?"
    CERCA_RIOS_QUEBRADAS_RASTA_CHOICES = [("SI", "SI"), ("NO", "NO")]
    RECUBRIMIENTO_VEGETAL_SUELO_RASTA_LABEL = "El suelo del terreno tiene recubrimiento vegetal?"
    RECUBRIMIENTO_VEGETAL_SUELO_RASTA_CHOICES = [
        ("BUENO", "BUENO"),
        ("ESPACIADO", "ESPACIADO"),
        ("MUY BUENO", "MUY BUENO"),
        ("REGULAR", "REGULAR"),
        ("SIN COBERTURA", "SIN COBERTURA"),
    ]
    PROF_EFECTIVA_LABEL = "Profundidad efectiva (cm)"
    D_INTERNO_LABEL = "Dreanaje Interno"
    D_INTERNO_CHOICES = [("BUENO", "BUENO"), ("EXCESIVO", "EXCESIVO"), ("LENTO A MUY LENTO", "LENTO A MUY LENTO")]
    DRENAJE_EXTERNO_LABEL = "Dreanaje Externo"
    DRENAJE_EXTERNO_CHOICES = [("LENTO", "LENTO"), ("NINGUNO", "NINGUNO")]
    PORC_A_LABEL = "Porcetaje Suelo Arenoso (%)"
    PORC_AR_LABEL = "Porcetaje Suelo Arcilloso (%)"
    PORC_ARA_LABEL = "Porcentaje de suelo Arcilloso (%)"
    PORC_ARL_LABEL = "Porcentaje suelo Arcilloso Limoso (%)"
    PORC_FRL_LABEL = "Porcentaje Suelo Franco Arcilloso limoso (%)"
    PORC_L_LABEL = "Porcentaje suelo Limoso (%)"
    PORC_F_LABEL = "Porcentaje suelo Franco (%)"
    PORC_X_LABEL = "Porcentaje suelo x (%)"
    PORC_Y_LABEL = "Porcentaje suelo y (%)"
    PORC_AF_LABEL = "Porcentaje suelo Arcilloso y Franco (%)"
    PORC_BLANDO_LABEL = "Porcentaje suelo blanco (%)"
    PORC_DURO_LABEL = "Porcentaje Suelo Duro (%)"
    PORC_EXT_DURO_LABEL = "Porcentaje Suelo ExtraDuro (%)"
    PORC_FRIABLE_LABEL = "Porcentaje de suelo Friable (%)"
    PORC_FIRME_LABEL = "Porcentaje de suelo Firme (%)"
    PORC_EXT_FIRME_LABEL = "Porcentaje de suelo Extra-Firme (%)"
    PORC_PLASTICO_LABEL = "Porcentaje suelo PLASTICO (%)"
    PORC_MUY_PLASTICO_LABEL = "Porcentaje suelo muy PLASTICO (%)"
    # variables del clima ----------------------------------------------------------------------------------
    TEMP_MAX_AVG_VEG_LABEL = "Promedio maximo de temperatura en la etapa vegetativa (°c)"
    TEMP_MIN_AVG_VEG_LABEL = "Promedio minimo de temperatura en la etapa vegetativa (°c)"
    TEMP_AVG_VEG_LABEL = "Promedio de temperatura en la etapa vegetativa (°c)"
    DIURNAL_RANGE_AVG_VEG_LABEL = "Temperatura media Rango diurno en etapa vegetativa (°c)"
    SOL_ENER_ACCU_VEG_LABEL = "Energía solar acumulada en etapa vegetativa (cal-cm^43)"
    TEMP_MAX_34_FREQ_VEG_LABEL = "Frecuencia de días con temperatura máxima superior a 34◦C en etapa vegetativa (°c)"
    RAIN_ACCU_VEG_LABEL = "Precipitación acumulada en etapa vegetativa (mm)"
    RAIN_10_FREQ_VEG_LABEL = "Frecuencia de días con más de 10 mm de precipitación en la etapa vegetativa"
    RHUM_AVG_VEG_LABEL = "Humedad relativa media en etapa vegetativa (%)"
    
    TEMP_MAX_AVG_FOR_LABEL = "Temperatura máxima promedio en etapa de formación (°c)"
    TEMP_MIN_AVG_FOR_LABEL = "Temperatura minima promedio en etapa de formación (°c)"
    TEMP_AVG_FOR_LABEL = "Temperatura promedio en etapa de formación (°c)"
    DIURNAL_RANGE_AVG_FOR_LABEL = "Temperatura media Rango diurno en etapa de formación (°c)"
    SOL_ENER_ACCU_FOR_LABEL = "Energía solar acumulada en etapa de Formacion (cal-cm^43)"
    TEMP_MAX_34_FREQ_FOR_LABEL = "Frecuencia de días con temperatura máxima superior a 34◦C en etapa de Formacion (°c)"
    RAIN_ACCU_FOR_LABEL = "Precipitación acumulada en etapa de Formación (mm)"
    RAIN_10_FREQ_FOR_LABEL = "Frecuencia de días con más de 10 mm de precipitación en la etapa de Formación"
    RHUM_AVG_FOR_LABEL = "Humedad relativa media en etapa de Formacion (%)"
    
    TEMP_MAX_AVG_MAD_LABEL = "Temperatura máxima promedio en etapa de Maduración(°c)"
    TEMP_MIN_AVG_MAD_LABEL = "Temperatura minima promedio en etapa de Maduración (°c)"
    TEMP_AVG_MAD_LABEL = "Temperatura promedio en etapa de Maduración (°c)"
    DIURNAL_RANGE_AVG_MAD_LABEL = "Temperatura media Rango diurno en etapa de Maduracion (°c)"
    SOL_ENER_ACCU_MAD_LABEL = "Energía solar acumulada en etapa de Maduración (cal-cm^43)"
    TEMP_MAX_34_FREQ_MAD_LABEL = "Frecuencia de días con temperatura máxima superior a 34◦C en etapa de Maduracion (°c)"
    RAIN_ACCU_MAD_LABEL = "Precipitación acumulada en etapa de Maduración (mm)"
    RAIN_10_FREQ_MAD_LABEL = "Frecuencia de días con más de 10 mm de precipitación en la etapa de maduración"
    RHUM_AVG_MAD_LABEL = "Humedad relativa media en etapa de Maduración(%)"        
    RDT_AJUSTADO_LABEL = "Rendimiento Obtenido Ajustado (kg/ha)"        


CONSTANTES = Constants()
