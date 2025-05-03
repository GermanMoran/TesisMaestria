from django.db import models
from constants import CONSTANTES
from django.contrib.auth.models import User


class CultivoMaiz(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        default=1,
        verbose_name="Usuario"
    )

    # seccion 1 - Variables Generales del Cultivo
    ID_LOTE = models.BigIntegerField(
        verbose_name=CONSTANTES.ID_LOTE_LABEL,
        default=3734
    )
    TIPO_SIEMBRA = models.CharField(
        verbose_name=CONSTANTES.TIPO_SIEMBRA_LABEL,
        choices=CONSTANTES.TIPO_SIEMBRA_CHOICES,
        max_length=20,
        default=CONSTANTES.TIPO_SIEMBRA_CHOICES[1][1]
    )
    SEM_TRATADAS = models.CharField(
        verbose_name=CONSTANTES.SEM_TRATADAS_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.SI_CHOICES
    )
    MATERIAL_GENETICO = models.CharField(
        verbose_name=CONSTANTES.MATERIAL_GENETICO_LABEL,
        choices=CONSTANTES.MATERIAL_GENETICO_CHOICES,
        max_length=150,
        default=CONSTANTES.MATERIAL_GENETICO_CHOICES[6][1]
    )
    CULT_ANT = models.CharField(
        verbose_name=CONSTANTES.CULT_ANT_LABEL,
        choices=CONSTANTES.CULT_ANT_CHOICES,
        max_length=150,
        default=CONSTANTES.CULT_ANT_CHOICES[0][1]
    )
    DRENAJE = models.CharField(
        verbose_name=CONSTANTES.DRENAJE_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.NO_CHOICES
    )
    METODO_COSECHA = models.CharField(
        verbose_name=CONSTANTES.METODO_COSECHA_LABEL,
        choices=CONSTANTES.METODO_COSECHA_CHOICES,
        max_length=100,
        default=CONSTANTES.METODO_COSECHA_CHOICES[0][1]
    )
    ALMACENAMIENTO_FINCA = models.CharField(
        verbose_name=CONSTANTES.ALMACENAMIENTO_FINCA_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.NO_CHOICES
    )
    DIAS_EN_EMERGER = models.IntegerField(
        verbose_name=CONSTANTES.DIAS_EN_EMERGER_LABEL,
        default=5
    )
    DIAS_EN_EMERGER_A_FLORECER = models.IntegerField(
        verbose_name=CONSTANTES.DIAS_EN_EMERGER_A_FLORECER_LABEL,
        default=49
    )
    DIAS_EN_FLORECER_A_COSECHAR = models.IntegerField(
        verbose_name=CONSTANTES.DIAS_EN_FLORECER_A_COSECHAR_LABEL,
        default=84
    )
    POBLACION_20DIAS_AJT = models.IntegerField(
        verbose_name=CONSTANTES.POBLACION_20DIAS_AJT_LABEL,
        default=65000
    )
    ALTURA_LOT = models.IntegerField(
        verbose_name=CONSTANTES.ALTURA_LOT_LABEL,
        default=7
    )
    # Sección 2 - Variables de Manejo del Cultivo
    ContEnfQui_Emer_Flor = models.IntegerField(
        verbose_name=CONSTANTES.CONT_ENF_QUI_EMER_FLOR_LABEL,
        default=0
    )
    ContEnfQui_Flor_Cose = models.IntegerField(
        verbose_name=CONSTANTES.CONT_ENF_QUI_FLOR_COSE_LABEL,
        default=0
    )
    ContMalMec_Siem_Emer = models.IntegerField(
        verbose_name=CONSTANTES.CONT_MAL_MEC_SIEM_EMER_LABEL,
        default=0
    )
    ContMalMec_Emer_Flor = models.IntegerField(
        verbose_name=CONSTANTES.CONT_MAL_MEC_EMER_FLOR_LABEL,
        default=0
    )
    ContMalMec_Flor_Cose = models.IntegerField(
        verbose_name=CONSTANTES.CONT_MAL_MEC_FLOR_COSE_LABEL,
        default=0
    )
    ContMalQui_Antes_Siem = models.IntegerField(
        verbose_name=CONSTANTES.CONT_MAL_QUI_ANTES_SIEM_LABEL,
        default=0
    )
    ContMalQui_Siem_Emer = models.IntegerField(
        verbose_name=CONSTANTES.CONT_MAL_QUI_SIEM_EMER_LABEL,
        default=0
    )
    ContMalQui_Emer_Flor = models.IntegerField(
        verbose_name=CONSTANTES.CONT_MAL_QUI_EMER_FLOR_LABEL,
        default=1
    )
    ContMalQui_Flor_Cose = models.IntegerField(
        verbose_name=CONSTANTES.CONT_MAL_QUI_FLOR_COSE_LABEL,
        default=0
    )
    ContPlaQui_Antes_Siem = models.IntegerField(
        verbose_name=CONSTANTES.CONT_PLA_QUI_ANTES_SIEM_LABEL,
        default=0
    )
    ContPlaQui_Siem_Emer = models.IntegerField(
        verbose_name=CONSTANTES.CONT_PLA_QUI_SIEM_EMER_LABEL,
        default=0
    )
    ContPlaQui_Emer_Flor = models.IntegerField(
        verbose_name=CONSTANTES.CONT_PLA_QUI_EMER_FLOR_LABEL,
        default=1
    )
    ContPlaQui_Flor_Cose = models.IntegerField(
        verbose_name=CONSTANTES.CONT_PLA_QUI_FLOR_COSE_LABEL,
        default=0
    )
    TotN_Antes_Siem = models.FloatField(
        verbose_name=CONSTANTES.TOT_N_ANTES_SIEM_LABEL,
        default=0.0
    )
    TotN_Siem_Emer = models.FloatField(
        verbose_name=CONSTANTES.TOT_N_SIEM_EMER_LABEL,
        default=0.0
    )
    TotN_Emer_Flor = models.FloatField(
        verbose_name=CONSTANTES.TOT_N_EMER_FLOR_LABEL,
        default=92.0
    )
    TotP_Antes_Siem = models.IntegerField(
        verbose_name=CONSTANTES.TOT_P_ANTES_SIEM_LABEL,
        default=0
    )
    TotP_Siem_Emer = models.FloatField(
        verbose_name=CONSTANTES.TOT_P_SIEM_EMER_LABEL,
        default=0.0
    )
    TotP_Emer_Flor = models.FloatField(
        verbose_name=CONSTANTES.TOT_P_EMER_FLOR_LABEL,
        default=0.0
    )
    TotK_Antes_Siem = models.IntegerField(
        verbose_name=CONSTANTES.TOT_K_ANTES_SIEM_LABEL,
        default=0
    )
    TotK_Siem_Emer = models.FloatField(
        verbose_name=CONSTANTES.TOT_K_SIEM_EMER_LABEL,
        default=0.0
    )
    TotK_Emer_Flor = models.FloatField(
        verbose_name=CONSTANTES.TOT_K_EMER_FLOR_LABEL,
        default=0.0
    )
    FerOrg_Emer_Flor = models.IntegerField(
        verbose_name=CONSTANTES.FER_ORG_EMER_FLOR_LABEL,
        default=0
    )
    FerQui_Antes_Siem = models.IntegerField(
        verbose_name=CONSTANTES.FER_QUI_ANTES_SIEM_LABEL,
        default=0
    )
    FerQui_Siem_Emer = models.IntegerField(
        verbose_name=CONSTANTES.FER_QUI_SIEM_EMER_LABEL,
        default=0
    )
    FerQui_Emer_Flor = models.IntegerField(
        verbose_name=CONSTANTES.FER_QUI_EMER_FLOR_LABEL,
        default=2
    )
    # Sección 3 - Variables del Suelo
    PENDIENTE_RASTA = models.FloatField(
        verbose_name=CONSTANTES.PENDIENTE_RASTA_LABEL,
        default=1.0
    )
    TERRENO_CIRCUN_RASTA = models.CharField(
        verbose_name=CONSTANTES.TERRENO_CIRCUN_RASTA_LABEL,
        choices=CONSTANTES.TERRENO_CIRCUN_RASTA_CHOICES,
        max_length=150,
        default=CONSTANTES.TERRENO_CIRCUN_RASTA_CHOICES[2][1]
    )
    POSICION_PERFIL_RASTA = models.CharField(
        verbose_name=CONSTANTES.POSICION_PERFIL_RASTA_LABEL,
        choices=CONSTANTES.POSICION_PERFIL_RASTA_CHOICES,
        max_length=150,
        default=CONSTANTES.POSICION_PERFIL_RASTA_CHOICES[4][1]
    )
    NO_CAPAS_RASTA = models.IntegerField(
        verbose_name=CONSTANTES.NO_CAPAS_RASTA_LABEL,
        default=2
    )
    PH_RASTA = models.FloatField(
        verbose_name=CONSTANTES.PH_RASTA_LABEL,
        default=5.5
    )
    PEDREG_PERFIL_ROCAS = models.CharField(
        verbose_name=CONSTANTES.PEDREG_PERFIL_ROCAS_LABEL,
        choices=CONSTANTES.PEDREG_PERFIL_ROCAS_CHOICES,
        max_length=50,
        default=CONSTANTES.PEDREG_PERFIL_ROCAS_CHOICES[1][1]
    )
    CAP_ENDURE_RASTA = models.CharField(
        verbose_name=CONSTANTES.CAP_ENDURE_RASTA_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.NO_CHOICES
    )
    PROFUND_CAP_ENDURE_RASTA = models.IntegerField(
        verbose_name=CONSTANTES.PROFUND_CAP_ENDURE_RASTA_LABEL,
        default=-1
    )
    ESPESOR_CAP_ENDURE_RASTA = models.FloatField(
        verbose_name=CONSTANTES.ESPESOR_CAP_ENDURE_RASTA_LABEL,
        default=-1.0
    )
    MOTEADOS_RASTA = models.CharField(
        verbose_name=CONSTANTES.MOTEADOS_RASTA_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.SI_CHOICES
    )
    PROFUND_MOTEADOS_RASTA = models.IntegerField(
        verbose_name=CONSTANTES.PROFUND_MOTEADOS_RASTA_LABEL,
        default=20
    )
    MOTEADOS_MAS70cm_RASTA = models.CharField(
        verbose_name=CONSTANTES.MOTEADOS_MAS70CM_RASTA_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.NO_CHOICES
    )
    ESTRUCTURA_RASTA = models.CharField(
        verbose_name=CONSTANTES.ESTRUCTURA_RASTA_LABEL,
        choices=CONSTANTES.ESTRUCTURA_RASTA_CHOICES,
        max_length=50,
        default=CONSTANTES.ESTRUCTURA_RASTA_CHOICES[1][1]
    )
    OBSERVA_EROSION_RASTA = models.CharField(
        verbose_name=CONSTANTES.OBSERVA_EROSION_RASTA_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.NO_CHOICES
    )
    OBSERVA_MOHO_RASTA = models.CharField(
        verbose_name=CONSTANTES.OBSERVA_MOHO_RASTA_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.NO_CHOICES
    )
    OBSERVA_COSTRAS_DURAS_RASTA = models.CharField(
        verbose_name=CONSTANTES.OBSERVA_COSTRAS_DURAS_RASTA_LABEL,
        choices=CONSTANTES.OBSERVA_COSTRAS_DURAS_RASTA_CHOICES,
        max_length=150,
        default=CONSTANTES.OBSERVA_COSTRAS_DURAS_RASTA_CHOICES[1][1]
    )
    SITIO_EXPUESTO_SOL_RASTA = models.CharField(
        verbose_name=CONSTANTES.SITIO_EXPUESTO_SOL_RASTA_LABEL,
        choices=CONSTANTES.SITIO_EXPUESTO_SOL_RASTA_CHOICES,
        max_length=150,
        default=CONSTANTES.SITIO_EXPUESTO_SOL_RASTA_CHOICES[1][1]
    )
    OBSERVA_COSTRAS_BLANCAS_RASTA = models.CharField(
        verbose_name=CONSTANTES.OBSERVA_COSTRAS_BLANCAS_RASTA_LABEL,
        choices=CONSTANTES.OBSERVA_COSTRAS_BLANCAS_RASTA_CHOICES,
        max_length=150,
        default=CONSTANTES.OBSERVA_COSTRAS_BLANCAS_RASTA_CHOICES[0][1]
    )
    OBSERVA_COSTAS_NEGRAS_RASTA = models.CharField(
        verbose_name=CONSTANTES.OBSERVA_COSTAS_NEGRAS_RASTA_LABEL,
        choices=CONSTANTES.OBSERVA_COSTAS_NEGRAS_RASTA_CHOICES,
        max_length=150,
        default=CONSTANTES.OBSERVA_COSTAS_NEGRAS_RASTA_CHOICES[0][1]
    )
    REGION_SECA_ARIDA_RASTA = models.CharField(
        verbose_name=CONSTANTES.REGION_SECA_ARIDA_RASTA_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.NO_CHOICES
    )
    OBSERVA_RAICES_VIVAS_RASTA = models.CharField(
        verbose_name=CONSTANTES.OBSERVA_RAICES_VIVAS_RASTA_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.SI_CHOICES
    )
    PROFUND_RAICES_VIVAS_RASTA = models.IntegerField(
        verbose_name=CONSTANTES.PROFUND_RAICES_VIVAS_RASTA_LABEL,
        default=21
    )
    OBSERVA_PLANTAS_PEQUENAS_RASTA = models.CharField(
        verbose_name=CONSTANTES.OBSERVA_PLANTAS_PEQUENAS_RASTA_LABEL,
        choices=CONSTANTES.OBSERVA_PLANTAS_PEQUENAS_RASTA_CHOICES,
        max_length=150,
        default=CONSTANTES.OBSERVA_PLANTAS_PEQUENAS_RASTA_CHOICES[2][1]
    )
    OBSERVA_HOJARASCA_MO_RASTA = models.CharField(
        verbose_name=CONSTANTES.OBSERVA_HOJARASCA_MO_RASTA_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.NO_CHOICES
    )
    SUELO_NEGRO_BLANDO_RASTA = models.CharField(
        verbose_name=CONSTANTES.SUELO_NEGRO_BLANDO_RASTA_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.NO_CHOICES
    )
    CUCHILLO_PRIMER_HTE_RASTA = models.CharField(
        verbose_name=CONSTANTES.CUCHILLO_PRIMER_HTE_RASTA_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.SI_CHOICES
    )
    CERCA_RIOS_QUEBRADAS_RASTA = models.CharField(
        verbose_name=CONSTANTES.CERCA_RIOS_QUEBRADAS_RASTA_LABEL,
        choices=CONSTANTES.SI_NO_CHOICES,
        max_length=2,
        default=CONSTANTES.SI_CHOICES
    )
    RECUBRIMIENTO_VEGETAL_SUELO_RASTA = models.CharField(
        verbose_name=CONSTANTES.RECUBRIMIENTO_VEGETAL_SUELO_RASTA_LABEL,
        choices=CONSTANTES.RECUBRIMIENTO_VEGETAL_SUELO_RASTA_CHOICES,
        max_length=150,
        default=CONSTANTES.RECUBRIMIENTO_VEGETAL_SUELO_RASTA_CHOICES[3][1]
    )
    prof_efectiva = models.IntegerField(
        verbose_name=CONSTANTES.PROF_EFECTIVA_LABEL,
        default=21
    )
    d_interno = models.CharField(
        verbose_name=CONSTANTES.D_INTERNO_LABEL,
        choices=CONSTANTES.D_INTERNO_CHOICES,
        max_length=150,
        default=CONSTANTES.D_INTERNO_CHOICES[2][1]
    )
    drenaje_externo = models.CharField(
        verbose_name=CONSTANTES.DRENAJE_EXTERNO_LABEL,
        choices=CONSTANTES.DRENAJE_EXTERNO_CHOICES,
        max_length=150,
        default=CONSTANTES.DRENAJE_EXTERNO_CHOICES[1][1]
    )
    Porc_A = models.IntegerField(
        verbose_name=CONSTANTES.PORC_A_LABEL,
        default=0
    )
    Porc_Ar = models.FloatField(
        verbose_name=CONSTANTES.PORC_AR_LABEL,
        default=28.33
    )
    Porc_ArA = models.FloatField(
        verbose_name=CONSTANTES.PORC_ARA_LABEL,
        default=0.0
    )
    Porc_ArL = models.FloatField(
        verbose_name=CONSTANTES.PORC_ARL_LABEL,
        default=0.0
    )
    Porc_FrL = models.FloatField(
        verbose_name=CONSTANTES.PORC_FRL_LABEL,
        default=0.0
    )
    Porc_L = models.FloatField(
        verbose_name=CONSTANTES.PORC_L_LABEL,
        default=0.0
    )
    Porc_F = models.FloatField(
        verbose_name=CONSTANTES.PORC_F_LABEL,
        default=0.0
    )
    porc_x = models.FloatField(
        verbose_name=CONSTANTES.PORC_X_LABEL,
        default=71.67
    )
    porc_y = models.FloatField(
        verbose_name=CONSTANTES.PORC_Y_LABEL,
        default=0.0
    )
    Porc_AF = models.FloatField(
        verbose_name=CONSTANTES.PORC_AF_LABEL,
        default=0.0
    )
    Porc_BLANDO = models.FloatField(
        verbose_name=CONSTANTES.PORC_BLANDO_LABEL,
        default=100.0
    )
    Porc_DURO = models.FloatField(
        verbose_name=CONSTANTES.PORC_DURO_LABEL,
        default=0.0
    )
    Porc_EXT_DURO = models.FloatField(
        verbose_name=CONSTANTES.PORC_EXT_DURO_LABEL,
        default=0.0
    )
    Porc_FRIABLE = models.FloatField(
        verbose_name=CONSTANTES.PORC_FRIABLE_LABEL,
        default=0.0
    )
    Porc_FIRME = models.FloatField(
        verbose_name=CONSTANTES.PORC_FIRME_LABEL,
        default=0.0
    )
    Porc_EXT_FIRME = models.FloatField(
        verbose_name=CONSTANTES.PORC_EXT_FIRME_LABEL,
        default=0.0
    )
    Porc_PLASTICO = models.FloatField(
        verbose_name=CONSTANTES.PORC_PLASTICO_LABEL,
        default=0.0
    )
    Porc_MUY_PLASTICO = models.FloatField(
        verbose_name=CONSTANTES.PORC_MUY_PLASTICO_LABEL,
        default=0.0
    )
    
# Sección 4 - Variables del Clima ----------------------------------------------------------------------------------
    Temp_Max_Avg_Veg = models.FloatField(
        verbose_name=CONSTANTES.TEMP_MAX_AVG_VEG_LABEL,
        default=32.84  
    )
    Temp_Min_Avg_Veg = models.FloatField(
        verbose_name=CONSTANTES.TEMP_MIN_AVG_VEG_LABEL,
        default=23.88
    )
    Temp_Avg_Veg = models.FloatField(
        verbose_name=CONSTANTES.TEMP_AVG_VEG_LABEL,
        default=28.36
    )
    Diurnal_Range_Avg_Veg = models.FloatField(
        verbose_name=CONSTANTES.DIURNAL_RANGE_AVG_VEG_LABEL,
        default=8.96
    )
    Sol_Ener_Accu_Veg = models.FloatField(
        verbose_name=CONSTANTES.SOL_ENER_ACCU_VEG_LABEL,
        default=16278.39
    )
    Temp_Max_34_Freq_Veg = models.FloatField(
        verbose_name=CONSTANTES.TEMP_MAX_34_FREQ_VEG_LABEL,
        default=0.25
    )
    Rain_Accu_Veg = models.FloatField(
        verbose_name=CONSTANTES.RAIN_ACCU_VEG_LABEL,
        default=266.6
    )
    Rain_10_Freq_Veg = models.FloatField(
        verbose_name=CONSTANTES.RAIN_10_FREQ_VEG_LABEL,
        default=0.2
    )
    Rhum_Avg_Veg = models.FloatField(
        verbose_name=CONSTANTES.RHUM_AVG_VEG_LABEL,
        default=84.38
    )
    Temp_Max_Avg_For = models.FloatField(
        verbose_name=CONSTANTES.TEMP_MAX_AVG_FOR_LABEL,
        default=32.65
    )
    Temp_Min_Avg_For = models.FloatField(
        verbose_name=CONSTANTES.TEMP_MIN_AVG_FOR_LABEL,
        default=23.32
    )
    Temp_Avg_For = models.FloatField(
        verbose_name=CONSTANTES.TEMP_AVG_FOR_LABEL,
        default=27.99
    )
    Diurnal_Range_Avg_For = models.FloatField(
        verbose_name=CONSTANTES.DIURNAL_RANGE_AVG_FOR_LABEL,
        default=9.33
    )
    Sol_Ener_Accu_For = models.FloatField(
        verbose_name=CONSTANTES.SOL_ENER_ACCU_FOR_LABEL,
        default=23852.62
    )
    Temp_Max_34_Freq_For = models.FloatField(
        verbose_name=CONSTANTES.TEMP_MAX_34_FREQ_FOR_LABEL,
        default=0.21
    )
    Rain_Accu_For = models.FloatField(
        verbose_name=CONSTANTES.RAIN_ACCU_FOR_LABEL,
        default=293.9
    )
    Rain_10_Freq_For = models.FloatField(
        verbose_name=CONSTANTES.RAIN_10_FREQ_FOR_LABEL,
        default=0.17
    )
    Rhum_Avg_For = models.FloatField(
        verbose_name=CONSTANTES.RHUM_AVG_FOR_LABEL,
        default=83.33
    )
    Temp_Max_Avg_Mad = models.FloatField(
        verbose_name=CONSTANTES.TEMP_MAX_AVG_MAD_LABEL,
        default=31.85
    )
    Temp_Min_Avg_Mad = models.FloatField(
        verbose_name=CONSTANTES.TEMP_MIN_AVG_MAD_LABEL,
        default=23.25
    )
    Temp_Avg_Mad = models.FloatField(
        verbose_name=CONSTANTES.TEMP_AVG_MAD_LABEL,
        default=27.55
    )
    Diurnal_Range_Avg_Mad = models.FloatField(
        verbose_name=CONSTANTES.DIURNAL_RANGE_AVG_MAD_LABEL,
        default=8.6
    )
    Sol_Ener_Accu_Mad = models.FloatField(
        verbose_name=CONSTANTES.SOL_ENER_ACCU_MAD_LABEL,
        default=16594.77
    )
    Temp_Max_34_Freq_Mad = models.FloatField(
        verbose_name=CONSTANTES.TEMP_MAX_34_FREQ_MAD_LABEL,
        default=0.0
    )
    Rain_Accu_Mad = models.FloatField(
        verbose_name=CONSTANTES.RAIN_ACCU_MAD_LABEL,
        default=58.7
    )
    Rain_10_Freq_Mad = models.FloatField(
        verbose_name=CONSTANTES.RAIN_10_FREQ_MAD_LABEL,
        default=0.02
    )
    Rhum_Avg_Mad = models.FloatField(
        verbose_name=CONSTANTES.RHUM_AVG_MAD_LABEL,
        default=82.87
    )
    RDT_AJUSTADO = models.FloatField(
        verbose_name=CONSTANTES.RDT_AJUSTADO_LABEL,
        default=4576.74
    )

    def __str__(self):
        return f"Cultivo {self.ID_LOTE} - {self.MATERIAL_GENETICO}"

