import csv
import os
from django.conf import settings
from django.core.management.base import BaseCommand
from myapp.models import CultivoMaiz

class Command(BaseCommand):
    help = 'Carga los datos desde DatasetFinal.csv a la base de datos'

    def handle(self, *args, **options):
        # Construir la ruta absoluta al archivo CSV.
        csv_path = os.path.join(settings.BASE_DIR, 'myapp', 'ml_models', 'DatasetFinal.csv')

        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f"Archivo CSV no encontrado: {csv_path}"))
            return

        count = 0
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Se asume que las columnas del CSV tienen el mismo nombre que los campos del modelo.
                # Se debe convertir cada dato al tipo correspondiente.
                instance = CultivoMaiz(
                    ID_LOTE = int(row.get('ID_LOTE') or 0),
                    TIPO_SIEMBRA = row.get('TIPO_SIEMBRA'),
                    SEM_TRATADAS = row.get('SEM_TRATADAS'),
                    MATERIAL_GENETICO = row.get('MATERIAL_GENETICO'),
                    CULT_ANT = row.get('CULT_ANT'),
                    DRENAJE = row.get('DRENAJE'),
                    METODO_COSECHA = row.get('METODO_COSECHA'),
                    ALMACENAMIENTO_FINCA = row.get('ALMACENAMIENTO_FINCA'),
                    DIAS_EN_EMERGER = int(row.get('DIAS_EN_EMERGER') or 0),
                    DIAS_EN_EMERGER_A_FLORECER = int(row.get('DIAS_EN_EMERGER_A_FLORECER') or 0),
                    DIAS_EN_FLORECER_A_COSECHAR = int(row.get('DIAS_EN_FLORECER_A_COSECHAR') or 0),
                    POBLACION_20DIAS_AJT = int(row.get('POBLACION_20DIAS_AJT') or 0),
                    ALTURA_LOT = int(row.get('ALTURA_LOT') or 0),

                    ContEnfQui_Emer_Flor = int(row.get('ContEnfQui_Emer_Flor') or 0),
                    ContEnfQui_Flor_Cose = int(row.get('ContEnfQui_Flor_Cose') or 0),
                    ContMalMec_Siem_Emer = int(row.get('ContMalMec_Siem_Emer') or 0),
                    ContMalMec_Emer_Flor = int(row.get('ContMalMec_Emer_Flor') or 0),
                    ContMalMec_Flor_Cose = int(row.get('ContMalMec_Flor_Cose') or 0),
                    ContMalQui_Antes_Siem = int(row.get('ContMalQui_Antes_Siem') or 0),
                    ContMalQui_Siem_Emer = int(row.get('ContMalQui_Siem_Emer') or 0),
                    ContMalQui_Emer_Flor = int(row.get('ContMalQui_Emer_Flor') or 0),
                    ContMalQui_Flor_Cose = int(row.get('ContMalQui_Flor_Cose') or 0),
                    ContPlaQui_Antes_Siem = int(row.get('ContPlaQui_Antes_Siem') or 0),
                    ContPlaQui_Siem_Emer = int(row.get('ContPlaQui_Siem_Emer') or 0),
                    ContPlaQui_Emer_Flor = int(row.get('ContPlaQui_Emer_Flor') or 0),
                    ContPlaQui_Flor_Cose = int(row.get('ContPlaQui_Flor_Cose') or 0),
                    TotN_Antes_Siem = float(row.get('TotN_Antes_Siem') or 0),
                    TotN_Siem_Emer = float(row.get('TotN_Siem_Emer') or 0),
                    TotN_Emer_Flor = float(row.get('TotN_Emer_Flor') or 0),
                    TotP_Antes_Siem = int(row.get('TotP_Antes_Siem') or 0),
                    TotP_Siem_Emer = float(row.get('TotP_Siem_Emer') or 0),
                    TotP_Emer_Flor = float(row.get('TotP_Emer_Flor') or 0),
                    TotK_Antes_Siem = int(row.get('TotK_Antes_Siem') or 0),
                    TotK_Siem_Emer = float(row.get('TotK_Siem_Emer') or 0),
                    TotK_Emer_Flor = float(row.get('TotK_Emer_Flor') or 0),
                    FerOrg_Emer_Flor = int(row.get('FerOrg_Emer_Flor') or 0),
                    FerQui_Antes_Siem = int(row.get('FerQui_Antes_Siem') or 0),
                    FerQui_Siem_Emer = int(row.get('FerQui_Siem_Emer') or 0),
                    FerQui_Emer_Flor = int(row.get('FerQui_Emer_Flor') or 0),

                    PENDIENTE_RASTA = float(row.get('PENDIENTE_RASTA') or 0),
                    TERRENO_CIRCUN_RASTA = row.get('TERRENO_CIRCUN_RASTA'),
                    POSICION_PERFIL_RASTA = row.get('POSICION_PERFIL_RASTA'),
                    NO_CAPAS_RASTA = int(row.get('NO_CAPAS_RASTA') or 0),
                    PH_RASTA = float(row.get('PH_RASTA') or 0),
                    PEDREG_PERFIL_ROCAS = row.get('PEDREG_PERFIL_ROCAS'),
                    CAP_ENDURE_RASTA = row.get('CAP_ENDURE_RASTA'),
                    PROFUND_CAP_ENDURE_RASTA = int(row.get('PROFUND_CAP_ENDURE_RASTA') or 0),
                    ESPESOR_CAP_ENDURE_RASTA = float(row.get('ESPESOR_CAP_ENDURE_RASTA') or 0),
                    MOTEADOS_RASTA = row.get('MOTEADOS_RASTA'),
                    PROFUND_MOTEADOS_RASTA = int(row.get('PROFUND_MOTEADOS_RASTA') or 0),
                    MOTEADOS_MAS70cm_RASTA = row.get('MOTEADOS_MAS70cm_RASTA'),
                    ESTRUCTURA_RASTA = row.get('ESTRUCTURA_RASTA'),
                    OBSERVA_EROSION_RASTA = row.get('OBSERVA_EROSION_RASTA'),
                    OBSERVA_MOHO_RASTA = row.get('OBSERVA_MOHO_RASTA'),
                    OBSERVA_COSTRAS_DURAS_RASTA = row.get('OBSERVA_COSTRAS_DURAS_RASTA'),
                    SITIO_EXPUESTO_SOL_RASTA = row.get('SITIO_EXPUESTO_SOL_RASTA'),
                    OBSERVA_COSTRAS_BLANCAS_RASTA = row.get('OBSERVA_COSTRAS_BLANCAS_RASTA'),
                    OBSERVA_COSTAS_NEGRAS_RASTA = row.get('OBSERVA_COSTAS_NEGRAS_RASTA'),
                    REGION_SECA_ARIDA_RASTA = row.get('REGION_SECA_ARIDA_RASTA'),
                    OBSERVA_RAICES_VIVAS_RASTA = row.get('OBSERVA_RAICES_VIVAS_RASTA'),
                    PROFUND_RAICES_VIVAS_RASTA = int(row.get('PROFUND_RAICES_VIVAS_RASTA') or 0),
                    OBSERVA_PLANTAS_PEQUENAS_RASTA = row.get('OBSERVA_PLANTAS_PEQUENAS_RASTA'),
                    OBSERVA_HOJARASCA_MO_RASTA = row.get('OBSERVA_HOJARASCA_MO_RASTA'),
                    SUELO_NEGRO_BLANDO_RASTA = row.get('SUELO_NEGRO_BLANDO_RASTA'),
                    CUCHILLO_PRIMER_HTE_RASTA = row.get('CUCHILLO_PRIMER_HTE_RASTA'),
                    CERCA_RIOS_QUEBRADAS_RASTA = row.get('CERCA_RIOS_QUEBRADAS_RASTA'),
                    RECUBRIMIENTO_VEGETAL_SUELO_RASTA = row.get('RECUBRIMIENTO_VEGETAL_SUELO_RASTA'),
                    prof_efectiva = int(row.get('prof_efectiva') or 0),
                    d_interno = row.get('d_interno'),
                    drenaje_externo = row.get('drenaje_externo'),
                    Porc_A = int(row.get('Porc_A') or 0),
                    Porc_Ar = float(row.get('Porc_Ar') or 0),
                    Porc_ArA = float(row.get('Porc_ArA') or 0),
                    Porc_ArL = float(row.get('Porc_ArL') or 0),
                    Porc_FrL = float(row.get('Porc_FrL') or 0),
                    Porc_L = float(row.get('Porc_L') or 0),
                    Porc_F = float(row.get('Porc_F') or 0),
                    porc_x = float(row.get('porc_x') or 0),
                    porc_y = float(row.get('porc_y') or 0),
                    Porc_AF = float(row.get('Porc_AF') or 0),
                    Porc_BLANDO = float(row.get('Porc_BLANDO') or 0),
                    Porc_DURO = float(row.get('Porc_DURO') or 0),
                    Porc_EXT_DURO = float(row.get('Porc_EXT_DURO') or 0),
                    Porc_FRIABLE = float(row.get('Porc_FRIABLE') or 0),
                    Porc_FIRME = float(row.get('Porc_FIRME') or 0),
                    Porc_EXT_FIRME = float(row.get('Porc_EXT_FIRME') or 0),
                    Porc_PLASTICO = float(row.get('Porc_PLASTICO') or 0),
                    Porc_MUY_PLASTICO = float(row.get('Porc_MUY_PLASTICO') or 0),
                    Temp_Max_Avg_Veg = float(row.get('Temp_Max_Avg_Veg') or 0),
                    Temp_Min_Avg_Veg = float(row.get('Temp_Min_Avg_Veg') or 0),
                    Temp_Avg_Veg = float(row.get('Temp_Avg_Veg') or 0),
                    Diurnal_Range_Avg_Veg = float(row.get('Diurnal_Range_Avg_Veg') or 0),
                    Sol_Ener_Accu_Veg = float(row.get('Sol_Ener_Accu_Veg') or 0),
                    Temp_Max_34_Freq_Veg = float(row.get('Temp_Max_34_Freq_Veg') or 0),
                    Rain_Accu_Veg = float(row.get('Rain_Accu_Veg') or 0),
                    Rain_10_Freq_Veg = float(row.get('Rain_10_Freq_Veg') or 0),
                    Rhum_Avg_Veg = float(row.get('Rhum_Avg_Veg') or 0),
                    Temp_Max_Avg_For = float(row.get('Temp_Max_Avg_For') or 0),
                    Temp_Min_Avg_For = float(row.get('Temp_Min_Avg_For') or 0),
                    Temp_Avg_For = float(row.get('Temp_Avg_For') or 0),
                    Diurnal_Range_Avg_For = float(row.get('Diurnal_Range_Avg_For') or 0),
                    Sol_Ener_Accu_For = float(row.get('Sol_Ener_Accu_For') or 0),
                    Temp_Max_34_Freq_For = float(row.get('Temp_Max_34_Freq_For') or 0),
                    Rain_Accu_For = float(row.get('Rain_Accu_For') or 0),
                    Rain_10_Freq_For = float(row.get('Rain_10_Freq_For') or 0),
                    Rhum_Avg_For = float(row.get('Rhum_Avg_For') or 0),
                    Temp_Max_Avg_Mad = float(row.get('Temp_Max_Avg_Mad') or 0),
                    Temp_Min_Avg_Mad = float(row.get('Temp_Min_Avg_Mad') or 0),
                    Temp_Avg_Mad = float(row.get('Temp_Avg_Mad') or 0),
                    Diurnal_Range_Avg_Mad = float(row.get('Diurnal_Range_Avg_Mad') or 0),
                    Sol_Ener_Accu_Mad = float(row.get('Sol_Ener_Accu_Mad') or 0),
                    Temp_Max_34_Freq_Mad = float(row.get('Temp_Max_34_Freq_Mad') or 0),
                    Rain_Accu_Mad = float(row.get('Rain_Accu_Mad') or 0),
                    Rain_10_Freq_Mad = float(row.get('Rain_10_Freq_Mad') or 0),
                    Rhum_Avg_Mad = float(row.get('Rhum_Avg_Mad') or 0),
                    RDT_AJUSTADO = float(row.get('RDT_AJUSTADO') or 0),
                )
                instance.save()
                count += 1

        self.stdout.write(self.style.SUCCESS(f"Se han cargado {count} registros"))
