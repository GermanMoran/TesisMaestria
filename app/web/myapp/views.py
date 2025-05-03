from django.template.loader import render_to_string
from django.core.paginator import Paginator
from django.shortcuts import redirect, render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.core import serializers
from .ml_models.prediccion_utils import (
    obtener_prediccion,
    obtener_prediccion_formulario,
)
from .ml_models.sistema_recomendacion_utils import optimizar_por_id_lote
from .forms import DatosPrediccionForm
import pandas as pd
import json
from django.views.decorators.csrf import csrf_exempt
from .forms import CustomUserCreationForm
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import logout
from constants import CONSTANTES

from django.core.paginator import EmptyPage, PageNotAnInteger

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
from django.conf import settings
from myapp.models import CultivoMaiz
from plotly.offline import plot
import plotly.io as pio
import datetime


def login_view(request):
    if request.method == "POST":
        username = request.POST["loginUsername"]
        password = request.POST["loginPassword"]

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("home_view")
        else:
            return render(
                request,
                "signup.html",
                {
                    "form": CustomUserCreationForm(),
                    "error": "Usuario o contraseña incorrectos",
                },
            )
    else:
        return render(request, "signup.html", {"form": CustomUserCreationForm()})


def logout_view(request):
    logout(request)
    return redirect("signup_view")


def signup_view(request):

    if request.method == "GET":
        return render(
            request,
            "signup.html",
            {
                "form": CustomUserCreationForm(),
                "show_register": True,
            },
        )
    elif request.method == "POST":
        if request.POST["password1"] == request.POST["password2"]:
            form = CustomUserCreationForm(request.POST)
            if form.is_valid():
                print("form is valid")
                user = form.save()
                login(request, user)
                return redirect("home_view")
            else:
                print(form.errors)
                return render(
                    request, "signup.html", {"form": form, "error": form.errors}
                )

        return render(
            request,
            "signup.html",
            {"form": CustomUserCreationForm(), "error": "Las contraseñas no coinciden"},
        )


@login_required(login_url="/signup/")
def home(request):
    return render(request, "home.html")


def clr_view(request):
    return render(request, "clr.html")


# @login_required(login_url="/signup/")
def cultivo_maiz_recomendacion_view(request):
    cultivos = CultivoMaiz.objects.filter(user=request.user.id).values_list(
        "ID_LOTE", flat=True
    )

    return render(request, "sistema_recomendacion.html", {"cultivos": cultivos})


@login_required(login_url="/signup/")
def prediccion_view(request):

    prediccion_valor, valor_real = obtener_prediccion()

    return render(
        request,
        "prediccion_resultado.html",
        {"prediccion_valor": prediccion_valor, "valor_real": valor_real},
    )


@login_required(login_url="/signup/")
def prediccion_formulario_view(request):
    print("predecir-btn")
    cultivos = CultivoMaiz.objects.filter(user=request.user.id).values_list(
        "ID_LOTE", flat=True
    )
    # Cargar la fila original del CSV y convertirla a DataFrame
    df_nuevo_dato_original = pd.read_csv(CONSTANTES.DATASET_PATH).iloc[
        [CONSTANTES.ROW_DATA]
    ]  # Doble corchete para mantener como DataFrame

    prediccion_valor = None
    valor_real = None

    if request.method == "POST":
        print("post")
        form = DatosPrediccionForm(request.POST)

        if form.is_valid():
            print("form is valid")
            # Obtener los datos ingresados por el usuario y convertirlos a DataFrame
            datos_formulario = form.cleaned_data
            df_formulario = pd.DataFrame(
                [datos_formulario], index=df_nuevo_dato_original.index
            )

            # Usar combine_first para combinar los DataFrames.
            # Los valores del formulario tienen prioridad.
            df_nuevo_dato = df_formulario.combine_first(df_nuevo_dato_original)

            # Obtener la predicción con los datos actualizados
            prediccion_valor, valor_real = obtener_prediccion_formulario(df_nuevo_dato)
            prediccion_valor = round(prediccion_valor, 2)
        else:
            print("form is not valid")
            print(form.errors)

    else:
        # Si es GET, mostrar el formulario con los valores precargados
        form = DatosPrediccionForm(
            df_nuevo_dato_original.iloc[0].to_dict()
        )  # Pasar el diccionario al formulario.

    return render(
        request,
        "predict_form.html",
        {"form": form, "prediccion_valor": prediccion_valor, "cultivos": cultivos},
    )


@login_required(login_url="/signup/")
def dash_view(request):

    return render(request, "dash_index.html")


@login_required
def dashboard_data_api(request):
    # Get selected ID_LOTE from request
    lote_id = int(request.GET.get("lote_id", None))
    print("dashboard_data_api lote_id: ", lote_id)

    # Load data
    df = pd.DataFrame(list(CultivoMaiz.objects.filter(user=request.user).values()))

    if df.empty:
        return JsonResponse({
            "error": "No hay datos disponibles para este usuario"
        }, status=404)

    # Process data similar to your Dash app
    # Operaciones df porcentajes (%)
    df_unpivot_percentage = df.melt(
        id_vars=["ID_LOTE"],
        value_vars=[
            "Porc_A",
            "Porc_Ar",
            "Porc_ArA",
            "Porc_ArL",
            "Porc_FrL",
            "Porc_L",
            "Porc_F",
            "porc_x",
            "porc_y",
            "Porc_AF",
        ],
        var_name="soil_type",
        value_name="percentage",
    )

    # Operaciondes df (Controles)
    df_controles = df.copy()
    df_controles["count_plagas_quimico"] = (
        df["ContPlaQui_Antes_Siem"]
        + df["ContPlaQui_Emer_Flor"]
        + df["ContPlaQui_Flor_Cose"]
        + df["ContPlaQui_Siem_Emer"]
    )
    df_controles["count_malezas_quimico"] = (
        df["ContMalQui_Antes_Siem"]
        + df["ContMalQui_Emer_Flor"]
        + df["ContMalQui_Flor_Cose"]
        + df["ContMalQui_Siem_Emer"]
    )
    df_controles["count_enfer_quimico"] = (
        df["ContEnfQui_Emer_Flor"] + df["ContEnfQui_Flor_Cose"]
    )
    df_controles = df_controles[
        [
            "ID_LOTE",
            "count_plagas_quimico",
            "count_malezas_quimico",
            "count_enfer_quimico",
        ]
    ]
    df_types_controls = df_controles.melt(
        id_vars=["ID_LOTE"],
        value_vars=[
            "count_plagas_quimico",
            "count_malezas_quimico",
            "count_enfer_quimico",
        ],
        var_name="control_type",
        value_name="cantidad",
    )

    # Operaciones Tarjetas (Nitrogeno, Fosforo, Potasio y rendimiento)
    df_npk = df.copy()
    df_npk["total_nitrogeno"] = (
        df_npk["TotN_Antes_Siem"] + df_npk["TotN_Emer_Flor"] + df_npk["TotN_Siem_Emer"]
    )
    df_npk["total_fosforo"] = (
        df_npk["TotP_Antes_Siem"] + df_npk["TotP_Emer_Flor"] + df_npk["TotP_Siem_Emer"]
    )
    df_npk["total_potasio"] = (
        df_npk["TotK_Antes_Siem"] + df_npk["TotK_Emer_Flor"] + df_npk["TotK_Siem_Emer"]
    )
    df_npk = df_npk[
        [
            "ID_LOTE",
            "TIPO_SIEMBRA",
            "total_nitrogeno",
            "total_fosforo",
            "total_potasio",
            "RDT_AJUSTADO",
        ]
    ]

    # Filter data based on selection
    if lote_id and lote_id != "":

        print("dentro del if, lote_id: ", lote_id)
        
        df_filtrado = df[df.ID_LOTE == lote_id]
        df_filtrado_suelo = df_unpivot_percentage[
            df_unpivot_percentage.ID_LOTE == lote_id
        ].reset_index(drop=True)
        df_filtrado_suelo = df_filtrado_suelo[df_filtrado_suelo.percentage != 0]
        df_filtrado_qcontroles = df_types_controls[
            df_types_controls.ID_LOTE == lote_id
        ].reset_index(drop=True)
        df_filtrado_tabla = df_npk[df_npk.ID_LOTE == lote_id]

        # Card values
        total_nitrogeno = round(
            df_npk[df_npk.ID_LOTE == lote_id]["total_nitrogeno"].values[0], 2
        )
        total_fosforo = round(
            df_npk[df_npk.ID_LOTE == lote_id]["total_fosforo"].values[0], 2
        )
        total_potasio = round(
            df_npk[df_npk.ID_LOTE == lote_id]["total_potasio"].values[0], 2
        )
        rendimiento_prom = round(
            df_npk[df_npk.ID_LOTE == lote_id]["RDT_AJUSTADO"].values[0], 2
        )
    else:
        # Default views without selection
        dff_avg = df.groupby("MATERIAL_GENETICO", as_index=False)["RDT_AJUSTADO"].mean()
        dff_avg = dff_avg.sort_values(by="RDT_AJUSTADO", ascending=False)
        df_filtrado = dff_avg

        df_agrupado_soil = (
            df_unpivot_percentage.groupby("soil_type")["percentage"]
            .mean()
            .reset_index(drop=False)
        )
        df_agrupado_soil = df_agrupado_soil.sort_values(
            by="percentage", ascending=False
        )
        df_filtrado_suelo = df_agrupado_soil

        df_controls_group = (
            df_types_controls.groupby("control_type")["cantidad"]
            .sum()
            .reset_index(drop=False)
        )
        df_controls_group = df_controls_group.sort_values(
            by="cantidad", ascending=False
        )
        df_filtrado_qcontroles = df_controls_group

        df_filtrado_tabla = df_npk

        # Card values (averages)
        total_nitrogeno = round(df_npk.total_nitrogeno.mean(), 2)
        total_fosforo = round(df_npk.total_fosforo.mean(), 2)
        total_potasio = round(df_npk.total_potasio.mean(), 2)
        rendimiento_prom = round(df_npk.RDT_AJUSTADO.mean(), 2)

    # Generate charts
    # 1. Yield by genetic material
    if lote_id and lote_id != "":
        print("dentro del if 2, lote_id: ", lote_id)
        yield_fig = px.bar(
            df_filtrado,
            x="MATERIAL_GENETICO",
            y="RDT_AJUSTADO",
            title="Rendimiento Ajustado [Kg/Ha] por Material Genético",
            labels={
                "MATERIAL_GENETICO": "Material Genético",
                "RDT_AJUSTADO": "Rendimiento Ajustado",
            },
            color="MATERIAL_GENETICO",
        )
    else:
        yield_fig = px.bar(
            df_filtrado,
            x="MATERIAL_GENETICO",
            y="RDT_AJUSTADO",
            title="Rendimiento Ajustado [Kg/Ha] por Material Genético",
            labels={
                "MATERIAL_GENETICO": "Material Genético",
                "RDT_AJUSTADO": "Rendimiento Ajustado",
            },
            color="MATERIAL_GENETICO",
        )

    yield_fig.update_layout(
        title_x=0.5,
        xaxis_title="Material Genético",
        yaxis_title="Rendimiento Ajustado [Kg/Ha]",
    )

    # 2. Soil type percentages
    soil_fig = px.pie(
        df_filtrado_suelo,
        names="soil_type",
        values="percentage",
        title="Distribución de Porcentaje por Tipo de Suelo",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Viridis,
    )

    max_index = df_filtrado_suelo["percentage"].idxmax()
    pull_values = [0.1 if i == max_index else 0 for i in range(len(df_filtrado_suelo))]

    soil_fig.update_traces(
        textinfo="label+percent",
        insidetextfont=dict(size=14, color="white"),
        outsidetextfont=dict(size=14, color="black"),
        marker=dict(line=dict(color="black", width=2)),
        pull=pull_values,
    )

    soil_fig.update_layout(
        title=dict(
            text="Porcentaje (%) asociado a cada capa del suelo",
            font=dict(size=18),
            x=0.5,
        ),
        legend_title="Tipo de Suelo",
        legend=dict(
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=1,
        ),
        margin=dict(t=50, b=50, l=50, r=50),
    )

    # 3. Control types
    controls_fig = px.bar(
        df_filtrado_qcontroles,
        x="cantidad",
        y="control_type",
        text="cantidad",
        title="Cantidad de controles x tipo",
        color="control_type",
    )

    controls_fig.update_layout(
        xaxis_title="Cantidad",
        yaxis_title="Tipo",
        title_x=0.5,
        legend_title="control_type",
        margin=dict(t=50, b=50, l=50, r=50),
    )

    controls_fig.update_traces(texttemplate="%{text}", textposition="outside")

    # Convert the figures to HTML
    yield_chart_html = plot(yield_fig, output_type="div", include_plotlyjs=False)
    soil_chart_html = plot(soil_fig, output_type="div", include_plotlyjs=False)
    controls_chart_html = plot(controls_fig, output_type="div", include_plotlyjs=False)

    # Convert table data to serializable format
    table_data = df_filtrado_tabla.to_dict("records")
    for row in table_data:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None

    # Prepare response data
    response_data = {
        "total_nitrogeno": total_nitrogeno,
        "total_fosforo": total_fosforo,
        "total_potasio": total_potasio,
        "rendimiento_prom": rendimiento_prom,
        "yield_chart": yield_chart_html,
        "soil_chart": soil_chart_html,
        "controls_chart": controls_chart_html,
        "table_data": table_data,
        "table_columns": list(df_filtrado_tabla.columns),
    }

    #print("response_data: ", response_data)
    return JsonResponse(response_data)


@login_required(login_url="/signup/")
def plotly_view(request):
    # Load data
    #df = pd.DataFrame(list(CultivoMaiz.objects.all().values()))
    df = pd.DataFrame(list(CultivoMaiz.objects.filter(user=request.user).values()))

    if df.empty:
        return render(request, "maiz_dashboard.html", {
            "error_no_cultivo": "No hay datos disponibles para este usuario"
        })

    # Get selected ID_LOTE from request, if any
    selected_lote = request.GET.get("lote_id", None)

    # Process data similar to your Dash app
    # Operaciones df porcentajes (%)
    df_unpivot_percentage = df.melt(
        id_vars=["ID_LOTE"],
        value_vars=[
            "Porc_A",
            "Porc_Ar",
            "Porc_ArA",
            "Porc_ArL",
            "Porc_FrL",
            "Porc_L",
            "Porc_F",
            "porc_x",
            "porc_y",
            "Porc_AF",
        ],
        var_name="soil_type",
        value_name="percentage",
    )

    # Operaciondes df (Controles)
    df_controles = df.copy()
    df_controles["count_plagas_quimico"] = (
        df["ContPlaQui_Antes_Siem"]
        + df["ContPlaQui_Emer_Flor"]
        + df["ContPlaQui_Flor_Cose"]
        + df["ContPlaQui_Siem_Emer"]
    )
    df_controles["count_malezas_quimico"] = (
        df["ContMalQui_Antes_Siem"]
        + df["ContMalQui_Emer_Flor"]
        + df["ContMalQui_Flor_Cose"]
        + df["ContMalQui_Siem_Emer"]
    )
    df_controles["count_enfer_quimico"] = (
        df["ContEnfQui_Emer_Flor"] + df["ContEnfQui_Flor_Cose"]
    )
    df_controles = df_controles[
        [
            "ID_LOTE",
            "count_plagas_quimico",
            "count_malezas_quimico",
            "count_enfer_quimico",
        ]
    ]
    df_types_controls = df_controles.melt(
        id_vars=["ID_LOTE"],
        value_vars=[
            "count_plagas_quimico",
            "count_malezas_quimico",
            "count_enfer_quimico",
        ],
        var_name="control_type",
        value_name="cantidad",
    )

    # Operaciones Tarjetas (Nitrogeno, Fosforo, Potasio y rendimiento)
    df_npk = df.copy()
    df_npk["total_nitrogeno"] = (
        df_npk["TotN_Antes_Siem"] + df_npk["TotN_Emer_Flor"] + df_npk["TotN_Siem_Emer"]
    )
    df_npk["total_fosforo"] = (
        df_npk["TotP_Antes_Siem"] + df_npk["TotP_Emer_Flor"] + df_npk["TotP_Siem_Emer"]
    )
    df_npk["total_potasio"] = (
        df_npk["TotK_Antes_Siem"] + df_npk["TotK_Emer_Flor"] + df_npk["TotK_Siem_Emer"]
    )
    df_npk = df_npk[
        [
            "ID_LOTE",
            "TIPO_SIEMBRA",
            "total_nitrogeno",
            "total_fosforo",
            "total_potasio",
            "RDT_AJUSTADO",
        ]
    ]

    # Filter data based on selection
    if selected_lote:
        df_filtrado = df[df.ID_LOTE == selected_lote]
        df_filtrado_suelo = df_unpivot_percentage[
            df_unpivot_percentage.ID_LOTE == selected_lote
        ].reset_index(drop=True)
        df_filtrado_suelo = df_filtrado_suelo[df_filtrado_suelo.percentage != 0]
        df_filtrado_qcontroles = df_types_controls[
            df_types_controls.ID_LOTE == selected_lote
        ].reset_index(drop=True)
        df_filtrado_tabla = df_npk[df_npk.ID_LOTE == selected_lote]

        # Card values
        total_nitrogeno = round(
            df_npk[df_npk.ID_LOTE == selected_lote]["total_nitrogeno"].values[0], 2
        )
        total_fosforo = round(
            df_npk[df_npk.ID_LOTE == selected_lote]["total_fosforo"].values[0], 2
        )
        total_potasio = round(
            df_npk[df_npk.ID_LOTE == selected_lote]["total_potasio"].values[0], 2
        )
        rendimiento_prom = round(
            df_npk[df_npk.ID_LOTE == selected_lote]["RDT_AJUSTADO"].values[0], 2
        )
    else:
        # Default views without selection
        dff_avg = df.groupby("MATERIAL_GENETICO", as_index=False)["RDT_AJUSTADO"].mean()
        dff_avg = dff_avg.sort_values(by="RDT_AJUSTADO", ascending=False)
        df_filtrado = dff_avg

        df_agrupado_soil = (
            df_unpivot_percentage.groupby("soil_type")["percentage"]
            .mean()
            .reset_index(drop=False)
        )
        df_agrupado_soil = df_agrupado_soil.sort_values(
            by="percentage", ascending=False
        )
        df_filtrado_suelo = df_agrupado_soil

        df_controls_group = (
            df_types_controls.groupby("control_type")["cantidad"]
            .sum()
            .reset_index(drop=False)
        )
        df_controls_group = df_controls_group.sort_values(
            by="cantidad", ascending=False
        )
        df_filtrado_qcontroles = df_controls_group

        df_filtrado_tabla = df_npk

        # Card values (averages)
        total_nitrogeno = round(df_npk.total_nitrogeno.mean(), 2)
        total_fosforo = round(df_npk.total_fosforo.mean(), 2)
        total_potasio = round(df_npk.total_potasio.mean(), 2)
        rendimiento_prom = round(df_npk.RDT_AJUSTADO.mean(), 2)

    # Generate charts
    # 1. Yield by genetic material
    if selected_lote:
        fig1 = px.bar(
            df_filtrado,
            x="MATERIAL_GENETICO",
            y="RDT_AJUSTADO",
            title="Rendimiento Ajustado [Kg/Ha] por Material Genético",
            labels={
                "MATERIAL_GENETICO": "Material Genético",
                "RDT_AJUSTADO": "Rendimiento Ajustado",
            },
            color="MATERIAL_GENETICO",
        )
    else:
        fig1 = px.bar(
            df_filtrado,
            x="MATERIAL_GENETICO",
            y="RDT_AJUSTADO",
            title="Rendimiento Ajustado [Kg/Ha] por Material Genético",
            labels={
                "MATERIAL_GENETICO": "Material Genético",
                "RDT_AJUSTADO": "Rendimiento Ajustado",
            },
            color="MATERIAL_GENETICO",
        )

    fig1.update_layout(
        title_x=0.5,
        xaxis_title="Material Genético",
        yaxis_title="Rendimiento Ajustado [Kg/Ha]",
    )

    # 2. Soil type percentages
    fig_pie = px.pie(
        df_filtrado_suelo,
        names="soil_type",
        values="percentage",
        title="Distribución de Porcentaje por Tipo de Suelo",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Viridis,
    )

    max_index = df_filtrado_suelo["percentage"].idxmax()
    pull_values = [0.1 if i == max_index else 0 for i in range(len(df_filtrado_suelo))]

    fig_pie.update_traces(
        textinfo="label+percent",
        insidetextfont=dict(size=14, color="white"),
        outsidetextfont=dict(size=14, color="black"),
        marker=dict(line=dict(color="black", width=2)),
        pull=pull_values,
    )

    fig_pie.update_layout(
        title=dict(
            text="Porcentaje (%) asociado a cada capa del suelo",
            font=dict(size=18),
            x=0.5,
        ),
        legend_title="Tipo de Suelo",
        legend=dict(
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=1,
        ),
        margin=dict(t=50, b=50, l=50, r=50),
    )

    # 3. Control types
    fig_controls = px.bar(
        df_filtrado_qcontroles,
        x="cantidad",
        y="control_type",
        text="cantidad",
        title="Cantidad de controles x tipo",
        color="control_type",
    )

    fig_controls.update_layout(
        xaxis_title="Cantidad",
        yaxis_title="Tipo",
        title_x=0.5,
        legend_title="control_type",
        margin=dict(t=50, b=50, l=50, r=50),
    )

    fig_controls.update_traces(texttemplate="%{text}", textposition="outside")

    # Convert the figures to div
    yield_chart = plot(fig1, output_type="div", include_plotlyjs=True)
    soil_chart = plot(fig_pie, output_type="div", include_plotlyjs=True)
    controls_chart = plot(fig_controls, output_type="div", include_plotlyjs=True)

    # Generate lote options for dropdown
    lote_options = sorted(df.ID_LOTE.unique())

    # Get table data
    table_data = df_filtrado_tabla.to_dict("records")

    paginator = Paginator(table_data, 10)  # registros por página.
    page = request.GET.get("page")
    try:
        table_page = paginator.page(page)
    except PageNotAnInteger:
        table_page = paginator.page(1)
    except EmptyPage:
        table_page = paginator.page(paginator.num_pages)
    
    page_range = get_page_range(table_page.number, paginator.num_pages, delta=2)

    context = {
        "yield_chart": yield_chart,
        "soil_chart": soil_chart,
        "controls_chart": controls_chart,
        "lote_options": lote_options,
        "selected_lote": selected_lote,
        "total_nitrogeno": total_nitrogeno,
        "total_fosforo": total_fosforo,
        "total_potasio": total_potasio,
        "rendimiento_prom": rendimiento_prom,
        "table_data": table_data,
        "table_columns": df_filtrado_tabla.columns,
        "year": datetime.datetime.now().year,
        "table_page": table_page,
        "paginator": paginator,
        "page_range": page_range,
    }

    return render(request, "maiz_dashboard.html", context)


@login_required(login_url="/signup/")
def cultivo_maiz_table_view(request):

    return render(request, "cultivo_maiz_table.html")


def cultivo_maiz_get(request, opcion):
    print("opcion: ", opcion)
    data = {"message": "Sin opcion"}

    return JsonResponse(data)


@csrf_exempt
@login_required(login_url="/signup/")
def api_post_cultivos_listar(request):
    print("cultivo_maiz_post")
    data = {"message": "Sin opcion"}
    if request.method == "POST":
        print("request post")
        data = json.loads(request.body)
        page_number = int(data.get("page", 1))
        page_size = int(data.get("page_size", 50))

        cultivos = CultivoMaiz.objects.filter(user=request.user.id)
        paginator = Paginator(cultivos, page_size)
        page_obj = paginator.get_page(page_number)

        cultivos_data = serializers.serialize("json", page_obj.object_list)

        is_admin = request.user.is_superuser
        return JsonResponse(
            {
                "message": "Success",
                "cultivos_data": cultivos_data,
                "page": page_number,
                "num_pages": paginator.num_pages,
                "total": paginator.count,
                "is_admin": is_admin,
            }
        )

    return JsonResponse(data)


@csrf_exempt
def api_post_formulario_cultivo(request, cultivo_id=None):
    print("formulario_cultivo_post")
    if cultivo_id:
        cultivo = get_object_or_404(CultivoMaiz, pk=cultivo_id)
        form = DatosPrediccionForm(instance=cultivo)
        print("cultivo_id: ", form.instance.id)
    else:
        form = DatosPrediccionForm()
        print("cultivo_id: None")

    form_html = render_to_string(
        "layouts/formulario_cultivo.html", {"form": form}, request=request
    )

    return JsonResponse({"form_html": form_html})


@csrf_exempt
def api_post_editar_formulario_cultivo(request, cultivo_id=None):
    print("editar_formulario_cultivo_post")
    if request.method == "POST":
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse(
                {"status": "error", "message": "JSON inválido"}, status=400
            )
    if cultivo_id:
        print("editar formulario cultivo_id: ", cultivo_id)
        cultivo = get_object_or_404(CultivoMaiz, pk=cultivo_id)
        form = DatosPrediccionForm(data, instance=cultivo)
        if form.is_valid():
            form.save()  # Actualiza el registro en la base de datos
            return JsonResponse(
                {"status": "success", "message": "Cultivo actualizado correctamente"}
            )
        else:
            # Devolvemos los errores del formulario en caso de datos inválidos
            return JsonResponse({"status": "error", "errors": form.errors}, status=400)

    else:
        print("cultivo_id: None")
        form = DatosPrediccionForm(data)
        if form.is_valid():
            cultivo = form.save(commit=False)  # No guarda inmediatamente en la BD
            cultivo.user = request.user
            form.save()
            return JsonResponse(
                {"status": "success", "message": "Cultivo creado correctamente"}
            )

    return JsonResponse({"test": "test"})


@csrf_exempt
def api_post_eliminar_formulario_cultivo(request, cultivo_id):
    if request.method == "DELETE":
        cultivo = get_object_or_404(CultivoMaiz, pk=cultivo_id)
        cultivo.delete()
        print("cultivo eliminado")
        return JsonResponse(
            {"status": "success", "message": "Cultivo eliminado correctamente"}
        )
    else:
        return JsonResponse(
            {"status": "error", "message": "Método no permitido"}, status=405
        )


@csrf_exempt
def obtener_cultivo(request):
    data = json.loads(request.body)
    id_lote = data.get("id_lote")
    if id_lote:
        try:
            cultivo = CultivoMaiz.objects.get(ID_LOTE=id_lote)
            form = DatosPrediccionForm(instance=cultivo)
            html = render_to_string(
                "partials/_cultivo_form.html", {"form": form}, request=request
            )
            return JsonResponse({"status": "success", "form_html": html})
        except CultivoMaiz.DoesNotExist:
            return JsonResponse(
                {"status": "error", "message": "Cultivo no encontrado."}
            )
    return JsonResponse(
        {"status": "error", "message": "No se proporcionó un ID de lote."}
    )


@csrf_exempt
def obtener_cultivo_variables_manejo(request):
    data = json.loads(request.body)
    id_lote = data.get("id_lote")

    if id_lote:
        try:
            cultivo = CultivoMaiz.objects.get(ID_LOTE=id_lote)

            # Create a new form with only the management variables
            management_fields = [
                "ContEnfQui_Emer_Flor",
                "ContEnfQui_Flor_Cose",
                "ContMalMec_Siem_Emer",
                "ContMalMec_Emer_Flor",
                "ContMalMec_Flor_Cose",
                "ContMalQui_Antes_Siem",
                "ContMalQui_Siem_Emer",
                "ContMalQui_Emer_Flor",
                "ContMalQui_Flor_Cose",
                "ContPlaQui_Antes_Siem",
                "ContPlaQui_Siem_Emer",
                "ContPlaQui_Emer_Flor",
                "ContPlaQui_Flor_Cose",
                "TotN_Antes_Siem",
                "TotN_Siem_Emer",
                "TotN_Emer_Flor",
                "TotP_Antes_Siem",
                "TotP_Siem_Emer",
                "TotP_Emer_Flor",
                "TotK_Antes_Siem",
                "TotK_Siem_Emer",
                "TotK_Emer_Flor",
            ]

            management_form_data = {
                field: getattr(cultivo, field) for field in management_fields
            }
            management_form = DatosPrediccionForm(initial=management_form_data)

            management_form.get_field_groups = lambda: [
                (
                    "Variables de Manejo del Cultivo",
                    [management_form[field] for field in management_fields],
                )
            ]

            html = render_to_string(
                "partials/_cultivo_form.html",
                {"form": management_form},
                request=request,
            )

            return JsonResponse({"status": "success", "form_html": html})

        except CultivoMaiz.DoesNotExist:
            return JsonResponse(
                {"status": "error", "message": "Cultivo no encontrado."}
            )

    return JsonResponse(
        {"status": "error", "message": "No se proporcionó un ID de lote."}
    )


@csrf_exempt
def optimizar_cultivo(request):
    max_length = 100
    try:
        data = json.loads(request.body)
        cultivo_id = data.get("id_lote", 80)
        precio_venta = int(data.get("precio_venta", 2500))
        presupuesto = int(data.get("presupuesto", 4000000))
        costos_unitarios = data.get("costos_unitarios", {})

        rendimiento_esperado, ganancia, costo, mejor_solucion = optimizar_por_id_lote(
            cultivo_id, precio_venta, presupuesto, costos_unitarios
        )
        #print(
        #    "mejor solucion...................................................................................................... ",
        #    mejor_solucion,
        #)
        # Convierte best_fitness_history a lista o string si es necesario
        # best_fitness_str = limitar_texto(str(best_fitness_history), max_length)
        # fitness_alpha_str = limitar_texto(str(fitness_alpha), max_length)
        # alpha_pos_str = limitar_texto(str(alpha_pos), max_length)

        html_response = (
            f"<fieldset class='collapsible-fieldset optimized'>"
            f"<legend><button type='button' class='collapse-toggle'><span class='toggle-icon'>▼</span></button>Variables de Manejo del Cultivo Optimizadas</legend>"
            f"<div class='fieldset-content'>"
        )

        for column in mejor_solucion.columns:

            label_mapping = {
                "ContEnfQui_Emer_Flor": CONSTANTES.CONT_ENF_QUI_EMER_FLOR_LABEL,
                "ContEnfQui_Flor_Cose": CONSTANTES.CONT_ENF_QUI_FLOR_COSE_LABEL,
                "ContMalMec_Siem_Emer": CONSTANTES.CONT_MAL_MEC_SIEM_EMER_LABEL,
                "ContMalMec_Emer_Flor": CONSTANTES.CONT_MAL_MEC_EMER_FLOR_LABEL,
                "ContMalMec_Flor_Cose": CONSTANTES.CONT_MAL_MEC_FLOR_COSE_LABEL,
                "ContMalQui_Antes_Siem": CONSTANTES.CONT_MAL_QUI_ANTES_SIEM_LABEL,
                "ContMalQui_Siem_Emer": CONSTANTES.CONT_MAL_QUI_SIEM_EMER_LABEL,
                "ContMalQui_Emer_Flor": CONSTANTES.CONT_MAL_QUI_EMER_FLOR_LABEL,
                "ContMalQui_Flor_Cose": CONSTANTES.CONT_MAL_QUI_FLOR_COSE_LABEL,
                "ContPlaQui_Antes_Siem": CONSTANTES.CONT_PLA_QUI_ANTES_SIEM_LABEL,
                "ContPlaQui_Siem_Emer": CONSTANTES.CONT_PLA_QUI_SIEM_EMER_LABEL,
                "ContPlaQui_Emer_Flor": CONSTANTES.CONT_PLA_QUI_EMER_FLOR_LABEL,
                "ContPlaQui_Flor_Cose": CONSTANTES.CONT_PLA_QUI_FLOR_COSE_LABEL,
                "TotN_Antes_Siem": CONSTANTES.TOT_N_ANTES_SIEM_LABEL,
                "TotN_Siem_Emer": CONSTANTES.TOT_N_SIEM_EMER_LABEL,
                "TotN_Emer_Flor": CONSTANTES.TOT_N_EMER_FLOR_LABEL,
                "TotP_Antes_Siem": CONSTANTES.TOT_P_ANTES_SIEM_LABEL,
                "TotP_Siem_Emer": CONSTANTES.TOT_P_SIEM_EMER_LABEL,
                "TotP_Emer_Flor": CONSTANTES.TOT_P_EMER_FLOR_LABEL,
                "TotK_Antes_Siem": CONSTANTES.TOT_K_ANTES_SIEM_LABEL,
                "TotK_Siem_Emer": CONSTANTES.TOT_K_SIEM_EMER_LABEL,
                "TotK_Emer_Flor": CONSTANTES.TOT_K_EMER_FLOR_LABEL,
            }

            label = label_mapping.get(column, column)

            html_response += f"""
                <label for='id_{column}'>{label}</label>
                <input type='text' name='{column}' id='id_{column}' value='{mejor_solucion[column].values[0]}' readonly>
                """

        html_response += "</div></fieldset>"

        html_response += (
            f"<fieldset class='optimized'><legend>Respuesta de optimización</legend>"
            f"<p><strong>El rendimiento esperado es [Kg/Ha]: :</strong></p>"
            f"<p>{rendimiento_esperado}</p>"
            f"<p><strong>La ganancia esperada es [Pesos Colmbianos]:</strong></p>"
            f"<p>{ganancia}</p>"
            f"<p><strong>El costo de Implementación de las practicas es [Pesos Colombianos]: </strong></p>"
            f"<p>{costo}</p>"
            f"</fieldset>"
        )

        return JsonResponse({"status": "success", "message": html_response})

    except Exception as e:
        # Puedes loggear el error si es necesario
        return JsonResponse(
            {"status": "error", "message": f"<p><strong>Error:</strong> {str(e)}</p>"}
        )


@csrf_exempt
def obtener_costos_unitarios(request):
    html = render_to_string("partials/costos_unitarios.html", request=request)
    return JsonResponse({"status": "success", "html": html})


def limitar_texto(texto, max_length):
    return texto if len(texto) <= max_length else texto[:max_length] + "..."


def get_page_range(current_page, total_pages, delta=2):
    """
    Return a list of page numbers to display in pagination control,
    with the current page in the middle (if possible).
    """
    if total_pages <= 5:
        # If total pages <= 5, show all pages
        return range(1, total_pages + 1)
    
    # Calculate range with current page in the middle
    left = max(1, current_page - delta)
    right = min(current_page + delta, total_pages)
    
    # Adjust if we're at the start or end
    if left == 1:
        right = min(5, total_pages)
    elif right == total_pages:
        left = max(1, total_pages - 4)
    
    return range(left, right + 1)