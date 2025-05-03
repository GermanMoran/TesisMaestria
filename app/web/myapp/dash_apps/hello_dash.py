from django_plotly_dash import DjangoDash
from dash import dcc, html, callback, Output, Input, dash_table, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import os
from django.conf import settings
import datetime
from myapp.models import CultivoMaiz

# Carga de Datos
# ========================================================================
# Ajusta la ruta para que funcione con Django
df_path = os.path.join(settings.BASE_DIR, "Data/DatasetFinal.csv")
#df = pd.read_csv(df_path)
df = pd.DataFrame(list(CultivoMaiz.objects.all().values()))


# Operaciones df porcentajes (%)
# ======================================================================
df_unpivot_percentage = df.melt(id_vars=["ID_LOTE"],
                                value_vars=['Porc_A','Porc_Ar','Porc_ArA','Porc_ArL','Porc_FrL','Porc_L','Porc_F','porc_x','porc_y','Porc_AF'],
                                var_name="soil_type",
                                value_name="percentage")


# Operaciondes df (Controles)
# ====================================================================
df_controles = df.copy()
df_controles["count_plagas_quimico"] = df["ContPlaQui_Antes_Siem"] +df["ContPlaQui_Emer_Flor"] + df["ContPlaQui_Flor_Cose"] + df["ContPlaQui_Siem_Emer"]
df_controles["count_malezas_quimico"] = df["ContMalQui_Antes_Siem"] +df["ContMalQui_Emer_Flor"] + df["ContMalQui_Flor_Cose"] + df["ContMalQui_Siem_Emer"]
df_controles["count_enfer_quimico"] = df["ContEnfQui_Emer_Flor"] + df["ContEnfQui_Flor_Cose"]
df_controles = df_controles[["ID_LOTE","count_plagas_quimico","count_malezas_quimico","count_enfer_quimico"]]
df_types_controls = df_controles.melt(id_vars=["ID_LOTE"],
                            value_vars=["count_plagas_quimico","count_malezas_quimico","count_enfer_quimico"],
                            var_name="control_type",
                            value_name="cantidad")


# Operaciones Tragetas (Nitrogeno, Fosforo, Potasio y rendimiento)
# =======================================================================
df_npk = df.copy()
df_npk["total_nitrogeno"]= df_npk["TotN_Antes_Siem"] +df_npk["TotN_Emer_Flor"] + df_npk["TotN_Siem_Emer"]
df_npk["total_fosforo"]= df_npk["TotP_Antes_Siem"] +df_npk["TotP_Emer_Flor"] + df_npk["TotP_Siem_Emer"]
df_npk["total_potasio"]= df_npk["TotK_Antes_Siem"] +df_npk["TotK_Emer_Flor"] + df_npk["TotK_Siem_Emer"]
df_npk = df_npk[["ID_LOTE", "TIPO_SIEMBRA","total_nitrogeno","total_fosforo","total_potasio","RDT_AJUSTADO"]]


# Inicializar la app con un tema de Bootstrap
# ======================================================
app = DjangoDash('MaizDashboard', 
                 external_stylesheets=[dbc.themes.BOOTSTRAP], 
                 serve_locally=True)

# Navbar
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Img(src="/static/icono.png", height="40px")),
            dbc.Col(
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Dashboard", active=True, href="#")),
                        dbc.NavItem(dbc.NavLink("Volver a Django", href="/")),
                    ],
                    navbar=True
                )
            ),
        ], align="center"),
    ]),
    color="dark",
    dark=True
)

# Cuerpo de la aplicación
clr_body = html.Div([
    dbc.Row([  
        dbc.Col(dcc.Dropdown(
                id='dropdown-selection',
                options=[{'label': i, 'value': i} for i in sorted(df.ID_LOTE.unique())],
                value=None,
                clearable=True,
                multi=False,
                placeholder="Seleccione ID Lote",
                className="custom-dropdown"
            ), width=4)
    ], className="mb-4 d-flex justify-content-start"),
    dbc.Row([
        # Tarjeta Total Nitrogeno
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Nitrogeno", className="card-title"),
                    html.H2(id="total_nitrogeno"),
                ])
            ], color="primary", inverse=True), width=3
        ),

        # Tarjeta total Fosforo
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Fosforo", className="card-title"),
                    html.H2(className="card-text",id="total_fosforo"),
                ])
            ], color="success", inverse=True), width=3
        ),

        # Tarjeta total potasio
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Potasio", className="card-title"),
                    html.H2( className="card-text",id="total_potasio"),
                ])
            ], color="danger", inverse=True), width=3
        ),

        # Trageta Total rendimiento
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Rendimiento", className="card-title"),
                    html.H2(className="card-text",id="total_rendimiento"),
                ])
            ], color="info", inverse=True), width=3
        )
    ], className="mb-4 d-flex justify-content-around"),
    
    dbc.Row([  
        dbc.Col(dcc.Graph(id='yield_by_genetic'),className="graph-container", width=6),
        dbc.Col(dcc.Graph(id='percentage_by_soil'),className="graph-container", width=6)
    ], className="mb-4 d-flex justify-content-around"),

    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id="tabla",
                columns=[{"name": col, "id": col} for col in df_npk.columns],
                data=df_npk.to_dict("records"),  # Datos iniciales sin filtrar
                page_size=10,  # Número de filas por página
                sort_action="native",
                sort_mode="multi",
                style_table={"overflowX": "auto"},  # Ajuste horizontal
                style_cell={"textAlign": "left", "padding": "10px"},  # Estilo de celdas
                style_header={"backgroundColor": "lightblue", "fontWeight": "bold"},  # Encabezado con estilo
                row_selectable="multi",  # Permite seleccionar solo una fila
                selected_rows=[]  # Lista vacía al inicio (ninguna fila seleccionada)
            )
        ], width=6),
        dbc.Col(dcc.Graph(id='cantidad_controles'), width=6)
    ], justify="center")
], className="ds4a-body")


clr_header = html.Div([
        html.H1('Analítica Descriptiva Cultivo Maíz', style={'textAlign': 'center', 'color': 'white'}),
], className="header-clr")

current_year = datetime.datetime.now().year
clr_footer = html.Footer(
    f"© {current_year} Maize App - Todos los derechos reservados",
    style={
        'textAlign': 'center',
        'padding': '10px',
        'backgroundColor': '#252e3f', 
        'color': 'white',
        'position': 'fixed',
        'bottom': '0',
        'width': '100%'
    }
)

# Layout de la aplicación
#navbar,
#clr_footer
#clr_header,
app.layout = html.Div([
    clr_body
], className="clr-app")



# Callback para actualizar el gráfico
@app.callback(
    Output('yield_by_genetic', 'figure'),
    Output('percentage_by_soil','figure'),
    Output('cantidad_controles','figure'),
    Output("total_nitrogeno",'children'),
    Output("total_fosforo",'children'),
    Output("total_potasio",'children'),
    Output("total_rendimiento",'children'),
    Output("tabla", "data"),
    Input('dropdown-selection', 'value'),
    Input("tabla", "selected_rows")
)


def update_graph(value, fila_seleccionada):

    if value == ' ' or value is None:
        # Material Genetico
        # =============================================================================
        dff_avg = df.groupby("MATERIAL_GENETICO", as_index=False)["RDT_AJUSTADO"].mean()
        dff_avg = dff_avg.sort_values(by="RDT_AJUSTADO", ascending=False)  # Ordenar por rendimiento
        fig1 = px.bar(dff_avg, x="MATERIAL_GENETICO", y="RDT_AJUSTADO",
                 title="Rendimiento Ajustado [Kg/Ha] por Material Genético",
                 labels={"MATERIAL_GENETICO": "Material Genético", "RDT_AJUSTADO": "Rendimiento Ajustado"},
                 color="MATERIAL_GENETICO")
        fig1.update_traces(marker=dict(line=dict(width=0)))  # Eliminar líneas de contorno en las barras
        fig1.update_layout(
            title_x=0.5,
            xaxis_title="Material Genético",
            yaxis_title="Rendimiento Ajustado [Kg/Ha]")
        
        # Porcentajes de Tierra 
        # ===============================================================================
        df_agrupado_soil = df_unpivot_percentage.groupby("soil_type")["percentage"].mean().reset_index(drop=False)
        df_agrupado_soil = df_agrupado_soil.sort_values(by="percentage", ascending=False)

        # Crear gráfico de pastel mejorado
        fig_pie = px.pie(
            df_agrupado_soil,  # DataFrame
            names="soil_type",  # Categorías
            values="percentage",  # Valores
            title="Distribución de Porcentaje por Tipo de Suelo",
            hole=0.4,  # Gráfico tipo "donut"
            color_discrete_sequence=px.colors.sequential.Viridis  # Colores llamativos
        )

        # Calcular "pull" dinámico para resaltar el segmento más grande
        max_index = df_agrupado_soil["percentage"].idxmax()
        pull_values = [0.1 if i == max_index else 0 for i in range(len(df_agrupado_soil))]

        # Personalizar etiquetas, colores y detalles visuales
        fig_pie.update_traces(
            textinfo="label+percent",  # Muestra etiqueta y porcentaje
            insidetextfont=dict(size=14, color="white"),  # Tamaño y color del texto interno
            outsidetextfont=dict(size=14, color="black"),  # Texto externo en negro
            marker=dict(line=dict(color="black", width=2)),  # Borde negro para mejor contraste
            pull=pull_values  # Resaltar solo el mayor segmento
        )

        # Personalizar título y diseño
        fig_pie.update_layout(
            title=dict(text="Porcentaje (%) asociado a cada capa del suelo", font=dict(size=18), x=0.5),
            legend_title="Tipo de Suelo",
            legend=dict(font=dict(size=12), bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1),
            margin=dict(t=50, b=50, l=50, r=50)
        )

        # Cantidad cantroles
        # =============================================================================================
        df_controls_group = df_types_controls.groupby("control_type")["cantidad"].sum().reset_index(drop=False)
        df_controls_group = df_controls_group.sort_values(by="cantidad", ascending=False)
        # Crear gráfico de barras horizontales
        fig_controls = px.bar(
            df_controls_group, 
            x="cantidad",  # Eje X
            y="control_type",  # Eje Y
            text="cantidad",  # Mostrar valores dentro de las barras
            title="Cantidad de controles x tipo",
            color ="control_type"
        )

        # Personalizar diseño
        fig_controls.update_layout(
            xaxis_title="Cantidad",
            yaxis_title="Tipo",
            title_x=0.5,  # Centrar título
            legend_title="control_type",
            margin=dict(t=50, b=50, l=50, r=50)
        )

        # Mostrar valores sobre las barras
        fig_controls.update_traces(texttemplate='%{text}', textposition='outside')

        # Targetas (NPK) y rendimiento
        # =======================================================================
        total_nitrogeno = round(df_npk.total_nitrogeno.mean(), 2)
        total_fosforo = round(df_npk.total_fosforo.mean(), 2)
        total_potasio = round(df_npk.total_potasio.mean(), 2)
        rendimeinto_prom = round(df_npk.RDT_AJUSTADO.mean(), 2)

        # Tabla
        # ===================================================================
        df_filtrado_tabla = df_npk.copy()
        datos = df_filtrado_tabla.to_dict("records")

        # Interaccion con la tabla 
        # ===================================================================
 
        if fila_seleccionada and len(fila_seleccionada) > 0:
            fila_index = fila_seleccionada[0]
            df_seleccionado = df.iloc[[fila_index]]  # Filtrar la fila seleccionada
            df_filtrado = df[df.ID_LOTE == df_seleccionado.ID_LOTE.values[0]]
            fig1 = px.bar(df_filtrado, x="MATERIAL_GENETICO", y="RDT_AJUSTADO",
                title="Rendimiento Ajustado [Kg/Ha] por Material Genético",
                labels={"MATERIAL_GENETICO": "Material Genético", "RDT_AJUSTADO": "Rendimiento Ajustado"},
                color="MATERIAL_GENETICO")
            fig1.update_layout(
                title_x=0.5,
                xaxis_title="Material Genético",
                yaxis_title="Rendimiento Ajustado [kg/HA]")

    else:
        # Material Genetico
        #===========================================================================
        df_filtrado = df[df.ID_LOTE == value]
        fig1 = px.bar(df_filtrado, x="MATERIAL_GENETICO", y="RDT_AJUSTADO",
            title="Rendimiento Ajustado [Kg/Ha] por Material Genético",
            labels={"MATERIAL_GENETICO": "Material Genético", "RDT_AJUSTADO": "Rendimiento Ajustado"},
            color="MATERIAL_GENETICO")
        fig1.update_layout(
            title_x=0.5,
            xaxis_title="Material Genético",
            yaxis_title="Rendimiento Ajustado [kg/HA]")
        
        # (%) segun el tipo de seuelo
        # =======================================================================
        df_filtrado_suelo = df_unpivot_percentage[df_unpivot_percentage.ID_LOTE == value].reset_index(drop=True)
        df_filtrado_suelo = df_filtrado_suelo[df_filtrado_suelo.percentage != 0]
        # Crear gráfico de pastel mejorado
        fig_pie = px.pie(
            df_filtrado_suelo,  # DataFrame
            names="soil_type",  # Categorías
            values="percentage",  # Valores
            title="Distribución de Porcentaje (%) por Tipo de Suelo",
            hole=0.4  # Gráfico tipo "donut"
        )

        # Calcular "pull" dinámico para resaltar el segmento más grande
        max_index = df_filtrado_suelo["percentage"].idxmax()
        pull_values = [0.1 if i == max_index else 0 for i in range(len(df_filtrado_suelo))]

        # Personalizar etiquetas, colores y detalles visuales
        fig_pie.update_traces(
            textinfo="label+percent",  # Muestra etiqueta y porcentaje
            insidetextfont=dict(size=14, color="white"),  # Tamaño y color del texto interno
            outsidetextfont=dict(size=14, color="black"),  # Texto externo en negro
            marker=dict(line=dict(color="black", width=2)),  # Borde negro para mejor contraste
            pull=pull_values  # Resaltar solo el mayor segmento
        )

        # Personalizar título y diseño
        fig_pie.update_layout(
            title=dict(text="Porcentaje (%) asociado a cada capa del suelo", font=dict(size=18), x=0.5),
            legend_title="Tipo de Suelo",
            legend=dict(font=dict(size=12), bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1),
            margin=dict(t=50, b=50, l=50, r=50)
        )

        # Cantidad de controles
        # ===============================================================================
        df_filtrado_qcontroles = df_types_controls[df_types_controls.ID_LOTE == value].reset_index(drop=True)
        # Crear gráfico de barras horizontales
        fig_controls = px.bar(
            df_filtrado_qcontroles, 
            x="cantidad",  # Eje X
            y="control_type",  # Eje Y
            text="cantidad",  # Mostrar valores dentro de las barras
            title="Cantidad de controles x Tipo",
            color="control_type"
        )

        # Personalizar diseño
        fig_controls.update_layout(
            xaxis_title="Cantidad",
            yaxis_title="Tipo",
            title_x=0.5,  # Centrar título
            legend_title="control_type",
            margin=dict(t=50, b=50, l=50, r=50)
        )

        # Mostrar valores sobre las barras
        fig_controls.update_traces(texttemplate='%{text}', textposition='outside')

        # Targetas
        #=========================================================================
        total_nitrogeno = round(df_npk[df_npk.ID_LOTE == value]["total_nitrogeno"].values[0], 2)
        total_fosforo = round(df_npk[df_npk.ID_LOTE == value]["total_fosforo"].values[0], 2)
        total_potasio = round(df_npk[df_npk.ID_LOTE == value]["total_potasio"].values[0], 2)
        rendimeinto_prom = round(df_npk[df_npk.ID_LOTE == value]["RDT_AJUSTADO"].values[0], 2)

        # Insertamos la tabla 
        # ========================================================================
        df_filtrado_tabla = df_npk[df_npk.ID_LOTE == value]
        datos = df_filtrado_tabla.to_dict("records")

    return [fig1, fig_pie, fig_controls, total_nitrogeno, total_fosforo, total_potasio, rendimeinto_prom, datos]