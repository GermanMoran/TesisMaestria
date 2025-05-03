from django.urls import path, include
from .views import home
from .views import prediccion_view
from .views import prediccion_formulario_view
from .views import cultivo_maiz_table_view
from .views import cultivo_maiz_get
from .views import api_post_cultivos_listar
from .views import api_post_formulario_cultivo
from .views import api_post_editar_formulario_cultivo
from .views import api_post_eliminar_formulario_cultivo
from .views import cultivo_maiz_recomendacion_view
from .views import obtener_cultivo
from .views import optimizar_cultivo
from .views import dash_view
from .views import signup_view
from .views import login_view
from .views import logout_view
from .views import clr_view
from .views import obtener_costos_unitarios
from .views import obtener_cultivo_variables_manejo
from . import views

urlpatterns = [
    path('', home, name='home_view'),
    path('prediccion/', prediccion_view, name='prediccion_view'),
    path('prediccion_formulario/', prediccion_formulario_view, name='prediccion_formulario_view'),
    path('cultivo_maiz_tabla/', cultivo_maiz_table_view, name='cultivo_maiz_table_view'),
    path('cultivo_maiz_sistema_recomendacion_gwo/', cultivo_maiz_recomendacion_view, name='cultivo_maiz_recomendacion_view'),
    path('api/cultivos/get/<str:opcion>/', cultivo_maiz_get, name='cultivo_maiz_get'),
    path('api/post/cultivos/listar/', api_post_cultivos_listar, name='api_post_cultivos_listar'),
    path('api/post/formulario_cultivo/', api_post_formulario_cultivo, name='api_post_formulario_cultivo'),
    path('api/post/formulario_cultivo/<int:cultivo_id>/', api_post_formulario_cultivo, name='api_post_formulario_cultivo_con_id'),
    path('api/post/editar_formulario_cultivo/', api_post_editar_formulario_cultivo, name='api_post_editar_formulario_cultivo'),
    path('api/post/editar_formulario_cultivo/<int:cultivo_id>/', api_post_editar_formulario_cultivo, name='api_post_editar_formulario_cultivo_con_id'),
    path('api/post/eliminar_formulario_cultivo/<int:cultivo_id>/', api_post_eliminar_formulario_cultivo, name='api_post_eliminar_formulario_cultivo_con_id'),
    path('api/post/obtener_cultivo/', obtener_cultivo, name='obtener_cultivo'),
    path('api/post/obtener_cultivo_variables_manejo/', obtener_cultivo_variables_manejo, name='obtener_cultivo_variables_manejo'),
    path('api/post/optimizar/', optimizar_cultivo, name='optimizar_cultivo'),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
    path('dash/', dash_view, name='dash_view'),
    path('signup/', signup_view, name='signup_view'),
    path('login/', login_view, name='login_view'),
    path('logout/', logout_view, name='logout_view'),
    path('clr/', clr_view, name='clr_view'),
    path('obtener_costos_unitarios/', obtener_costos_unitarios, name='obtener_costos_unitarios'),
    path('dashboard/', views.plotly_view, name='plotly_view'),
    path('api/dashboard-data/', views.dashboard_data_api, name='dashboard_data_api'),
]