from django.apps import AppConfig


class CultivoMaizConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myapp'
    label = 'myapp'

    def ready(self):
        import myapp.dash_apps.hello_dash

