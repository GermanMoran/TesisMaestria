# myapp/templatetags/my_filters.py
from django import template

register = template.Library()

@register.filter
def get_field_value(obj, field_name):
    """Devuelve el valor del atributo 'field_name' del objeto 'obj'."""
    return getattr(obj, field_name)
