from django import template

register = template.Library()

@register.filter
def render_disabled(field):
    return field.as_widget(attrs={'disabled': 'disabled'})
