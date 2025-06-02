from django import template
import os

register = template.Library()

@register.filter
def clean_after_last_underscore(value):
    base = os.path.basename(value)
    name, ext = os.path.splitext(base)
    if "_" in name:
        name = "_".join(name.split("_")[:-1])
    return f"{name}{ext}"
