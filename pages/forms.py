from django import forms
from django.utils.translation import gettext_lazy as _
from apis.models import UploadedCSV

class UploadCSVForm(forms.ModelForm):
    class Meta:
        model = UploadedCSV
        fields = ['file']
        labels = {
            'file': _("Select CSV file"),
        }