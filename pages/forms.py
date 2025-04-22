from django import forms
from apis.models import UploadedCSV

class UploadCSVForm(forms.ModelForm):
    class Meta:
        model = UploadedCSV
        fields = ['file']