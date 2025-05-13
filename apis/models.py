from django.db import models
from django.conf import settings

# Create your models here.
class UploadedCSV(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='uploaded_csvs', null=True, blank=True)
    file = models.FileField(upload_to='csv_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CSV uploaded at: {self.uploaded_at} - {self.user.username}"

class TemperatureForecast(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='temperature_forecasts', null=True, blank=True)
    source_type = models.CharField(max_length=20, choices=[('csv', 'CSV'), ('city', 'City')])
    source_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    time_list = models.JSONField()
    temperature_list = models.JSONField()

    def __str__(self):
        return f"{self.source_type.upper()} Forecast for {self.source_name} at {self.created_at.strftime('%Y-%m-%d %H:%M')}"