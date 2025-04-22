from django.db import models
from django.conf import settings

# Create your models here.
class UploadedCSV(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='uploaded_csvs', null=True, blank=True)
    file = models.FileField(upload_to='csv_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CSV uploaded at: {self.uploaded_at} - {self.user.username}"
