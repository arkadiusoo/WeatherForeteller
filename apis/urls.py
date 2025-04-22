from django.urls import path

from .views import  UploadCSVView, ListUploadedCSVView

urlpatterns = [
    path('upload-csv/', UploadCSVView.as_view(), name='upload_csv'),
    path("list-csv/", ListUploadedCSVView.as_view(), name="list_uploaded_csv"),
]
