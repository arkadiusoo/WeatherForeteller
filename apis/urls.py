from django.urls import path

from .views import UploadCSVView, ListUploadedCSVView, PredictFromCSVView, PredictFromCityView, ForecastListView, \
    ForecastDetailView, ForecastDownloadCSVView

urlpatterns = [
    path('upload-csv/', UploadCSVView.as_view(), name='upload_csv'),
    path("list-csv/", ListUploadedCSVView.as_view(), name="list_uploaded_csv"),
    path("predict/csv/", PredictFromCSVView.as_view(), name="predict_csv"),
    path("predict/city/", PredictFromCityView.as_view(), name="predict_city"),
    path("forecasts/", ForecastListView.as_view(), name="forecast_list"),
    path("forecasts/<int:id>/", ForecastDetailView.as_view(), name="forecast_detail"),
    path("forecasts/<int:id>/download/", ForecastDownloadCSVView.as_view(), name="forecast_download"),
]
