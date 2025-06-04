import csv
import os
from datetime import datetime

from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse, OpenApiExample

from .models import UploadedCSV, TemperatureForecast

from .utils import predict_from_csv, getCityData


class UploadCSVView(APIView):
    @extend_schema(
        tags=["Forecasts"],
        request={
            "multipart/form-data": {"type": "object", "properties": {"file": {"type": "string", "format": "binary"}}}},
        responses={
            200: OpenApiResponse(description="CSV uploaded successfully."),
            400: OpenApiResponse(description="Only CSV files are supported.")
        },
        description="Upload a CSV file containing weather data. Only `.csv` files are accepted."
    )
    def post(self, request, format=None):
        file = request.FILES.get('file')
        if not file or not file.name.endswith('.csv'):
            return Response({'error': 'Only CSV files are supported.'}, status=status.HTTP_400_BAD_REQUEST)

        UploadedCSV.objects.create(file=file, user=request.user)
        return Response({'message': f'File {file.name} uploaded successfully.'}, status=status.HTTP_200_OK)


class ListUploadedCSVView(APIView):
    @extend_schema(
        tags=["Forecasts"],
        responses={
            200: OpenApiResponse(description="List of CSV files uploaded by the user"),
        },
        description="Get a list of CSV files uploaded by the authenticated user."
    )
    def get(self, request, format=None):
        uploaded_files = UploadedCSV.objects.filter(user=request.user).order_by('-uploaded_at')
        data = [
            {
                'id': f.id,
                'filename': f.file.name,
                'uploaded_at': f.uploaded_at
            }
            for f in uploaded_files
        ]
        return Response({'csv_files': data}, status=status.HTTP_200_OK)


class PredictFromCSVView(APIView):
    @extend_schema(
        tags=["Forecasts"],
        request={"application/json": {"type": "object", "properties": {"csv_id": {"type": "integer"}}}},
        responses={
            201: OpenApiResponse(description="Forecast generated from uploaded CSV file."),
            404: OpenApiResponse(description="CSV not found or not yours."),
        },
        description="Generate temperature prediction based on a previously uploaded CSV file."
    )
    def post(self, request):
        csv_id = request.data.get('csv_id')
        try:
            csv_file = UploadedCSV.objects.get(id=csv_id, user=request.user)
        except UploadedCSV.DoesNotExist:
            return Response({'error': 'CSV not found or not yours'}, status=404)

        path = os.path.join(csv_file.file.storage.location, csv_file.file.name)

        time_list, temp_list, hum_list, rain_list = predict_from_csv(path)

        obj = TemperatureForecast.objects.create(
            user=request.user,
            source_type='csv',
            source_name=f'{csv_file.file.name} Forecast',
            time_list=time_list,
            temperature_list=temp_list,
            humidity_list=hum_list,
            rain_list=rain_list
        )

        return Response({
            'id': obj.id,
            'source': obj.source_name,
            'time': time_list,
            'temperature': temp_list,
            'humidity': hum_list,
            'rain': rain_list
        }, status=201)


class PredictFromCityView(APIView):
    @extend_schema(
        tags=["Forecasts"],
        request={"application/json": {"type": "object", "properties": {"city": {"type": "string"}}}},
        responses={
            201: OpenApiResponse(description="Forecast generated based on city name."),
            400: OpenApiResponse(description="City is required.")
        },
        description="Generate temperature forecast using city name (location-based weather prediction)."
    )
    def post(self, request):
        city = request.data.get('city')
        if not city:
            return Response({'error': 'City is required'}, status=400)

        time_list, temp_list, hum_list, rain_list = getCityData(city)

        obj = TemperatureForecast.objects.create(
            user=request.user,
            source_type='city',
            source_name=f'{city} Forecast',
            time_list=time_list,
            temperature_list=temp_list,
            humidity_list=hum_list,
            rain_list=rain_list
        )

        return Response({
            'id': obj.id,
            'source': f'City: {city}',
            'time': time_list,
            'temperature': temp_list,
            'humidity': hum_list,
            'rain': rain_list
        }, status=201)


class ForecastListView(APIView):
    @extend_schema(
        tags=["Forecasts"],
        responses={
            200: OpenApiResponse(description="List of all forecasts.")
        },
        description="Retrieve a list of all generated temperature forecasts."
    )
    def get(self, request):
        forecasts = TemperatureForecast.objects.filter(user=request.user).order_by('-created_at')
        data = [
            {
                'id': f.id,
                'name': f.source_name,
                'created_at': f.created_at,
                'time': f.time_list,
                'temperature': f.temperature_list,
                'humidity': f.humidity_list,
                'rain': f.rain_list
            } for f in forecasts
        ]
        return Response(data, status=status.HTTP_200_OK)


class ForecastDetailView(APIView):
    @extend_schema(
        tags=["Forecasts"],
        parameters=[OpenApiParameter("id", int, OpenApiParameter.PATH)],
        responses={
            200: OpenApiResponse(description="Details of a specific forecast."),
            404: OpenApiResponse(description="Forecast not found.")
        },
        description="Get detailed data of a specific forecast by ID."
    )
    def get(self, request, id):
        try:
            forecast = TemperatureForecast.objects.get(id=id, user=request.user)
        except TemperatureForecast.DoesNotExist:
            return Response({'error': 'Forecast not found'}, status=status.HTTP_404_NOT_FOUND)

        data = {
            'id': forecast.id,
            'name': forecast.source_name,
            'created_at': forecast.created_at,
            'time': forecast.time_list,
            'temperature': forecast.temperature_list,
            'humidity': forecast.humidity_list,
            'rain': forecast.rain_list
        }
        return Response(data, status=status.HTTP_200_OK)


class ForecastDownloadCSVView(APIView):
    @extend_schema(
        tags=["Forecasts"],
        parameters=[OpenApiParameter("id", int, OpenApiParameter.PATH)],
        responses={
            200: OpenApiResponse(description="CSV file with forecast data."),
            404: OpenApiResponse(description="Forecast not found.")
        },
        description="Download the forecast as a CSV file."
    )
    def get(self, request, id):
        try:
            forecast = TemperatureForecast.objects.get(id=id, user=request.user)
        except TemperatureForecast.DoesNotExist:
            return Response({'error': 'Forecast not found'}, status=status.HTTP_404_NOT_FOUND)

        response = HttpResponse(content_type='text/csv')
        timestamp = datetime.now().strftime("%d_%m_%Y_%H:%M")
        if forecast.source_type == 'city':
            response['Content-Disposition'] = f'attachment; filename={forecast.source_name}.csv'
        else:
            response['Content-Disposition'] = f'attachment; filename=forecast {timestamp}.csv'

        writer = csv.writer(response)
        writer.writerow(['Time', 'Temperature', 'Humidity', 'Rain'])

        for t, temp, hum, rain in zip(forecast.time_list, forecast.temperature_list, forecast.humidity_list, forecast.rain_list):
            writer.writerow([t, temp, hum, rain])

        return response