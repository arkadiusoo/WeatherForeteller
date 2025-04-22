import csv
import os

from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import UploadedCSV, TemperatureForecast
from rest_framework import status
# Create your views here.

class UploadCSVView(APIView):
    def post(self, request, format=None):
        file = request.FILES.get('file')
        if not file or not file.name.endswith('.csv'):
            return Response({'error': 'Only CSV files are supported.'}, status=status.HTTP_400_BAD_REQUEST)

        UploadedCSV.objects.create(user=request.user, file=file)  # zapis do bazy
        return Response({'message': f'File {file.name} uploaded successfully.'}, status=status.HTTP_200_OK)

class ListUploadedCSVView(APIView):
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
    def post(self, request):
        csv_id = request.data.get('csv_id')
        try:
            csv_file = UploadedCSV.objects.get(id=csv_id)
        except UploadedCSV.DoesNotExist:
            return Response({'error': 'CSV not found or not yours'}, status=404)

        path = os.path.join(csv_file.file.storage.location, csv_file.file.name)
        #time_list, temp_list = predict_from_csv(path)
        """
        Tutaj mamy path czyli sciezke do csv ktory mamy wrzucony juz przez siebie i przesylamy do skryptu sciezke do pliku csv z ktorego chcemy zrobic predykcje
        """
        time_list = ["19.00","20.00","21.00","22.00","23.00"]
        temp_list = [1.8,2.8,3.8,4.8,5.8]

        obj = TemperatureForecast.objects.create(
            source_type='csv',
            source_name=f'CSV #{csv_file.id}',
            time_list=time_list,
            temperature_list=temp_list
        )

        return Response({
            'id': obj.id,
            'source': obj.source_name,
            'time': time_list,
            'temperature': temp_list
        }, status=201)

class PredictFromCityView(APIView):
    def post(self, request):
        city = request.data.get('city')
        if not city:
            return Response({'error': 'City is required'}, status=400)

        """
        Tutaj mamy miasto czyli uzywajac 
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="city_locator")
        location = geolocator.geocode(city)
        location = location.latitude, location.longitude
        Mamy wspolrzedne miasta dla ktorego obliczamy temperature i teraz uzywajac tego api co szyymon w tym skrypcie mamy pogode z ostatniego czasu
        """
        #time_list, temp_list = predict_from_city(location.latitude,location.longitude)

        time_list = ["19.00","20.00","21.00","22.00","23.00"]
        temp_list = [1.8,2.8,3.8,4.8,5.8]

        obj = TemperatureForecast.objects.create(
            source_type='city',
            source_name=city,
            time_list=time_list,
            temperature_list=temp_list
        )

        return Response({
            'id': obj.id,
            'source': f'City: {city}',
            'time': time_list,
            'temperature': temp_list
        }, status=201)

class ForecastListView(APIView):
    def get(self, request):
        forecasts = TemperatureForecast.objects.all().order_by('-created_at')
        data = [
            {
                'id': f.id,
                'name': f.source_name,
                'created_at': f.created_at,
                'time': f.time_list,
                'temperature': f.temperature_list
            } for f in forecasts
        ]
        return Response(data, status=status.HTTP_200_OK)

class ForecastDetailView(APIView):
    def get(self, request, id):
        try:
            forecast = TemperatureForecast.objects.get(id=id)
        except TemperatureForecast.DoesNotExist:
            return Response({'error': 'Forecast not found'}, status=status.HTTP_404_NOT_FOUND)

        data = {
            'id': forecast.id,
            'name': forecast.source_name,
            'created_at': forecast.created_at,
            'time': forecast.time_list,
            'temperature': forecast.temperature_list
        }
        return Response(data, status=status.HTTP_200_OK)

class ForecastDownloadCSVView(APIView):
    def get(self, request, id):
        try:
            forecast = TemperatureForecast.objects.get(id=id)
        except TemperatureForecast.DoesNotExist:
            return Response({'error': 'Forecast not found'}, status=status.HTTP_404_NOT_FOUND)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename=forecast_{id}.csv'

        writer = csv.writer(response)
        writer.writerow(['Time', 'Temperature'])

        for t, temp in zip(forecast.time_list, forecast.temperature_list):
            writer.writerow([t, temp])

        return response