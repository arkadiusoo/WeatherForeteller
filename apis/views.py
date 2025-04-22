from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import UploadedCSV
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