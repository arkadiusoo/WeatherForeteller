from django.views import View
from django.shortcuts import render, redirect
from django.views.generic import TemplateView

from apis.models import UploadedCSV, TemperatureForecast
from .forms import UploadCSVForm
from django.contrib.auth.mixins import LoginRequiredMixin

class HomePageView(LoginRequiredMixin, View):
    template_name = "pages/home.html"

    def get(self, request):
        form = UploadCSVForm()
        user_csvs = UploadedCSV.objects.filter(user=request.user).order_by('-uploaded_at')
        user_forecasts = TemperatureForecast.objects.filter().order_by('-created_at')
        for forecast in user_forecasts:
            forecast.pairs = list(zip(forecast.time_list, forecast.temperature_list, forecast.humidity_list))
        return render(request, self.template_name, {'form': form, 'user_csvs': user_csvs, 'user_forecasts' : user_forecasts})

    def post(self, request):
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            csv_instance = form.save(commit=False)
            csv_instance.user = request.user
            csv_instance.save()
            return redirect('home')

        user_csvs = UploadedCSV.objects.filter(user=request.user).order_by('-uploaded_at')
        user_forecasts = TemperatureForecast.objects.filter(user=request.user).order_by('-created_at')
        return render(request, self.template_name, {'form': form, 'user_csvs': user_csvs, 'user_forecasts' : user_forecasts})

class AboutPageView(TemplateView):
    template_name = "pages/about.html"