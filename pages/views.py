from django.views import View
from django.shortcuts import render, redirect
from django.views.generic import TemplateView

from apis.models import UploadedCSV
from .forms import UploadCSVForm
from django.contrib.auth.mixins import LoginRequiredMixin

class HomePageView(LoginRequiredMixin, View):
    template_name = "pages/home.html"

    def get(self, request):
        form = UploadCSVForm()
        user_csvs = UploadedCSV.objects.filter(user=request.user).order_by('-uploaded_at')
        return render(request, self.template_name, {'form': form, 'user_csvs': user_csvs})

    def post(self, request):
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            csv_instance = form.save(commit=False)
            csv_instance.user = request.user
            csv_instance.save()
            return redirect('home')

        user_csvs = UploadedCSV.objects.filter(user=request.user).order_by('-uploaded_at')
        return render(request, self.template_name, {'form': form, 'user_csvs': user_csvs})

class AboutPageView(TemplateView):
    template_name = "pages/about.html"