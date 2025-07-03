# 🌦️ WeatherForeteller

**WeatherForeteller** is a full-stack web application that allows users to upload weather data (CSV format), receive AI-powered forecasts, and visualize the results using a modern interface. It combines a Django-based backend, a server-rendered HTML/CSS/JavaScript frontend, and a machine learning model trained on real datasets.

---

## 🎯 Project Goal

To deliver accurate, hourly weather forecasts based on uploaded CSV data, while providing users with an intuitive and responsive interface for visualization, history tracking, and secure authentication.

---

## ✅ Core Components

### 📦 Backend – Django (Python)

The backend is built with Django and Django REST Framework to handle data processing, user management, and model inference.

**Features:**

- Project setup and SQLite integration
- Secure user authentication using token-based auth (JWT or Django Allauth)
- Storage of user accounts and forecast history
- REST API endpoints:
  - `POST /api/upload` – upload CSV weather data
  - `POST /api/predict` – return AI-generated forecast
  - `GET /api/history` – fetch previous forecast results
- Prediction results returned in CSV format

---

### 🌤️ Machine Learning – Forecasting Engine

A core component is a trained model capable of generating hourly weather forecasts from structured input.

**Features:**

- Trained on selected Kaggle datasets
- Produces predictions in 1-hour intervals
- Connected directly to the backend `/api/predict` endpoint
- Outputs results formatted for CSV and visual display

---

### 💻 Frontend – HTML/CSS/JavaScript (server-rendered)

The frontend is built using classic HTML templates, styled with CSS, and enhanced with JavaScript.

**Features:**

- User login and registration pages
- CSV file upload interface with visual feedback
- Forecast charts mimicking Google Weather
- Option to download predictions as CSV
- History view of past forecasts (with download link)
- API communication via JavaScript (AJAX/fetch)

---

### 🔒 User Management

The system ensures that only authenticated users can interact with forecasting features.

**Features:**

- Secure registration and login (JWT-based or session)
- Authenticated access to prediction tools
- Personalized forecast history per user

---

## 🧪 Technologies Used

- **Backend:** Python 3, Django, Django REST Framework, SQLite
- **Frontend:** HTML, CSS, JavaScript (vanilla), Django templates
- **ML & Prediction:** Pandas, Scikit-learn or TensorFlow
- **Auth:** JWT or Django Allauth
- **API Communication:** REST endpoints, JavaScript fetch
- **Data Format:** CSV (upload and download)

---

<sub>🇵🇱 Wersja polska poniżej</sub>

---

# 🌦️ WeatherForeteller

**WeatherForeteller** to pełnostosowa aplikacja internetowa, która umożliwia użytkownikom przesyłanie danych pogodowych w formacie CSV, otrzymywanie prognoz generowanych przez sztuczną inteligencję oraz wizualizację wyników w nowoczesnym interfejsie. System łączy backend oparty na Django, klasyczny frontend (HTML, CSS, JavaScript) renderowany po stronie serwera oraz model uczenia maszynowego wytrenowany na rzeczywistych danych.

---

## 🎯 Cel projektu

Celem projektu jest dostarczanie dokładnych prognoz pogody w interwałach godzinowych na podstawie przesyłanych plików CSV, przy zachowaniu prostego i przejrzystego interfejsu użytkownika do wizualizacji, historii oraz bezpiecznej autoryzacji.

---

## ✅ Główne komponenty

### 📦 Backend – Django (Python)

Backend zbudowano przy użyciu Django i Django REST Framework, odpowiada on za przetwarzanie danych, uwierzytelnianie oraz komunikację z modelem ML.

**Funkcje:**

- Konfiguracja projektu oraz integracja z bazą SQLite
- Bezpieczne logowanie użytkowników (JWT lub Django Allauth)
- Przechowywanie kont użytkowników i historii prognoz
- Endpointy REST API:
  - `POST /api/upload` – przesyłanie danych pogodowych (CSV)
  - `POST /api/predict` – otrzymanie prognozy wygenerowanej przez AI
  - `GET /api/history` – pobranie historii wcześniejszych prognoz
- Wyniki zwracane w formacie CSV

---

### 🌤️ Uczenie maszynowe – silnik prognozowania

Główny komponent odpowiedzialny za generowanie godzinowych prognoz na podstawie danych wejściowych.

**Funkcje:**

- Model wytrenowany na wybranych zbiorach danych z Kaggle
- Prognozy w interwałach jednogodzinnych
- Bezpośrednia integracja z endpointem `/api/predict`
- Wyniki przystosowane do pobrania i wizualizacji (CSV)

---

### 💻 Frontend – HTML/CSS/JavaScript (renderowany przez Django)

Frontend korzysta z klasycznych szablonów HTML, stylowany jest CSS-em, a interaktywność zapewnia JavaScript.

**Funkcje:**

- Widoki logowania i rejestracji użytkownika
- Formularz przesyłania CSV z informacją o postępie
- Wizualizacja prognoz na wykresie (w stylu Google Weather)
- Możliwość pobrania prognoz w formacie CSV
- Widok historii prognoz z możliwością pobierania
- Komunikacja z backendem za pomocą JavaScript (fetch/AJAX)

---

### 🔒 Zarządzanie użytkownikami

Tylko zalogowani użytkownicy mogą korzystać z funkcji prognozowania.

**Funkcje:**

- Bezpieczna rejestracja i logowanie (JWT lub sesja)
- Dostęp do funkcji prognozowania tylko dla zalogowanych
- Historia prognoz przypisana do konkretnego użytkownika

---

## 🧪 Wykorzystane technologie

- **Backend:** Python 3, Django, Django REST Framework, SQLite
- **Frontend:** HTML, CSS, JavaScript (vanilla), szablony Django
- **ML i prognozowanie:** Pandas, Scikit-learn lub TensorFlow
- **Uwierzytelnianie:** JWT lub Django Allauth
- **Komunikacja API:** REST + fetch w JavaScript
- **Format danych:** CSV (upload i download)

