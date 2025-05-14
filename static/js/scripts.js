document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM wczytany');

    const csvButtons = document.querySelectorAll('.generate-forecast-csv-btn');
    csvButtons.forEach(button => {
        button.addEventListener('click', function () {
            const csvId = this.getAttribute('data-csv-id');
            console.log('Kliknięto Generuj CSV dla id:', csvId);

            this.disabled = true;
            this.textContent = 'Generowanie...';

            fetch('/api/predict/csv/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: JSON.stringify({ csv_id: csvId })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Błąd sieci podczas generowania z CSV');
                }
                return response.json();
            })
            .then(data => {
                console.log('Sukces generowania z CSV:', data);
                location.reload();
            })
            .catch(error => {
                console.error('Błąd:', error);
                this.disabled = false;
                this.textContent = 'Generuj';
                alert('Błąd podczas generowania prognozy z pliku CSV.');
            });
        });
    });

    // Obsługa kliknięcia "Generuj" dla miasta
    const cityButton = document.getElementById('generate-city-btn');
    if (cityButton) {
        cityButton.addEventListener('click', function () {
            const cityName = document.getElementById('city-name-input').value;
            console.log('Kliknięto Generuj dla miasta:', cityName);

            if (!cityName.trim()) {
                alert('Podaj nazwę miasta!');
                return;
            }

            cityButton.disabled = true;
            cityButton.textContent = 'Generowanie...';

            fetch('/api/predict/city/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: JSON.stringify({ city: cityName })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Błąd sieci podczas generowania z miasta');
                }
                return response.json();
            })
            .then(data => {
                console.log('Sukces generowania z miasta:', data);
                location.reload();
            })
            .catch(error => {
                console.error('Błąd:', error);
                cityButton.disabled = false;
                cityButton.textContent = 'Generuj';
                alert('Błąd podczas generowania prognozy dla miasta.');
            });
        });
    }

    // Funkcja do pobierania CSRF tokena (wymagana dla Django)
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});


document.addEventListener('DOMContentLoaded', function() {
  const detailButtons = document.querySelectorAll('.toggle-details-btn');

  detailButtons.forEach(function(button) {
    button.addEventListener('click', function() {
      const forecastId = button.getAttribute('data-forecast-id');
      const detailsDiv = document.getElementById(`details-${forecastId}`);

      if (detailsDiv.style.display === 'none') {
        detailsDiv.style.display = 'block';
        button.textContent = 'Hide';
      } else {
        detailsDiv.style.display = 'none';
        button.textContent = 'Details';
      }
    });
  });
});

document.addEventListener('DOMContentLoaded', function() {
  const sliders = document.querySelectorAll('.forecast-slider');

  sliders.forEach(slider => {
    const forecastId = slider.id.split('-')[1];
    const slides = slider.querySelectorAll('.forecast-slide');
    const prevButton = document.querySelector(`.prev-slide[data-forecast-id="${forecastId}"]`);
    const nextButton = document.querySelector(`.next-slide[data-forecast-id="${forecastId}"]`);
    let currentIndex = 0;

    function updateSlides() {
      slides.forEach((slide, index) => {
        slide.style.display = (index === currentIndex) ? 'block' : 'none';
      });
      prevButton.disabled = (currentIndex === 0);
      nextButton.disabled = (currentIndex === slides.length - 1);
    }

    prevButton.addEventListener('click', () => {
      if (currentIndex > 0) {
        currentIndex--;
        updateSlides();
      }
    });

    nextButton.addEventListener('click', () => {
      if (currentIndex < slides.length - 1) {
        currentIndex++;
        updateSlides();
      }
    });
  });
});