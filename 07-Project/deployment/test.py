import requests
house_data = {
    "MedianIncome": 12,
    "HouseAgeYears": 60,
    "AverageRoomsPerHouse": 7,
    "AverageBedroomsPerHouse": 3,
    "TotalPopulation": 5000,
    "AverageOccupantsPerHouse": 2,
    "Latitude": 36.88,
    "Longitude": -128.23
}

#url = 'http://127.0.0.1:9696/predict'
url = "https://mlops-zoomcamp.onrender.com/predict"
response = requests.post(url, json=house_data)
print(f"Response Status Code: {response.status_code}")

predicted_value = response.json().get('HousePrice')[0]
print(predicted_value )

house_price = predicted_value * 100000
print(f"Predicted House Price: ${house_price:,.2f}")
