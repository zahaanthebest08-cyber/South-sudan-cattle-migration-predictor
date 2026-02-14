# South-sudan-cattle-migration-predictor
This project is a cattle migration predictor designed for South Sudan. It tracks and predicts the movement of cattle across the region using a machine learning model and interactive maps. The tool can help researchers, herders, and officials understand and anticipate cattle migration patterns.
Features

Interactive Map – Click anywhere in South Sudan to predict cattle movement.

Animated Migration Paths – See predicted routes with start and end markers.

Data-Driven Predictions – Uses a dataset of past cattle movements along with environmental features like NDVI (vegetation index), water availability, and elevation.

Machine Learning Model – A simple LSTM (Long Short-Term Memory) neural network predicts the next positions based on recent locations and environmental data.
How It Works

Dataset – cattle_data.csv contains historical cattle positions (lat, lon) and environmental features (ndvi, water, elev), along with the next position (lat_next, lon_next).

Training – The LSTM model learns patterns from the dataset to predict future locations.

Prediction – When a user clicks on the map, the app predicts the next several positions of the cattle, showing the likely migration path.

Visualization – The predicted path is displayed on a Leaflet map with animated lines and markers for start and end points.
Requirements

Python 3.x

Flask

Pandas

NumPy

TensorFlow / Keras

Leaflet.js (included in the HTML)

Install Python packages with:

pip install -r requirements.txt
How to Run

Clone the repository:

git clone <your-repo-link>
cd South-Sudan-Cattle-Migration-Predictor


Run the Flask app:

python tksapp.py


Open your browser at http://127.0.0.1:5000/

Click on the map to see predicted cattle migration paths.

Notes

The dataset is simulated for demonstration purposes but reflects realistic movement and environmental patterns.

Predictions are meant for educational and research purposes.

License

This project is licensed under the MIT License.
