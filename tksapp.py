import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from flask import Flask, render_template_string, request, jsonify

# -----------------------------
# Load real cattle dataset
# -----------------------------
data = pd.read_csv(r"C:\Users\sarah\cattle_data.csv")

# Features and targets
features = ['lat', 'lon', 'ndvi', 'water', 'elev']
targets = ['lat_next', 'lon_next']
X = data[features].values
y = data[targets].values

# -----------------------------
# Create sequences for LSTM
# -----------------------------
timesteps = 5
X_seq = []
y_seq = []

for i in range(len(X) - timesteps):
    X_seq.append(X[i:i+timesteps])
    y_seq.append(y[i+timesteps-1])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("X_seq shape:", X_seq.shape)
print("y_seq shape:", y_seq.shape)

# -----------------------------
# Train LSTM model
# -----------------------------
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(timesteps, X_seq.shape[2])),
    Dense(32, activation='relu'),
    Dense(2)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_seq, y_seq, epochs=50, batch_size=16, verbose=1)

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Cattle Migration Predictor</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/leaflet-ant-path@1.3.0/dist/leaflet-ant-path.min.js"></script>
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<style>
  #map { height: 90vh; width: 100%; }
</style>
</head>
<body>
<h2>Click anywhere in South Sudan to predict cattle migration</h2>
<div id="map"></div>

<script>
var map = L.map('map').setView([7.0, 31.0], 7);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

var lastPath = null;
var startMarker = null;
var endMarker = null;

map.on('click', function(e) {
    $.post("/predict", {lat: e.latlng.lat, lon: e.latlng.lng}, function(data) {
        if(lastPath) map.removeLayer(lastPath);
        if(startMarker) map.removeLayer(startMarker);
        if(endMarker) map.removeLayer(endMarker);

        lastPath = L.polyline.antPath(data.path, {color:'red', weight:5, delay:200, opacity:0.9}).addTo(map);

        startMarker = L.marker(data.path[0], {icon:L.icon({iconUrl:'https://maps.google.com/mapfiles/ms/icons/green-dot.png',iconSize:[32,32]})}).addTo(map).bindPopup('Start');
        endMarker = L.marker(data.path[data.path.length-1], {icon:L.icon({iconUrl:'https://maps.google.com/mapfiles/ms/icons/red-dot.png',iconSize:[32,32]})}).addTo(map).bindPopup('Predicted End');
    });
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    lat = float(request.form.get("lat"))
    lon = float(request.form.get("lon"))

    # Build initial input sequence with default features
    default_ndvi = 0.5
    default_water = 0.5
    default_elev = 0.5

    seq = np.array([[lat, lon, default_ndvi, default_water, default_elev]] * timesteps).reshape(1, timesteps, 5)
    path = [[lat, lon]]

    for _ in range(8):
        pred = model.predict(seq, verbose=0)[0]

        # Convert to regular floats for JSON
        lat_pred = float(min(max(pred[0], 3.5), 12.5))
        lon_pred = float(min(max(pred[1], 24.0), 36.0))
        path.append([lat_pred, lon_pred])

        # Update sequence
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, :2] = [lat_pred, lon_pred]

    return jsonify({"path": path})

if __name__ == "__main__":
    app.run(debug=True)
