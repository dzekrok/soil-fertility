from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model dan preprocessing tools
model = load_model('models/ann_fuzzy_resample.h5')

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def get_recommendation(label):
    if label == 'S1':
        return "Tanah sangat sesuai. Pertahankan kondisi tanah dengan pemupukan seimbang."
    elif label == 'S2':
        return "Tanah cukup sesuai. Disarankan penambahan bahan organik dan pengaturan pH."
    else:
        return "Tanah marginal. Perlu perbaikan intensif seperti pengapuran dan pemupukan NPK."

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    recommendation = None

    if request.method == 'POST':
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorus'])
        K = float(request.form['kalium'])
        pH = float(request.form['ph'])
        moisture = float(request.form['moisture'])

        data = np.array([[N, P, K, pH, moisture]])
        data = scaler.transform(data)

        pred = model.predict(data)
        class_index = np.argmax(pred, axis=1)
        label = label_encoder.inverse_transform(class_index)[0]

        # Konversi kode model → teks ramah manusia
        label_map = {
            'S1': 'Sangat Baik / Sangat Sesuai',
            'S2': 'Baik / Sesuai',
            'S3': 'Buruk / Marginal'
        }

        label_text = label_map[label]

        # Yang tampil di website
        result = label_text

        # Yang dipakai logika rekomendasi
        recommendation = get_recommendation(label)

    return render_template('index.html', result=result, recommendation=recommendation)

import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

