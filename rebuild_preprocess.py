import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset asli
df = pd.read_csv('data/datasetfitroh_labeled.csv')

X = df[['Nitrogen','pHospHorus','Kalium','pH Tanah','Soil Moisture']]
y = df['Label Fuzzy']   # kolom S1 / S2 / S3

# Rebuild scaler
scaler = StandardScaler()
scaler.fit(X)

# Rebuild label encoder
le = LabelEncoder()
le.fit(y)

# Simpan ulang
with open('models/scaler.pkl','wb') as f:
    pickle.dump(scaler,f)

with open('models/label_encoder.pkl','wb') as f:
    pickle.dump(le,f)

print("Scaler & encoder rebuilt successfully.")
