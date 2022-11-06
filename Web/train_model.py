import pipeline
import pandas as pd
import pickle

data = pd.read_csv('Stars_original.csv')
X = data.drop('Type',axis = 1)
y = data['Type']
star_predict = pipeline.Star_Prediction()
star_predict.fit(X,y)
pickle.dump(star_predict, open('star_predict.pkl', 'wb'))
X_predict = [30003,0.2400,0.10,16.12,"White","M"]
data_for_prediction = pd.DataFrame([X_predict], columns = ['Temperature','L','R','A_M','Color','Spectral_Class'])
print(star_predict.predict(data_for_prediction))

