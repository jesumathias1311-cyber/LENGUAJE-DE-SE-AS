import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


dic_datos = pickle.load(open('./data.pickle', 'rb'))

datos = np.asarray(dic_datos['data'])
etiquetas = np.asarray(dic_datos['labels'])

x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(datos, etiquetas, test_size=0.2, shuffle=True, stratify=etiquetas)

modelo = RandomForestClassifier()

modelo.fit(x_entrenamiento, y_entrenamiento)

y_prediccion = modelo.predict(x_prueba)

precision = accuracy_score(y_prediccion, y_prueba)

print('{}% of samples were classified correctly !'.format(precision * 100))

archivo = open('model.p', 'wb')
pickle.dump({'model': modelo}, archivo)
archivo.close()