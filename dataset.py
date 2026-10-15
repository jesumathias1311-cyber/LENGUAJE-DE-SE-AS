#crear el dataset
import pickle
import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
mp_estilo_dibujo = mp.solutions.drawing_styles

manos = mp_manos.Hands(static_image_mode=True, min_detection_confidence=0.3)

DIR_DATOS = './data'

datos = []
etiquetas = []
for carpeta in os.listdir(DIR_DATOS):
    for ruta_imagen in os.listdir(os.path.join(DIR_DATOS, carpeta)):
        datos_aux = []
        lista_x =[]
        lista_y =[]

        imagen = cv2.imread(os.path.join(DIR_DATOS, carpeta, ruta_imagen))
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        resultado = manos.process(imagen_rgb)

        # Si no detecta ninguna mano, el resultado es None
        if resultado.multi_hand_landmarks:
            for marcas_mano in resultado.multi_hand_landmarks:
                for i in range(len(marcas_mano.landmark)):
                    x = marcas_mano.landmark[i].x
                    y = marcas_mano.landmark[i].y

                    lista_x.append(x)
                    lista_y.append(y)

                for i in range(len(marcas_mano.landmark)):
                    x = marcas_mano.landmark[i].x
                    y = marcas_mano.landmark[i].y
                    datos_aux.append(x - min(lista_x))
                    datos_aux.append(y - min(lista_y))

            datos.append(datos_aux)
            etiquetas.append(carpeta)
archivo = open('data.pickle', 'wb')
pickle.dump({'data': datos, 'labels': etiquetas}, archivo)
archivo.close()