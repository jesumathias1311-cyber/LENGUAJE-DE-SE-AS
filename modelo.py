import pickle
import cv2
import mediapipe as mp
import numpy as np

dic_modelo = pickle.load(open('./model.p', 'rb'))
modelo = dic_modelo['model']

camara = cv2.VideoCapture(0)
if not camara.isOpened():
    print("No se pudo abrir la camara. Prueba con otro indice (0, 1, 2...)")
    exit()

mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
mp_estilo_dibujo = mp.solutions.drawing_styles

manos = mp_manos.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.3)

UMBRAL = 0.20

while True:
    datos_aux = []
    lista_x = []
    lista_y = []

    exito, fotograma = camara.read()
    if not exito:
        break
    alto, ancho, _ = fotograma.shape

    fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)

    resultados = manos.process(fotograma_rgb)
    if resultados.multi_hand_landmarks:
        for marcas_mano in resultados.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(
                fotograma,
                marcas_mano,
                mp_manos.HAND_CONNECTIONS,
                mp_estilo_dibujo.get_default_hand_landmarks_style(),
                mp_estilo_dibujo.get_default_hand_connections_style())

        for marcas_mano in resultados.multi_hand_landmarks:
            for i in range(len(marcas_mano.landmark)):
                lista_x.append(marcas_mano.landmark[i].x)
                lista_y.append(marcas_mano.landmark[i].y)

            for i in range(len(marcas_mano.landmark)):
                datos_aux.append(marcas_mano.landmark[i].x - min(lista_x))
                datos_aux.append(marcas_mano.landmark[i].y - min(lista_y))

        # Cuadro alrededor de la mano (con margen de 10 px por cada lado)
        x1 = int(min(lista_x) * ancho) - 10
        y1 = int(min(lista_y) * alto) - 10
        x2 = int(max(lista_x) * ancho) + 10
        y2 = int(max(lista_y) * alto) + 10

        # Probabilidad que el modelo asigna a cada letra
        probabilidades = modelo.predict_proba([np.asarray(datos_aux)])[0]
        indice = int(np.argmax(probabilidades))
        confianza = probabilidades[indice]
        caracter_predicho = modelo.classes_[indice]

        # Solo se muestra la letra si el modelo esta suficientemente seguro
        if confianza >= UMBRAL:
            cv2.rectangle(fotograma, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(fotograma, '{} ({:.0f}%)'.format(caracter_predicho, confianza * 100), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', fotograma)
    cv2.waitKey(1)

camara.release()
cv2.destroyAllWindows()