import os
import string
import cv2
#crea un fichero
DIR_DATOS = './data'
os.makedirs(DIR_DATOS, exist_ok=True)

# Carpetas creadas desde la A hasta la Z
letras = list(string.ascii_uppercase)
tamano_dataset = 100
VENTANA = 'Capturar Imagen'


def ventana_cerrada():
    # Confirmacion cuando el usuario cierra la ventana con el botón X
    return cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1


camara = cv2.VideoCapture(0)
if not camara.isOpened():
    print("No se pudo abrir la camara. Prueba con otro indice (0, 1, 2...)")
    exit()

salir = False
for letra in letras:
    if salir:
        break

    carpeta = os.path.join(DIR_DATOS, letra)
    os.makedirs(carpeta, exist_ok=True)

    print('Recolectando datos para la letra {}'.format(letra))

    # Fase de espera: prepara tu mano y presiona Q para iniciar la captura
    while True:
        exito, fotograma = camara.read()
        if not exito:
            salir = True
            break
        cv2.putText(fotograma, 'Letra {}: "Q" para capturar, ESC para salir'.format(letra), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(VENTANA, fotograma)
        tecla = cv2.waitKey(25) & 0xFF
        if tecla == ord('q'):
            break
        if tecla == 27 or ventana_cerrada():  # ESC o boton X
            salir = True
            break

    if salir:
        break

    # Fase de captura: se toman las 100 imagenes automaticamente
    contador = 0
    while contador < tamano_dataset:
        exito, fotograma = camara.read()
        if not exito:
            salir = True
            break
        cv2.imshow(VENTANA, fotograma)
        tecla = cv2.waitKey(25) & 0xFF
        if tecla == 27 or ventana_cerrada():
            salir = True
            break
        cv2.imwrite(os.path.join(carpeta, '{}.jpg'.format(contador)), fotograma)
        contador += 1

    print('Letra {}: {} imagenes guardadas en {}'.format(letra, contador, carpeta))

camara.release()
cv2.destroyAllWindows()