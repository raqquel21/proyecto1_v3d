import cv2
import numpy as np
import os

# --- CONFIGURACIÓN CÁMARA ---
IP_CAM = "192.168.2.104"
URL = f"http://{IP_CAM}:8080/video"
cap = cv2.VideoCapture(URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Color verde para el tracking
lower_color = np.array([35, 70, 70])
upper_color = np.array([85, 255, 255])

# --- CARGAR IMAGEN DEL BUSCAMINAS ---
img_buscaminas = cv2.imread('buscaminas.png')
if img_buscaminas is None:
    # Si no encuentra la imagen, crea un fondo gris
    img_buscaminas = np.full((480, 640, 3), 100, dtype=np.uint8)
    print("Imagen no encontrada, usando fondo gris.")
else:
    img_buscaminas = cv2.resize(img_buscaminas, (640, 480))


def dibujar_gemelo_digital(cx, cy):
    """ Dibuja una representación 3D del objeto detectado en el mundo virtual. """
    # 1. Crear lienzo negro para el mundo virtual
    virtual_world = img_buscaminas.copy()

    # 2. Simular perspectiva 3D (Puntos de fuga)
    # Convertimos las coordenadas (cx, cy) de la cámara a la posición de la esfera
    # Añadimos un pequeño desplazamiento para que parezca que "flota" en 3D
    offset_x = int((cx - 320) * 0.1)
    offset_y = int((cy - 240) * 0.1)

    # Dibujar la "Sombra" del objeto real en el mundo virtual (Efecto Gemelo)
    cv2.circle(virtual_world, (cx, cy), 15, (50, 50, 50), -1)  # Sombra gris

    # Dibujar el Objeto Real Virtualizado (Esfera Roja) con efecto de brillo
    virt_x, virt_y = cx + offset_x, cy + offset_y
    cv2.circle(virtual_world, (virt_x, virt_y), 20, (0, 0, 255), -1)  # Cuerpo
    cv2.circle(virtual_world, (virt_x - 5, virt_y - 5), 5, (200, 200, 255), -1)  # Brillo

    # Añadir líneas de ejes 3D para que parezca un entorno de simulación
    cv2.line(virtual_world, (0, 480), (virt_x, virt_y), (0, 255, 0), 1)  # Eje X
    cv2.line(virtual_world, (640, 480), (virt_x, virt_y), (0, 255, 0), 1)  # Eje Y

    return virtual_world


print("Iniciando Sistema de Gemelo Digital...")
print("Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al recibir video.")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    # Procesamiento de color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.dilate(mask, None, iterations=2)

    # Inicializar gemelo con el fondo estático por si no hay detección
    gemelo_digital = img_buscaminas.copy()

    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # Dibujar en cámara real
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
        cv2.putText(frame, "OBJETO REAL", (cx + 15, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Crear la escena del Gemelo Digital
        gemelo_digital = dibujar_gemelo_digital(cx, cy)

    # --- PANTALLA DIVIDIDA ---
    # Unimos ambos mundos en una sola ventana
    resultado = np.hstack((frame, gemelo_digital))

    # Etiquetas
    cv2.rectangle(resultado, (0, 0), (1280, 40), (30, 30, 30), -1)
    cv2.putText(resultado, "MUNDO REAL (CAPTURA)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(resultado, "GEMELO DIGITAL (ESCENA VIRTUAL 3D)", (650, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("PROYECTO: GEMELO DIGITAL", resultado)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()