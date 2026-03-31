import cv2
import numpy as np
import socket
import os

# Configuración UDP:
#   Para enviar la posición del objeto detectado (cintas)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_ip = "127.0.0.1"
udp_port = 5005

# Configuración para el móvil
URL = "http://192.168.93.202:4747/video"
H_FILE = "homografia_mesa.npz"  # Archivo generado por el otro código
cap = cv2.VideoCapture(URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Cargar homografía
mask_mesa = None
if os.path.exists(H_FILE):
    data = np.load(H_FILE)
    H_w2i = data['H_w2i']

    # Creamos la máscara basada en los límites de la mesa (0,0 a 30,20 cm)
    mask_mesa = np.zeros((480, 640), dtype=np.uint8)
    # Definimos las esquinas de la mesa en el mundo real (según los WORLD_POINTS)
    esquinas_mundo = np.array([[0, 0], [30, 0], [30, 20], [0, 20]], dtype=np.float32).reshape(-1, 1, 2)
    # Proyectamos a píxeles
    pts_pixel = cv2.perspectiveTransform(esquinas_mundo, H_w2i).astype(np.int32)
    cv2.fillPoly(mask_mesa, [pts_pixel], 255)
    print("Homografía cargada: Filtro de mesa activado.")
else:
    # Nos aseguramos que la cámara podrá funcionar igualmente
    print("ADVERTENCIA: No se encontró 'homografia_mesa.npz'. Se usará toda la imagen.")

# --- RANGOS DE COLOR ---
verde_lower = np.array([35, 70, 70])
verde_upper = np.array([85, 255, 255])
morado_lower = np.array([130, 50, 50])
morado_upper = np.array([170, 255, 255])

print("Iniciando Mando Bi-Color con Máscara de Mesa. ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))

    # Aplicar la máscara de la mesa
    if mask_mesa is not None:
        frame_mesa = cv2.bitwise_and(frame, frame, mask=mask_mesa)
    else:
        frame_mesa = frame.copy()

    hsv = cv2.cvtColor(frame_mesa, cv2.COLOR_BGR2HSV)

    # 1. Crear Máscaras
    mask_verde = cv2.inRange(hsv, verde_lower, verde_upper)
    mask_morado = cv2.inRange(hsv, morado_lower, morado_upper)

    kernel = np.ones((5, 5), np.uint8)
    mask_verde = cv2.morphologyEx(mask_verde, cv2.MORPH_OPEN, kernel)
    mask_morado = cv2.morphologyEx(mask_morado, cv2.MORPH_OPEN, kernel)

    # 2. Detectar presencia
    hay_verde = cv2.countNonZero(mask_verde) > 200
    hay_morado = cv2.countNonZero(mask_morado) > 500

    estado = "REPOSO"
    color_puntero = (255, 255, 255)

    if hay_morado:
        M = cv2.moments(mask_morado)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if hay_verde:
            estado, color_puntero, id_estado = "MOVIENDO", (0, 255, 0), 1
        else:
            estado, color_puntero, id_estado = "CLICK", (0, 0, 255), 2

        cv2.circle(frame, (cx, cy), 15, color_puntero, -1)
        data = f"{cx},{cy},640,480,{id_estado}"
        sock.sendto(data.encode(), (udp_ip, udp_port))

    # Visualización
    mask_combinada = cv2.bitwise_or(mask_verde, mask_morado)
    mask_bgr = cv2.cvtColor(mask_combinada, cv2.COLOR_GRAY2BGR)

    # Dibujamos el contorno de la mesa para saber dónde detecta (para que se vea más visual)
    if mask_mesa is not None:
        cv2.polylines(frame, [pts_pixel], True, (255, 255, 0), 2)

    visor = np.hstack((frame, mask_bgr))
    cv2.putText(visor, f"ESTADO: {estado}", (20, 50), 1, 2, color_puntero, 3)

    cv2.imshow("Control por Color (Doble Cinta + Mesa)", visor)

    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()