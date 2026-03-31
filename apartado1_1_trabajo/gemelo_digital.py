import cv2
import numpy as np
import socket
import os

# ---  CONFIGURACIÓN UDP (se busca reducir latencia) ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_ip = "127.0.0.1"
udp_port = 5005

# --- CONFIGURACIÓN CÁMARA IP ---
IP_CAM = "192.168.2.107"
URL = f"http://{IP_CAM}:8080/video"
cap = cv2.VideoCapture(URL)
# Intentar reducir buffer
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# --- CARGAR CALIBRACIÓN DE CÁMARA ---
calibration_file = "camera_calibration.npz"
mtx, dist = None, None
if os.path.exists(calibration_file):
    try:
        with np.load(calibration_file) as data:
            mtx, dist = data['mtx'], data['dist']
        print("Calibración cargada exitosamente.")
    except:
        print("Error al cargar calibración.")


#azul, lower = [90, 100, 100] | upper = [130, 255, 255]
# --- Color verde ---
lower_color = np.array([35, 70, 70])
upper_color = np.array([85, 255, 255])

print(f"Transmitiendo desde {URL}...")
print("Presiona ESC para salir.")

while True:
    # Tratar de vaciar el buffer
    for _ in range(5):
        cap.grab()

    ret, frame = cap.retrieve()
    if not ret:
        print("No se recibe video de la cámara IP.")
        break

    # --- CORRECCIÓN DE LENTE ---
    if mtx is not None and dist is not None:
        h, w = frame.shape[:2]
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # --- PROCESAMIENTO MODO ESPEJO ---
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))  # Tamaño estándar para fluidez

    # Convertir a HSV y crear máscara
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Limpiar la máscara para que el contorno sea sólido (copiar figura)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # --- CÁLCULO DE CENTRO Y ENVÍO UDP ---
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # Dibujar punto guía verde en la cámara real
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        # Enviar coordenadas por UDP
        data = f"{cx},{cy},640,480"
        sock.sendto(data.encode(), (udp_ip, udp_port))

    # --- PREPARAR PANTALLA DIVIDIDA ---
    # Conversion mascara
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Unir Mundo Real (Izquierda) y Gemelo Digital/Máscara (Derecha)
    visor_unificado = np.hstack((frame, mask_bgr))

    # Añadir Etiquetas
    cv2.putText(visor_unificado, "ESCENA REAL", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(visor_unificado, "GEMELO DIGITAL (SILUETA)", (650, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar ventana única
    cv2.imshow("Mundo Real vs Gemelo Digital", visor_unificado)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()