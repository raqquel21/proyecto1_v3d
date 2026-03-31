# Tracker automático que implementa el modelo CSRT.
# Detecta el movimiento y extrae el objeto (mano) del fondo (mesa)
import cv2
import numpy as np
import time


def abrirCamara(url):
    #Establece la conexión con la cámara IP
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Error: No se pudo conectar a la URL: {url}")
        return None
    return cap


def trackearManoAutomatica(captura):
    #Lógica de detección automática y seguimiento.
    # Configuración de herramientas
    tracker = cv2.TrackerCSRT.create()
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    deteccion_completada = False
    tramas_aprendizaje = 0
    ancho_fijo = 120  # Tamaño del cuadro para evitar el brazo

    print("Calibrando fondo... No metas la mano todavía.")

    while True:
        ok, frame = captura.read()
        if not ok:
            break

        # Redimensionado para consistencia y velocidad
        frame = cv2.resize(frame, (640, 480))

        # 1. FASE DE APRENDIZAJE (Calibración)
        if tramas_aprendizaje < 60:
            backSub.apply(frame)
            tramas_aprendizaje += 1
            cv2.putText(frame, f"Calibrando: {tramas_aprendizaje}/60", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Proyecto - Apartado 1.2", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        # 2. FASE DE DETECCIÓN (Si no hay objeto fijado)
        if not deteccion_completada:
            fgmask = backSub.apply(frame)

            # Limpieza de la máscara
            kernel = np.ones((5, 5), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.dilate(fgmask, kernel, iterations=2)

            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)

                # Filtro de tamaño para detectar solo la mano
                if 3000 < area < (640 * 480 * 0.5):
                    (x, y), radio = cv2.minEnclosingCircle(c)

                    # Definimos el Bounding Box centrado en la masa principal
                    bbox = (int(x - ancho_fijo / 2), int(y - ancho_fijo / 2), ancho_fijo, ancho_fijo)

                    tracker = cv2.TrackerKCF.create()
                    tracker.init(frame, bbox)
                    deteccion_completada = True
                    print("Detección automática exitosa.")

            cv2.putText(frame, "Buscando mano...", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. FASE DE SEGUIMIENTO (Tracker activo)
        else:
            ok_tracker, bbox = tracker.update(frame)
            if ok_tracker:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

                # Centroide para el futuro apartado 1.3
                centro = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
                cv2.circle(frame, centro, 5, (0, 0, 255), -1)
            else:
                print("Seguimiento perdido, reiniciando detección.")
                deteccion_completada = False

        cv2.imshow("Proyecto - Apartado 1.2", frame)

        # Salida con tecla ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    captura.release()
    cv2.destroyAllWindows()


def main():
    # Configuración inicial
    url = "http://192.168.93.211:4747/video"

    captura = abrirCamara(url)

    if captura is not None:
        trackearManoAutomatica(captura)


if __name__ == '__main__':
    main()