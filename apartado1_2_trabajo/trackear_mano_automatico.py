import cv2
import numpy as np
import time

def ejecutar_aplicacion(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened(): return

    # Usamos KCF para que sea más ligero y responda mejor a la velocidad
    tracker = cv2.TrackerCSRT.create()
    # history=500 para que tenga buena memoria del fondo blanco (hay que fijarse si vamos a cambiar de fondo!!)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    deteccion_completada = False
    tramas_aprendizaje = 0

    print("Calibrando fondo... No metas la mano todavía.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (640, 480))

        # 1. Fase de aprendizaje del fondo (aprox 2-3 segundos)
        if tramas_aprendizaje < 60:
            backSub.apply(frame)
            tramas_aprendizaje += 1
            cv2.putText(frame, f"Calibrando: {tramas_aprendizaje}/60", (20, 30), 1, 1.5, (255, 0, 0), 2)
            cv2.imshow("Proyecto 1.2", frame)
            continue

        if not deteccion_completada:
            # 2. Detección con filtros de tamaño
            fgmask = backSub.apply(frame)

            # Limpiar ruido
            kernel = np.ones((5, 5), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.dilate(fgmask, kernel, iterations=2)

            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)

                if 3000 < area < (640 * 480 * 0.5):
                    # usamos un truco de "Círculo Mínimo" - para ajustar el seg. a la mano
                    (x, y), radio = cv2.minEnclosingCircle(c)

                    # Creamos un bbox cuadrado fijo alrededor de ese centro
                    # Esto ignora el largo del brazo y se centra en la "masa" de la mano
                    ancho_fijo = 120  # Ajusta este valor al tamaño de tu mano en pantalla
                    bbox = (int(x - ancho_fijo / 2), int(y - ancho_fijo / 2), ancho_fijo, ancho_fijo)

                    tracker = cv2.TrackerKCF.create()
                    tracker.init(frame, bbox)
                    deteccion_completada = True

        else:
            # 3. Seguimiento
            ok, bbox = tracker.update(frame)
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

                # Punto central (para el apartado 1.3) - ana?
                centro = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
                cv2.circle(frame, centro, 5, (0, 0, 255), -1)
            else:
                print("Seguimiento perdido.")
                deteccion_completada = False

            cv2.imshow("Proyecto - apartado 1.2", frame)

        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()


ejecutar_aplicacion("http://192.168.93.209:4747/video")