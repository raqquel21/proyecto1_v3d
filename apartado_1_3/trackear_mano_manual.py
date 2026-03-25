import cv2

SOURCE = 0
FRAME_SIZE = (640, 480)
TRACKER_TYPE = "KCF"   # "KCF" o "CSRT"


def abrir_camara(source=0):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        return None

    return cap


def crear_tracker(tracker_type="KCF"):
    tracker_type = tracker_type.upper()

    if tracker_type == "MIL":
        return cv2.TrackerMIL_create()
    if tracker_type == "KCF":
        return cv2.TrackerKCF_create()
    if tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()
    if tracker_type == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()

    raise ValueError(f"Tracker no soportado: {tracker_type}")


def main():
    cap = abrir_camara(SOURCE)
    if cap is None:
        print("No se pudo abrir la cámara.")
        return

    ok, frame = cap.read()
    if not ok:
        print("No se pudo leer el primer frame.")
        cap.release()
        return

    frame = cv2.resize(frame, FRAME_SIZE)

    print("Selecciona la mano con el ratón y pulsa ENTER o ESPACIO.")
    print("Pulsa c para cancelar.")

    bbox = cv2.selectROI("Seleccion ROI", frame, False, False)
    cv2.destroyWindow("Seleccion ROI")

    if bbox[2] == 0 or bbox[3] == 0:
        print("ROI cancelada.")
        cap.release()
        cv2.destroyAllWindows()
        return

    tracker = crear_tracker(TRACKER_TYPE)
    tracker.init(frame, bbox)

    print(f"Tracking manual iniciado con {TRACKER_TYPE}.")
    print("Pulsa ESC para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer frame.")
            break

        frame = cv2.resize(frame, FRAME_SIZE)

        timer = cv2.getTickCount()
        ok_tracker, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok_tracker:
            x, y, w, h = bbox
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)

            # Centro del bbox
            centro = (int(x + w / 2), int(y + h / 2))
            cv2.circle(frame, centro, 4, (0, 0, 255), -1)

            # Punto de contacto con el plano (útil para el 1.3)
            contacto = (int(x + w / 2), int(y + h))
            cv2.circle(frame, contacto, 4, (255, 0, 255), -1)

        else:
            cv2.putText(
                frame,
                "Tracking failure detected",
                (80, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"{TRACKER_TYPE} Tracker",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 170, 50),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 170, 50),
            2
        )

        cv2.imshow("Tracking manual - mano", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()