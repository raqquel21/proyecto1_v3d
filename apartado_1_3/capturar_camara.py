import cv2

SOURCE = 0          # 0 = webcam del portátil / webcam USB
FRAME_SIZE = (640, 480)


def abrir_camara(source=0):
    # Para webcam local, probamos primero la apertura normal.
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        # Si algún día vuelves a IP, probamos FFMPEG y luego normal.
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        return None

    return cap


def main():
    cap = abrir_camara(SOURCE)
    if cap is None:
        print("No se pudo abrir la cámara.")
        return

    print("Cámara abierta correctamente.")
    print("Pulsa ESC o q para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame.")
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        cv2.imshow("Captura - Webcam local", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()