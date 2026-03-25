import cv2
import numpy as np

SOURCE = "http://192.168.1.34:4747/video"
FRAME_SIZE = (640, 480)
H_FILE = "homografia_mesa.npz"

clicked_point = None
last_world = None


def abrir_camara(source):
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


def image_to_world(H_i2w, pt):
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    w = cv2.perspectiveTransform(p, H_i2w)[0, 0]
    return float(w[0]), float(w[1])


def on_mouse(event, x, y, flags, param):
    global clicked_point, last_world
    H_i2w = param

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        last_world = image_to_world(H_i2w, clicked_point)
        print(f"Pixel: ({x}, {y})  ->  Mundo: ({last_world[0]:.2f}, {last_world[1]:.2f}) cm")


def dibujar_texto(frame, texto, y, color=(255, 255, 255)):
    cv2.putText(
        frame,
        texto,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


def main():
    global clicked_point, last_world

    data = np.load(H_FILE)
    H_i2w = data["H_i2w"]

    cap = abrir_camara(SOURCE)
    if cap is None:
        print("No se pudo abrir la cámara.")
        return

    print("Pulsa ESPACIO para congelar la imagen.")
    print("Haz clic sobre el plano para ver coordenadas mundo.")
    print("Pulsa R para volver a capturar.")
    print("Pulsa ESC para salir.")

    frozen = None
    mode_frozen = False

    cv2.namedWindow("Verificar homografia")

    while True:
        if not mode_frozen:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer frame.")
                break

            frame = cv2.resize(frame, FRAME_SIZE)
            vis = frame.copy()
            dibujar_texto(vis, "ESPACIO: congelar | ESC: salir", 25, (0, 255, 0))
            cv2.imshow("Verificar homografia", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32:
                frozen = frame.copy()
                mode_frozen = True
                clicked_point = None
                last_world = None
                cv2.setMouseCallback("Verificar homografia", on_mouse, H_i2w)

        else:
            vis = frozen.copy()

            if clicked_point is not None:
                cv2.circle(vis, clicked_point, 5, (0, 0, 255), -1)
                texto = f"({last_world[0]:.2f}, {last_world[1]:.2f}) cm"
                cv2.putText(
                    vis,
                    texto,
                    (clicked_point[0] + 10, clicked_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            dibujar_texto(vis, "Click: coordenadas mundo | R: recapturar | ESC: salir", 25, (0, 255, 255))
            cv2.imshow("Verificar homografia", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("r"):
                mode_frozen = False
                clicked_point = None
                last_world = None
                cv2.setMouseCallback("Verificar homografia", lambda *args: None)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()