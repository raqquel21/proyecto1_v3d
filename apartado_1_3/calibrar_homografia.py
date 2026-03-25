import cv2
import numpy as np

SOURCE = "http://192.168.1.34:4747/video"
FRAME_SIZE = (640, 480)
OUT_FILE = "homografia_mesa.npz"

WORLD_POINTS = np.array([
    [0.0,  0.0],
    [15.0, 0.0],
    [30.0, 0.0],
    [30.0, 10.0],
    [30.0, 20.0],
    [15.0, 20.0],
    [0.0,  20.0],
    [0.0,  10.0],
], dtype=np.float32)

clicked_points = []
frame_frozen = None


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


def dibujar_info(frame, texto, y=25, color=(255, 255, 255)):
    cv2.putText(
        frame,
        texto,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


def on_mouse(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < len(WORLD_POINTS):
        clicked_points.append([x, y])


def dibujar_puntos(frame, puntos):
    for i, p in enumerate(puntos):
        x, y = int(p[0]), int(p[1])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            str(i),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


def main():
    global frame_frozen, clicked_points

    cap = abrir_camara(SOURCE)
    if cap is None:
        print("No se pudo abrir la cámara.")
        return

    print("Pulsa ESPACIO para congelar la imagen del plano.")
    print("Pulsa ESC para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer frame.")
            cap.release()
            return

        frame = cv2.resize(frame, FRAME_SIZE)
        vis = frame.copy()
        dibujar_info(vis, "ESPACIO: congelar imagen   |   ESC: salir")
        cv2.imshow("Camara", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == 32:
            frame_frozen = frame.copy()
            break

    cap.release()
    cv2.destroyWindow("Camara")

    cv2.namedWindow("Calibracion homografia")
    cv2.setMouseCallback("Calibracion homografia", on_mouse)

    print("\nHaz clic en los 8 puntos en este orden:")
    print("0, 1, 2, 3, 4, 5, 6, 7")
    print("Tecla R: resetear puntos")
    print("Tecla ENTER: calcular homografia")
    print("Tecla ESC: salir")

    while True:
        vis = frame_frozen.copy()
        dibujar_puntos(vis, clicked_points)

        dibujar_info(vis, f"Puntos seleccionados: {len(clicked_points)}/{len(WORLD_POINTS)}", 25, (0, 255, 0))
        dibujar_info(vis, "R: resetear   |   ENTER: calcular   |   ESC: salir", 55, (255, 255, 255))

        cv2.imshow("Calibracion homografia", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cv2.destroyAllWindows()
            return

        if key == ord("r"):
            clicked_points = []

        if key == 13:
            if len(clicked_points) != len(WORLD_POINTS):
                print("Te faltan puntos.")
                continue

            img_points = np.array(clicked_points, dtype=np.float32).reshape(-1, 1, 2)
            world_points = WORLD_POINTS.reshape(-1, 1, 2)

            H_w2i, status = cv2.findHomography(world_points, img_points, cv2.RANSAC, 3.0)

            if H_w2i is None:
                print("No se pudo calcular la homografía.")
                continue

            H_i2w = np.linalg.inv(H_w2i)

            img_points_est = cv2.perspectiveTransform(world_points, H_w2i)
            err = np.linalg.norm(
                img_points_est[:, 0, :] - img_points[:, 0, :],
                axis=1
            ).mean()

            # Visualización rápida del ajuste
            vis_check = frame_frozen.copy()
            dibujar_puntos(vis_check, clicked_points)

            for p in img_points_est[:, 0, :]:
                x, y = int(round(p[0])), int(round(p[1]))
                cv2.circle(vis_check, (x, y), 5, (0, 255, 255), 2)

            dibujar_info(vis_check, f"Error reproyeccion: {err:.2f} px", 25, (0, 255, 255))
            cv2.imshow("Verificacion homografia", vis_check)
            cv2.waitKey(800)

            np.savez(
                OUT_FILE,
                H_w2i=H_w2i,
                H_i2w=H_i2w,
                world_points=WORLD_POINTS,
                image_points=np.array(clicked_points, dtype=np.float32),
                reprojection_error=np.array([err], dtype=np.float32)
            )

            print("\nHomografía guardada en:", OUT_FILE)
            print(f"Error medio de reproyección: {err:.3f} px")
            print("H_w2i =\n", H_w2i)
            print("H_i2w =\n", H_i2w)

            cv2.destroyAllWindows()
            return


if __name__ == "__main__":
    main()