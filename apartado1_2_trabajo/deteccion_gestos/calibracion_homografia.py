import cv2
import numpy as np

# --- CONFIGURACIÓN PARA MÓVIL ---
IP_MOVIL = "192.168.93.209"
PUERTO = "4747"
# Usamos la URL de video de DroidCam
SOURCE = f"http://{IP_MOVIL}:{PUERTO}/video"

FRAME_SIZE = (640, 480)
OUT_FILE = "homografia_mesa.npz"

# Puntos reales en el mundo (cm)
WORLD_POINTS = np.array( [
    [0.0, 0.0],
    [15.0, 0.0],
    [30.0, 0.0],
    [30.0, 10.0],
    [30.0, 20.0],
    [15.0, 20.0],
    [0.0, 20.0],
    [0.0, 10.0],
], dtype=np.float32)

clicked_points = []
frame_frozen = None


def abrir_camara(source):
    # Para URLs de IP, OpenCV suele funcionar mejor con FFMPEG
    cap = cv2.VideoCapture(source)

    # Reducir el lag del stream
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        return None
    return cap


def dibujar_info(frame, texto, y=25, color=(255, 255, 255)):
    cv2.putText(frame, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def on_mouse(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < len(WORLD_POINTS):
        clicked_points.append([x, y])


def dibujar_puntos(frame, puntos):
    for i, p in enumerate(puntos):
        x, y = int(p[0]), int(p[1])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def main():
    global frame_frozen, clicked_points

    print(f"Conectando a: {SOURCE}...")
    cap = abrir_camara(SOURCE)
    if cap is None:
        print("No se pudo conectar al móvil. Revisa la IP y que DroidCam esté abierto.")
        return

    print("Pulsa ESPACIO para congelar la imagen cuando la mesa esté bien encuadrada.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error recibiendo video del móvil.")
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        vis = frame.copy()
        dibujar_info(vis, "ESPACIO: congelar imagen   |   ESC: salir")
        cv2.imshow("Camara Movil", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == 32:
            frame_frozen = frame.copy()
            break

    cap.release()
    cv2.destroyWindow("Camara Movil")

    cv2.namedWindow("Calibracion homografia")
    cv2.setMouseCallback("Calibracion homografia", on_mouse)

    print("\nHaz clic en los 8 puntos de la mesa siguiendo el orden de WORLD_POINTS.")

    while True:
        vis = frame_frozen.copy()
        dibujar_puntos(vis, clicked_points)
        dibujar_info(vis, f"Puntos: {len(clicked_points)}/{len(WORLD_POINTS)}", 25, (0, 255, 0))
        dibujar_info(vis, "R: reset | ENTER: calcular | ESC: salir", 55, (255, 255, 255))

        cv2.imshow("Calibracion homografia", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == 27: break
        if key == ord("r"): clicked_points = []
        if key == 13:
            if len(clicked_points) != len(WORLD_POINTS):
                print("Faltan puntos por seleccionar.")
                continue

            img_points = np.array(clicked_points, dtype=np.float32).reshape(-1, 1, 2)
            world_points = WORLD_POINTS.reshape(-1, 1, 2)

            H_w2i, _ = cv2.findHomography(world_points, img_points, cv2.RANSAC, 3.0)
            if H_w2i is None:
                print("Error calculando homografía.")
                continue

            H_i2w = np.linalg.inv(H_w2i)

            # Calcular error de reproyección
            img_points_est = cv2.perspectiveTransform(world_points, H_w2i)
            err = np.linalg.norm(img_points_est[:, 0, :] - img_points[:, 0, :], axis=1).mean()

            np.savez(OUT_FILE, H_w2i=H_w2i, H_i2w=H_i2w, world_points=WORLD_POINTS,
                     image_points=np.array(clicked_points, dtype=np.float32),
                     reprojection_error=np.array([err], dtype=np.float32))

            print(f"\n¡Éxito! Archivo '{OUT_FILE}' guardado.")
            print(f"Error de precisión: {err:.3f} píxeles.")
            cv2.destroyAllWindows()
            return


if __name__ == "__main__":
    main()