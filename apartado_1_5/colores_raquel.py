import cv2
import numpy as np
import pyautogui
import os

# --- CONFIGURACIÓN CÁMARA ---
SOURCE = "http://192.168.1.34:4747/video"
FRAME_SIZE = (640, 480)
TRACKER_TYPE = "MOSSE"
H_FILE = "../apartado_1_3/homografia_mesa.npz"

# --- RANGOS DE COLOR (HSV) ---
verde_lower = np.array([35, 70, 70])
verde_upper = np.array([85, 255, 255])
azul_lower = np.array([105, 120, 40])
azul_upper = np.array([130, 255, 140])

# --- FUNCIONES AUXILIARES ---
def cargar_homografia(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra '{path}'")
    data = np.load(path)
    return data["H_w2i"], data["H_i2w"]

def image_to_world(H_i2w, pt):
    p = np.array([[[float(pt[0]), float(pt[1])]]], np.float32)
    w = cv2.perspectiveTransform(p, H_i2w)[0, 0]
    return float(w[0]), float(w[1])

def world_to_screen(H_w2i, xw, yw):
    p = np.array([[[xw, yw]]], np.float32)
    px = cv2.perspectiveTransform(p, H_w2i)[0,0]
    screen_w, screen_h = pyautogui.size()
    sx = np.clip(int((px[0] / FRAME_SIZE[0]) * screen_w), 0, screen_w-1)
    sy = np.clip(int((px[1] / FRAME_SIZE[1]) * screen_h), 0, screen_h-1)
    return sx, sy

def crear_tracker(tracker_type="MOSSE"):
    tracker_type = tracker_type.upper()
    if tracker_type == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    raise ValueError(f"Tracker no soportado: {tracker_type}")

def dibujar_zona(frame, H_w2i):
    # Dibujar zona homografiada como rectángulo azul
    corners_world = np.array([[0,0],[30,0],[30,20],[0,20]], dtype=np.float32).reshape(-1,1,2)
    corners_img = cv2.perspectiveTransform(corners_world, H_w2i)
    corners_img = np.int32(corners_img.reshape(-1,2))
    cv2.polylines(frame, [corners_img], True, (255,0,0), 2)

# --- MAIN ---
def main():
    H_w2i, H_i2w = cargar_homografia(H_FILE)
    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    tracker = None
    deteccion_completada = False
    prev_cursor = [None, None]
    click_estado = False
    estado_text = "BUSCANDO MANO"

    while True:
        for _ in range(3):
            cap.grab()
        ok, frame = cap.retrieve()
        if not ok:
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- DETECCIÓN INICIAL ---
        if not deteccion_completada:
            mask_verde = cv2.inRange(hsv, verde_lower, verde_upper)
            mask_azul = cv2.inRange(hsv, azul_lower, azul_upper)
            mask = cv2.bitwise_or(mask_verde, mask_azul)

            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 200:
                    x, y, w, h = cv2.boundingRect(c)
                    tracker = crear_tracker(TRACKER_TYPE)
                    tracker.init(frame, (x, y, w, h))
                    deteccion_completada = True
                    estado_text = "MOVIENDO"

        # --- TRACKING ---
        if deteccion_completada and tracker is not None:
            ok_tracker, bbox = tracker.update(frame)
            if ok_tracker:
                x, y, w, h = [int(v) for v in bbox]
                centro = (x + w//2, y + h//2)
                contacto = (x + w//2, y + h)

                # Dibujar rectángulo y centro
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.circle(frame, centro, 4, (0,0,255), -1)
                cv2.circle(frame, contacto, 4, (255,0,255), -1)

                # Coordenadas en mundo real
                xw, yw = image_to_world(H_i2w, contacto)

                # --- CONTROLAR CURSOR ---
                cursor_x, cursor_y = world_to_screen(H_w2i, xw, yw)
                if prev_cursor[0] is not None:
                    cursor_x = int(prev_cursor[0] + 0.2*(cursor_x - prev_cursor[0]))
                    cursor_y = int(prev_cursor[1] + 0.2*(cursor_y - prev_cursor[1]))
                pyautogui.moveTo(cursor_x, cursor_y)
                prev_cursor = [cursor_x, cursor_y]

                # --- ESTADO Y CLICK ---
                hay_verde = cv2.countNonZero(cv2.inRange(hsv, verde_lower, verde_upper)) > 50
                hay_azul = cv2.countNonZero(cv2.inRange(hsv, azul_lower, azul_upper)) > 50

                if hay_azul and not hay_verde and not click_estado:
                    pyautogui.click()
                    estado_text = "CLICK"
                    click_estado = True
                elif hay_azul and hay_verde:
                    estado_text = "MOVIENDO"
                    click_estado = False
                elif not hay_azul:
                    estado_text = "BUSCANDO MANO"

            else:
                # Tracker perdido
                deteccion_completada = False
                tracker = None
                estado_text = "BUSCANDO MANO"

        # --- DIBUJAR ZONA Y ESTADO ---
        dibujar_zona(frame, H_w2i)
        cv2.putText(frame, estado_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Camara", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()