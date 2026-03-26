import os
import cv2
import numpy as np
import open3d as o3d
import pyautogui
import mediapipe as mp

# --- CONFIGURACIÓN ---
SOURCE = "http://192.168.1.34:4747/video"
FRAME_SIZE = (640, 480)
H_FILE = "../apartado_1_3/homografia_mesa.npz"

PLANE_W = 30.0
PLANE_H = 20.0

# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_draw = mp.solutions.drawing_utils


# --- FUNCIONES ---
def abrir_camara(source):
    cap = cv2.VideoCapture(source)
    return cap if cap.isOpened() else None


def crear_tracker():
    return cv2.legacy.TrackerMOSSE_create()


def cargar_homografia(path):
    data = np.load(path)
    return data["H_w2i"], data["H_i2w"], float(data["reprojection_error"][0])


def image_to_world(H_i2w, pt):
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    w = cv2.perspectiveTransform(p, H_i2w)[0, 0]
    return float(w[0]), float(w[1])


def clamp_bbox(bbox, shape):
    h, w = shape[:2]
    x, y, bw, bh = map(int, map(round, bbox))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    bw = max(1, min(w - x, bw))
    bh = max(1, min(h - y, bh))
    return x, y, bw, bh


# --- MAIN ---
def main():
    pyautogui.FAILSAFE = False

    H_w2i, H_i2w, _ = cargar_homografia(H_FILE)
    cap = abrir_camara(SOURCE)
    tracker = crear_tracker()
    back_sub = cv2.createBackgroundSubtractorMOG2()

    prev_x = prev_y = None
    last_click = 0
    frame_count = 0

    deteccion = False
    learning = 0

    es_click = False
    hand_landmarks_cache = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        frame_count += 1

        # --- DETECCIÓN INICIAL ---
        if learning < 60:
            back_sub.apply(frame)
            learning += 1
            state = "Calibrando"

        elif not deteccion:
            fg = back_sub.apply(frame)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            state = "Buscando mano"

            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 3000:
                    (x, y), _ = cv2.minEnclosingCircle(c)
                    bbox = (x - 60, y - 60, 120, 120)
                    tracker = crear_tracker()
                    tracker.init(frame, bbox)
                    deteccion = True

        else:
            ok_t, bbox = tracker.update(frame)

            if ok_t:
                x, y, w, h = clamp_bbox(bbox, frame.shape)
                contacto = (x + w // 2, y + h)

                # --- COORDENADAS MUNDO ---
                xw, yw = image_to_world(H_i2w, contacto)

                # --- MEDIAPIPE SOLO EN ROI ---
                roi = frame[y:y+h, x:x+w]

                if roi.size != 0:
                    roi_small = cv2.resize(roi, None, fx=0.5, fy=0.5)
                    rgb = cv2.cvtColor(roi_small, cv2.COLOR_BGR2RGB)

                    res = hands_detector.process(rgb)

                    if res.multi_hand_landmarks:
                        hand_landmarks_cache = res.multi_hand_landmarks
                        hlms = hand_landmarks_cache[0]

                        # Puño = click
                        es_click = hlms.landmark[8].y > hlms.landmark[6].y
                    else:
                        es_click = False

                # --- DIBUJAR ESQUELETO (REESCALADO) ---
                if hand_landmarks_cache:
                    for hlms in hand_landmarks_cache:
                        for lm in hlms.landmark:
                            lm.x = (lm.x * w + x) / FRAME_SIZE[0]
                            lm.y = (lm.y * h + y) / FRAME_SIZE[1]

                        mp_draw.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)

                # --- MOUSE ---
                sw, sh = pyautogui.size()
                tx = int((xw / PLANE_W) * sw)
                ty = int((yw / PLANE_H) * sh)

                if prev_x is not None:
                    cx = int(prev_x + 0.3 * (tx - prev_x))
                    cy = int(prev_y + 0.3 * (ty - prev_y))
                else:
                    cx, cy = tx, ty

                pyautogui.moveTo(np.clip(cx, 0, sw - 1),
                                 np.clip(cy, 0, sh - 1))

                # --- CLICK CONTROLADO ---
                if es_click and (frame_count - last_click > 10):
                    pyautogui.click()
                    last_click = frame_count

                prev_x, prev_y = cx, cy

                # --- VISUAL ---
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(frame, contacto, 6, (255, 0, 255), -1)

                state = "Tracking"

            else:
                deteccion = False
                state = "Reiniciando"

        # --- INFO ---
        cv2.putText(frame, state, (20, 40), 0, 1, (0, 255, 0), 2)

        cv2.imshow("Control Raton PRO", frame)

        if cv2.waitKey(1) & 0xFF == 13:  # ENTER
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()