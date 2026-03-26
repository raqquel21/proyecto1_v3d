import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# --- CONFIGURACIÓN ---
SOURCE = "http://192.168.1.34:4747/video"
FRAME_SIZE = (640, 480)
H_FILE = "../apartado_1_3/homografia_mesa.npz"
PLANE_W, PLANE_H = 30.0, 20.0

# Desactivar lag de pyautogui
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def cargar_homografia(path):
    data = np.load(path)
    return data["H_i2w"]


# --- DETECCIÓN DE PUÑO ---
def es_puno_cerrado(hand_landmarks):
    tips_ids = [8, 12, 16, 20]  # índice, medio, anular, meñique
    dedos_doblados = 0

    for tip_id in tips_ids:
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[tip_id - 2]

        if tip.y > pip.y:
            dedos_doblados += 1

    return dedos_doblados >= 3


def main():
    H_i2w = cargar_homografia(H_FILE)
    cap = cv2.VideoCapture(SOURCE)
    sw, sh = pyautogui.size()

    prev_x, prev_y = 0, 0
    smooth_factor = 0.25

    click_activo = False  # evita múltiples clicks

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, FRAME_SIZE)

        # ❌ SIN espejo (ya lo quitaste)
        # frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_detector.process(rgb)

        if res.multi_hand_landmarks:
            mano = res.multi_hand_landmarks[0]

            # Landmark estable para mover
            lm = mano.landmark[9]

            px, py = lm.x * FRAME_SIZE[0], lm.y * FRAME_SIZE[1]

            # --- HOMOGRAFÍA ---
            p = np.array([[[float(px), float(py)]]], dtype=np.float32)
            w_pt = cv2.perspectiveTransform(p, H_i2w)[0, 0]
            xw, yw = w_pt[0], w_pt[1]

            # --- MAPEO A PANTALLA ---
            tx = np.interp(xw, [0, PLANE_W], [0, sw])
            ty = np.interp(yw, [0, PLANE_H], [0, sh])

            # Suavizado
            cx = prev_x + (tx - prev_x) * smooth_factor
            cy = prev_y + (ty - prev_y) * smooth_factor

            pyautogui.moveTo(int(cx), int(cy))
            prev_x, prev_y = cx, cy

            # --- CLICK CON PUÑO ---
            if es_puno_cerrado(mano):
                if not click_activo:
                    pyautogui.click()
                    click_activo = True
            else:
                click_activo = False

            # Dibujar mano
            mp.solutions.drawing_utils.draw_landmarks(
                frame, mano, mp_hands.HAND_CONNECTIONS
            )

        cv2.imshow("Control sin Espejo", frame)
        if cv2.waitKey(1) & 0xFF == 13:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()