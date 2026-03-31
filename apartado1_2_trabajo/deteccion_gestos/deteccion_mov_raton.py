import cv2
import mediapipe as mp
import numpy as np
import os

# --- CONFIGURACIÓN ---
URL = "http://192.168.93.211:4747/video"
H_FILE = "homografia_mesa.npz"
FRAME_SIZE = (640, 480)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


def cargar_homografia(path):
    if not os.path.exists(path):
        print(f"ERROR: No se encuentra {path}")
        return None
    data = np.load(path)
    return data["H_i2w"]


def image_to_world(H_i2w, pt):
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    w = cv2.perspectiveTransform(p, H_i2w)[0, 0]
    return float(w[0]), float(w[1])


def ejecutar_aplicacion():
    H_i2w = cargar_homografia(H_FILE)
    cap = cv2.VideoCapture(URL)

    print("MediaPipe Hand Tracker iniciado. ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, FRAME_SIZE)
        # MediaPipe necesita RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Dibujar esqueleto de la mano
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 2. Obtener coordenadas del dedo índice (Landmark 8 es la punta, 5 es la base)
                # Usamos la punta del índice para el movimiento del ratón
                idx_tip = hand_landmarks.landmark[8]
                idx_base = hand_landmarks.landmark[5]

                # Convertir coordenadas normalizadas (0-1) a píxeles
                cx, cy = int(idx_tip.x * FRAME_SIZE[0]), int(idx_tip.y * FRAME_SIZE[1])

                # 3. Lógica de Gesto (Detección de CLICK)
                # Si la punta del índice (8) está por debajo de su articulación media (6), es un puño/click
                idx_pip = hand_landmarks.landmark[6]
                es_click = idx_tip.y > idx_pip.y  # En imagen, Y aumenta hacia abajo

                # 4. Homografía (Mapeo a la mesa)
                if H_i2w is not None:
                    xw, yw = image_to_world(H_i2w, (cx, cy))
                    cv2.putText(frame, f"Mesa: {xw:.1f}, {yw:.1f} cm", (20, 450),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 5. Visualización
                color = (0, 0, 255) if es_click else (0, 255, 0)
                estado = "CLICK (PUNO)" if es_click else "MOVIENDO (ABIERTA)"

                cv2.circle(frame, (cx, cy), 10, color, -1)
                cv2.putText(frame, estado, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow("Mando Virtual MediaPipe", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ejecutar_aplicacion()