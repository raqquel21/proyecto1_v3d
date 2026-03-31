import os
import cv2
import numpy as np
import open3d as o3d
import pyautogui

# --- CONFIGURACIÓN ---
SOURCE = "http://192.168.1.34:4747/video"  # cambia a 0 si usas webcam
FRAME_SIZE = (640, 480)
TRACKER_TYPE = "CSRT"
H_FILE = "../apartado_1_3/homografia_mesa.npz"

PLANE_W = 30.0
PLANE_H = 20.0
GOAL_WORLD = (20.0, 10.0)
VIS_SCALE = 10.0

LEARNING_FRAMES = 60
FIXED_BOX_SIZE = 120
CLICK_THRESHOLD = 0.65


# ------------------ UTILIDADES ------------------

def abrir_camara(source):
    cap = cv2.VideoCapture(source)
    return cap if cap.isOpened() else None


def crear_tracker():
    # versión robusta para OpenCV modernos
    return cv2.legacy.TrackerCSRT_create()


def cargar_homografia(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra '{path}'")
    data = np.load(path)
    return data["H_w2i"], data["H_i2w"], float(data["reprojection_error"][0])


def image_to_world(H_i2w, pt):
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    w = cv2.perspectiveTransform(p, H_i2w)[0, 0]
    return float(w[0]), float(w[1])


def clamp(val, low, high):
    return max(low, min(high, val))


def clamp_bbox(bbox, shape):
    h, w = shape[:2]
    x, y, bw, bh = bbox

    x = clamp(int(x), 0, w - 1)
    y = clamp(int(y), 0, h - 1)
    bw = clamp(int(bw), 1, w - x)
    bh = clamp(int(bh), 1, h - y)

    return (float(x), float(y), float(bw), float(bh))  # 🔥 importante


# ------------------ ESCENA 3D ------------------

class VirtualScene:
    def __init__(self, vis_scale=10.0):
        self.vis_scale = vis_scale
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Visor virtual", width=640, height=480)

        self.hand = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        self.hand.paint_uniform_color([1, 0, 0])
        self.vis.add_geometry(self.hand)

        self.current_pos = np.zeros(3)

    def update_hand(self, xw, yw):
        new_pos = np.array([xw / self.vis_scale, yw / self.vis_scale, 0])
        delta = new_pos - self.current_pos
        self.hand.translate(delta)
        self.current_pos = new_pos
        self.vis.update_geometry(self.hand)

    def render(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


# ------------------ MAIN ------------------

def main():
    try:
        H_w2i, H_i2w, _ = cargar_homografia(H_FILE)
    except Exception as e:
        print(e)
        return

    cap = abrir_camara(SOURCE)
    if cap is None:
        print("No se pudo abrir la cámara")
        return

    tracker = None
    back_sub = cv2.createBackgroundSubtractorMOG2()

    scene = VirtualScene(VIS_SCALE)

    deteccion = False
    learning = 0

    area_ref = None
    calibrando = 0
    click_hecho = False

    prev_x, prev_y = None, None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, FRAME_SIZE)

        # --- CALIBRACIÓN DE FONDO ---
        if learning < LEARNING_FRAMES:
            back_sub.apply(frame)
            learning += 1
            state = f"Calibrando fondo {learning}"

        # --- DETECCIÓN INICIAL ---
        elif not deteccion:
            fg = back_sub.apply(frame)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            state = "Buscando mano..."

            if contours:
                c = max(contours, key=cv2.contourArea)
                (x, y), _ = cv2.minEnclosingCircle(c)

                bbox = (x - 60, y - 60, 120, 120)
                bbox = clamp_bbox(bbox, frame.shape)

                tracker = crear_tracker()
                tracker.init(frame, bbox)

                deteccion = True
                calibrando = 20

        # --- TRACKING ---
        else:
            ok_tracker, bbox = tracker.update(frame)

            if ok_tracker:
                x, y, w, h = map(int, bbox)
                area = w * h

                if calibrando > 0:
                    area_ref = area if area_ref is None else (area_ref + area) / 2
                    calibrando -= 1
                    state = "Calibrando mano"

                else:
                    ratio = area / area_ref

                    if ratio < CLICK_THRESHOLD:
                        state = "CLICK"
                        if not click_hecho:
                            pyautogui.click()
                            click_hecho = True
                    else:
                        state = "Tracking"
                        click_hecho = False

                    contacto = (x + w // 2, y + h)
                    xw, yw = image_to_world(H_i2w, contacto)

                    scene.update_hand(xw, yw)

                    sw, sh = pyautogui.size()
                    cx = int((xw / PLANE_W) * sw)
                    cy = int((yw / PLANE_H) * sh)

                    if prev_x is not None:
                        cx = int(prev_x + 0.25 * (cx - prev_x))
                        cy = int(prev_y + 0.25 * (cy - prev_y))

                    pyautogui.moveTo(cx, cy)
                    prev_x, prev_y = cx, cy

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                deteccion = False
                state = "Reiniciando"

        cv2.putText(frame, state, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Tracking Mano", frame)
        scene.render()

        if cv2.waitKey(1) & 0xFF == 13:
            break

    cap.release()
    cv2.destroyAllWindows()
    scene.close()


if __name__ == "__main__":
    main()