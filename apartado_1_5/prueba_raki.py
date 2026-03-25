import os
import cv2
import numpy as np
import open3d as o3d
import pyautogui
import mediapipe as mp

# --- CONFIGURACIÓN ---
SOURCE = "http://192.168.1.34:4747/video"
FRAME_SIZE = (640, 480)
TRACKER_TYPE = "MOSSE"  # "KCF" o "CSRT"
H_FILE = "../apartado_1_3/homografia_mesa.npz"

# Plano de trabajo (cm)
PLANE_W = 30.0
PLANE_H = 20.0
GOAL_WORLD = (20.0, 10.0)
VIS_SCALE = 10.0

# --- INICIALIZAR MEDIAPIPE PARA EL CLICK ---
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# --- FUNCIONES DE APOYO ---
def abrir_camara(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cap if cap.isOpened() else None


def crear_tracker(tracker_type="MOSSE"):
    tracker_type = tracker_type.upper()
    if tracker_type == "KCF": return cv2.TrackerKCF_create()
    if tracker_type == "CSRT": return cv2.TrackerCSRT_create()
    if tracker_type == "MOSSE": return cv2.legacy.TrackerMOSSE_create()
    return cv2.TrackerMIL_create()


def cargar_homografia(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra '{path}'")
    data = np.load(path)
    return data["H_w2i"], data["H_i2w"], float(data["reprojection_error"][0])


def image_to_world(H_i2w, pt):
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    w = cv2.perspectiveTransform(p, H_i2w)[0, 0]
    return float(w[0]), float(w[1])


def world_to_image(H_w2i, world_xy):
    p = np.array([[[float(world_xy[0]), float(world_xy[1])]]], dtype=np.float32)
    q = cv2.perspectiveTransform(p, H_w2i)[0, 0]
    return int(round(q[0])), int(round(q[1]))


def clamp_bbox(bbox, frame_shape):
    h_img, w_img = frame_shape[:2]
    x, y, w, h = map(int, map(round, bbox))
    x = max(0, min(w_img - 1, x))
    y = max(0, min(h_img - 1, y))
    w = max(1, min(w_img - x, w))
    h = max(1, min(h_img - y, h))
    return x, y, w, h


# --- CLASE ESCENA VIRTUAL ---
class VirtualScene:
    def __init__(self, goal_world, vis_scale=10.0):
        self.vis_scale = vis_scale
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Visor virtual 1.3", width=640, height=480)
        self.vis.get_render_option().background_color = np.asarray([1.0, 1.0, 1.0])
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)
        self.hand = o3d.geometry.TriangleMesh.create_sphere(radius=0.20)
        self.hand.paint_uniform_color([1.0, 0.2, 0.2])
        self.goal = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.12)
        self.goal.paint_uniform_color([0.2, 0.2, 1.0])
        self.goal.translate([goal_world[0] / vis_scale, goal_world[1] / vis_scale, 0.0])
        self.vis.add_geometry(self.frame);
        self.vis.add_geometry(self.hand);
        self.vis.add_geometry(self.goal)
        self.current_pos = np.array([0.0, 0.0, 0.0])

    def update_hand(self, xw, yw):
        new_pos = np.array([xw / self.vis_scale, yw / self.vis_scale, 0.0])
        self.hand.translate(new_pos - self.current_pos)
        self.current_pos = new_pos
        self.vis.update_geometry(self.hand)

    def render(self):
        self.vis.poll_events();
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


# --- FUNCIONES DE DIBUJO ---
def dibujar_plano_ra(frame, H_w2i):
    pts = np.array([[0, 0], [PLANE_W, 0], [PLANE_W, PLANE_H], [0, PLANE_H]], dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H_w2i).astype(int)
    cv2.polylines(frame, [dst], True, (180, 220, 255), 2)


def dibujar_info(frame, tracker_name, fps, err, contacto, xw, yw, state, es_click=False):
    color = (0, 0, 255) if es_click else (50, 170, 50)
    cv2.putText(frame, f"Estado: {state} {'[CLICK]' if es_click else ''}", (20, 30), 0, 0.7, color, 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 60), 0, 0.6, (255, 255, 255), 1)


# --- MAIN ---
def main():
    pyautogui.FAILSAFE = False
    try:
        H_w2i, H_i2w, reproj_error = cargar_homografia(H_FILE)
    except Exception as e:
        print(f"Error: {e}");
        return

    cap = abrir_camara(SOURCE)
    if not cap: return

    tracker = crear_tracker(TRACKER_TYPE)
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
    scene = VirtualScene(GOAL_WORLD, VIS_SCALE)

    deteccion_completada = False
    learning_count = 0
    prev_x, prev_y = None, None

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.resize(frame, FRAME_SIZE)
        timer = cv2.getTickCount()
        es_click = False

        if learning_count < 60:
            back_sub.apply(frame)
            learning_count += 1
            state_text = "Calibrando..."
        elif not deteccion_completada:
            fg = back_sub.apply(frame)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            state_text = "Buscando mano..."
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 3000:
                    (x, y), r = cv2.minEnclosingCircle(c)
                    bbox = (x - 60, y - 60, 120, 120)
                    tracker = crear_tracker(TRACKER_TYPE)
                    tracker.init(frame, bbox)
                    deteccion_completada = True
        else:
            ok_t, bbox = tracker.update(frame)
            if ok_t:
                x, y, w, h = clamp_bbox(bbox, frame.shape)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                contacto = (x + w // 2, y + h)
                xw, yw = image_to_world(H_i2w, contacto)
                scene.update_hand(xw, yw)

                # --- CLICK CON MEDIAPIPE ---
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands_detector.process(rgb)
                if res.multi_hand_landmarks:
                    for hlms in res.multi_hand_landmarks:
                        if hlms.landmark[8].y > hlms.landmark[6].y: es_click = True

                # --- MOUSE ---
                sw, sh = pyautogui.size()
                tx, ty = int((xw / PLANE_W) * sw), int((yw / PLANE_H) * sh)
                if prev_x is not None:
                    cx, cy = int(prev_x + 0.3 * (tx - prev_x)), int(prev_y + 0.3 * (ty - prev_y))
                else:
                    cx, cy = tx, ty

                pyautogui.moveTo(np.clip(cx, 0, sw - 1), np.clip(cy, 0, sh - 1))
                if es_click: pyautogui.click()
                prev_x, prev_y = cx, cy
                state_text = "Tracking"
            else:
                deteccion_completada = False

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        dibujar_plano_ra(frame, H_w2i)
        dibujar_info(frame, TRACKER_TYPE, fps, reproj_error, None, 0, 0, state_text, es_click)
        cv2.imshow("Control Raton", frame)
        scene.render()
        if cv2.waitKey(1) & 0xFF == 13: break

    cap.release();
    cv2.destroyAllWindows();
    scene.close()


if __name__ == "__main__":
    main()