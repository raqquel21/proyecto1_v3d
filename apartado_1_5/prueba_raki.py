import os
import cv2
import numpy as np
import open3d as o3d
import pyautogui
import mediapipe as mp

# --- CONFIGURACIÓN ---
SOURCE = "http://192.168.1.34:4747/video"
FRAME_SIZE = (640, 480)
TRACKER_TYPE = "MOSSE"
H_FILE = "../apartado_1_3/homografia_mesa.npz"

PLANE_W = 30.0
PLANE_H = 20.0
GOAL_WORLD = (20.0, 10.0)
VIS_SCALE = 10.0

# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_draw = mp.solutions.drawing_utils


# --- FUNCIONES BASE ---
def abrir_camara(source):
    cap = cv2.VideoCapture(source)
    return cap if cap.isOpened() else None


def crear_tracker(tracker_type="MOSSE"):
    if tracker_type == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "KCF":
        return cv2.TrackerKCF_create()
    if tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()
    return cv2.TrackerMIL_create()


def cargar_homografia(path):
    data = np.load(path)
    return data["H_w2i"], data["H_i2w"], float(data["reprojection_error"][0])


def image_to_world(H_i2w, pt):
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    w = cv2.perspectiveTransform(p, H_i2w)[0, 0]
    return float(w[0]), float(w[1])


def world_to_image(H_w2i, pt):
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    q = cv2.perspectiveTransform(p, H_w2i)[0, 0]
    return int(q[0]), int(q[1])


def clamp_bbox(bbox, shape):
    h, w = shape[:2]
    x, y, bw, bh = map(int, map(round, bbox))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    bw = max(1, min(w - x, bw))
    bh = max(1, min(h - y, bh))
    return x, y, bw, bh


# --- ESCENA 3D ---
class VirtualScene:
    def __init__(self, goal_world, vis_scale=10.0):
        self.vis_scale = vis_scale
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Visor virtual", 640, 480)

        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)
        self.hand = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        self.hand.paint_uniform_color([1, 0, 0])

        self.goal = o3d.geometry.TriangleMesh.create_box(0.5, 0.5, 0.1)
        self.goal.paint_uniform_color([0, 0, 1])
        self.goal.translate([goal_world[0]/vis_scale, goal_world[1]/vis_scale, 0])

        self.vis.add_geometry(self.frame)
        self.vis.add_geometry(self.hand)
        self.vis.add_geometry(self.goal)

        self.pos = np.array([0, 0, 0], dtype=float)

    def update_hand(self, xw, yw):
        new = np.array([xw/self.vis_scale, yw/self.vis_scale, 0])
        self.hand.translate(new - self.pos)
        self.pos = new
        self.vis.update_geometry(self.hand)

    def render(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


# --- DIBUJO RA ---
def dibujar_plano_ra(frame, H_w2i):
    pts = np.array([[0,0],[PLANE_W,0],[PLANE_W,PLANE_H],[0,PLANE_H]], dtype=np.float32).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H_w2i).astype(int)
    cv2.polylines(frame, [dst], True, (180,220,255), 2)

    goal = world_to_image(H_w2i, GOAL_WORLD)
    cv2.circle(frame, goal, 10, (255,0,255), 2)


# --- MAIN ---
def main():
    pyautogui.FAILSAFE = False

    H_w2i, H_i2w, _ = cargar_homografia(H_FILE)
    cap = abrir_camara(SOURCE)
    tracker = crear_tracker(TRACKER_TYPE)
    back_sub = cv2.createBackgroundSubtractorMOG2()

    scene = VirtualScene(GOAL_WORLD, VIS_SCALE)

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

        dibujar_plano_ra(frame, H_w2i)

        # --- DETECCIÓN ---
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
                    bbox = (x-60, y-60, 120, 120)
                    tracker = crear_tracker(TRACKER_TYPE)
                    tracker.init(frame, bbox)
                    deteccion = True

        else:
            ok_t, bbox = tracker.update(frame)

            if ok_t:
                x, y, w, h = clamp_bbox(bbox, frame.shape)
                contacto = (x + w//2, y + h)

                xw, yw = image_to_world(H_i2w, contacto)
                scene.update_hand(xw, yw)

                # --- MEDIAPIPE SOLO EN ROI ---
                roi = frame[y:y+h, x:x+w]
                if roi.size != 0:
                    roi_small = cv2.resize(roi, None, fx=0.5, fy=0.5)
                    rgb = cv2.cvtColor(roi_small, cv2.COLOR_BGR2RGB)

                    res = hands_detector.process(rgb)

                    if res.multi_hand_landmarks:
                        hand_landmarks_cache = res.multi_hand_landmarks
                        hlms = hand_landmarks_cache[0]
                        es_click = hlms.landmark[8].y > hlms.landmark[6].y
                    else:
                        es_click = False

                # --- DIBUJAR ESQUELETO ---
                if hand_landmarks_cache:
                    for hlms in hand_landmarks_cache:
                        for lm in hlms.landmark:
                            lm.x = (lm.x * w + x) / FRAME_SIZE[0]
                            lm.y = (lm.y * h + y) / FRAME_SIZE[1]
                        mp_draw.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)

                # --- RATÓN ---
                sw, sh = pyautogui.size()
                tx = int((xw / PLANE_W) * sw)
                ty = int((yw / PLANE_H) * sh)

                if prev_x is not None:
                    cx = int(prev_x + 0.3*(tx-prev_x))
                    cy = int(prev_y + 0.3*(ty-prev_y))
                else:
                    cx, cy = tx, ty

                pyautogui.moveTo(cx, cy)

                if es_click and (frame_count - last_click > 10):
                    pyautogui.click()
                    last_click = frame_count

                prev_x, prev_y = cx, cy

                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                cv2.circle(frame, contacto, 6, (255,0,255), -1)

                state = "Tracking"

            else:
                deteccion = False
                state = "Reiniciando"

        cv2.putText(frame, state, (20,40), 0, 1, (0,255,0), 2)

        cv2.imshow("Control Raton PRO", frame)
        scene.render()

        if cv2.waitKey(1) & 0xFF == 13:
            break

    cap.release()
    cv2.destroyAllWindows()
    scene.close()


if __name__ == "__main__":
    main()