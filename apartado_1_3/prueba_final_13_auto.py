import os
import cv2
import numpy as np
import open3d as o3d

SOURCE = 0
FRAME_SIZE = (640, 480)
TRACKER_TYPE = "KCF"   # "KCF" o "CSRT"
H_FILE = "homografia_mesa.npz"

# Plano de trabajo (cm)
PLANE_W = 30.0
PLANE_H = 20.0
GOAL_WORLD = (20.0, 10.0)
VIS_SCALE = 10.0

# Deteccion automatica
LEARNING_FRAMES = 60
FIXED_BOX_SIZE = 120
MIN_AREA = 3000
MAX_AREA_RATIO = 0.50

# Mostrar mascara de movimiento para depurar
SHOW_MASK = False


def abrir_camara(source=0):
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


def crear_tracker(tracker_type="KCF"):
    tracker_type = tracker_type.upper()
    if tracker_type == "KCF":
        return cv2.TrackerKCF_create()
    if tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()
    if tracker_type == "MIL":
        return cv2.TrackerMIL_create()
    if tracker_type == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    raise ValueError(f"Tracker no soportado: {tracker_type}")


def cargar_homografia(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra '{path}'. Ejecuta antes calibrar_homografia.py")
    data = np.load(path)
    H_w2i = data["H_w2i"]
    H_i2w = data["H_i2w"]
    err = float(data["reprojection_error"][0])
    return H_w2i, H_i2w, err


def image_to_world(H_i2w, pt):
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    w = cv2.perspectiveTransform(p, H_i2w)[0, 0]
    return float(w[0]), float(w[1])


def world_to_image(H_w2i, world_xy):
    p = np.array([[[float(world_xy[0]), float(world_xy[1])]]], dtype=np.float32)
    q = cv2.perspectiveTransform(p, H_w2i)[0, 0]
    return int(round(q[0])), int(round(q[1]))


def clamp(value, low, high):
    return max(low, min(high, value))


def clamp_bbox(bbox, frame_shape):
    h_img, w_img = frame_shape[:2]
    x, y, w, h = bbox
    x = int(round(x))
    y = int(round(y))
    w = int(round(w))
    h = int(round(h))
    x = clamp(x, 0, w_img - 1)
    y = clamp(y, 0, h_img - 1)
    w = clamp(w, 1, w_img - x)
    h = clamp(h, 1, h_img - y)
    return x, y, w, h


class VirtualScene:
    def __init__(self, goal_world, vis_scale=10.0):
        self.vis_scale = vis_scale
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Visor virtual 1.3 auto", width=640, height=480, visible=True)
        self.vis.get_render_option().background_color = np.asarray([1.0, 1.0, 1.0])

        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])
        self.frame.compute_vertex_normals()

        self.hand = o3d.geometry.TriangleMesh.create_sphere(radius=0.20)
        self.hand.paint_uniform_color([1.0, 0.2, 0.2])
        self.hand.compute_vertex_normals()

        self.goal = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.12)
        self.goal.paint_uniform_color([0.2, 0.2, 1.0])
        self.goal.compute_vertex_normals()
        self.goal.translate([
            goal_world[0] / self.vis_scale,
            goal_world[1] / self.vis_scale,
            0.0,
        ])

        self.vis.add_geometry(self.frame)
        self.vis.add_geometry(self.hand)
        self.vis.add_geometry(self.goal)

        self.current_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.7)

    def update_hand(self, xw, yw):
        new_pos = np.array([xw / self.vis_scale, yw / self.vis_scale, 0.0], dtype=np.float64)
        delta = new_pos - self.current_pos
        self.hand.translate(delta)
        self.current_pos[:] = new_pos
        self.vis.update_geometry(self.hand)

    def render(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


def dibujar_plano_ra(frame, H_w2i):
    corners_world = np.array([
        [0.0, 0.0],
        [PLANE_W, 0.0],
        [PLANE_W, PLANE_H],
        [0.0, PLANE_H],
    ], dtype=np.float32).reshape(-1, 1, 2)

    corners_img = cv2.perspectiveTransform(corners_world, H_w2i)
    corners_img = np.int32(corners_img.reshape(-1, 2))
    cv2.polylines(frame, [corners_img], True, (180, 220, 255), 2)

    goal_img = world_to_image(H_w2i, GOAL_WORLD)
    cv2.circle(frame, goal_img, 10, (255, 0, 255), 2)
    cv2.putText(frame, "OBJ", (goal_img[0] + 10, goal_img[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


def crear_mascara_movimiento(back_sub, frame):
    fgmask = back_sub.apply(frame)
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    return fgmask


def detectar_bbox_inicial(fgmask, frame_shape, fixed_box_size=120):
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    h_img, w_img = frame_shape[:2]
    if not (MIN_AREA < area < (w_img * h_img * MAX_AREA_RATIO)):
        return None

    (x, y), _ = cv2.minEnclosingCircle(c)
    bbox = (x - fixed_box_size / 2, y - fixed_box_size / 2, fixed_box_size, fixed_box_size)
    return clamp_bbox(bbox, frame_shape)


def dibujar_info(frame, tracker_name, fps, reproj_error, contacto, xw, yw, state_text):
    cv2.putText(frame, f"Estado: {state_text}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
    cv2.putText(frame, f"Tracker: {tracker_name}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)
    cv2.putText(frame, f"err_h: {reproj_error:.2f} px", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if contacto is not None:
        cv2.putText(frame, f"Pixel contacto: ({contacto[0]}, {contacto[1]})", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 255, 220), 2)
        cv2.putText(frame, f"Mundo: ({xw:.2f}, {yw:.2f}) cm", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)


def main():
    try:
        H_w2i, H_i2w, reproj_error = cargar_homografia(H_FILE)
    except Exception as e:
        print(e)
        return

    print(f"Homografia cargada. Error medio de reproyeccion: {reproj_error:.3f} px")

    cap = abrir_camara(SOURCE)
    if cap is None:
        print("No se pudo abrir la cámara.")
        return

    tracker = crear_tracker(TRACKER_TYPE)
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    scene = VirtualScene(GOAL_WORLD, VIS_SCALE)

    deteccion_completada = False
    learning_count = 0

    print("\nModo automatico 1.3 iniciado.")
    print("Durante la calibracion de fondo NO metas la mano en escena.")
    print("Pulsa ESC para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer frame.")
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        timer = cv2.getTickCount()
        contacto = None
        xw = yw = None
        state_text = ""

        dibujar_plano_ra(frame, H_w2i)

        if learning_count < LEARNING_FRAMES:
            back_sub.apply(frame)
            learning_count += 1
            state_text = f"Calibrando fondo {learning_count}/{LEARNING_FRAMES}"

        elif not deteccion_completada:
            fgmask = crear_mascara_movimiento(back_sub, frame)
            bbox = detectar_bbox_inicial(fgmask, frame.shape, FIXED_BOX_SIZE)
            state_text = "Buscando mano"

            if SHOW_MASK:
                cv2.imshow("Mascara movimiento", fgmask)

            if bbox is not None:
                tracker = crear_tracker(TRACKER_TYPE)
                tracker.init(frame, bbox)
                deteccion_completada = True
                state_text = "Tracking iniciado"
                print("Deteccion automatica exitosa.")

        else:
            ok_tracker, bbox = tracker.update(frame)
            if ok_tracker:
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                x, y, w, h = clamp_bbox(bbox, frame.shape)
                p1 = (x, y)
                p2 = (x + w, y + h)
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)

                centro = (x + w // 2, y + h // 2)
                contacto = (x + w // 2, y + h)
                cv2.circle(frame, centro, 4, (0, 0, 255), -1)
                cv2.circle(frame, contacto, 5, (255, 0, 255), -1)

                xw, yw = image_to_world(H_i2w, contacto)
                scene.update_hand(xw, yw)
                state_text = "Tracking mano"
                dibujar_info(frame, TRACKER_TYPE, fps, reproj_error, contacto, xw, yw, state_text)
            else:
                deteccion_completada = False
                state_text = "Seguimiento perdido -> reiniciando"
                print("Seguimiento perdido. Reiniciando deteccion.")
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                dibujar_info(frame, TRACKER_TYPE, fps, reproj_error, None, 0.0, 0.0, state_text)
                cv2.imshow("Proyecto 1.3 - automatico", frame)
                scene.render()
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        if contacto is None:
            dibujar_info(frame, TRACKER_TYPE, fps, reproj_error, None, 0.0, 0.0, state_text)

        cv2.imshow("Proyecto 1.3 - automatico", frame)
        scene.render()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    scene.close()


if __name__ == "__main__":
    main()
