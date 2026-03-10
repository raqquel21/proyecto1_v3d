import cv2
import sys
import numpy as np

def abrirCamara(url):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("No se pudo conectar")
        return None

    return cap

def trackearMano(captura):
    tracker = cv2.TrackerMIL_create()
    ok, frame = captura.read()
    if not ok:
        print("Cannot read video")
        return

    bbox = cv2.selectROI("Tracking", frame, False)

    tracker.init(frame, bbox)

    while True:

        ok, frame = captura.read()
        if not ok:
            break

        timer = cv2.getTickCount()

        ok, bbox = tracker.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2)
        else:
            cv2.putText(frame,"Tracking failure",(100,80), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)

        cv2.putText(frame,"FPS: "+str(int(fps)),(20,40), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    captura.release()
    cv2.destroyAllWindows()
def main():
    url = "http://:4747/video"
    captura = abrirCamara(url)
    if captura is None:
        return
    trackearMano(captura)

if __name__ == '__main__':
    main()