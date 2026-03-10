import cv2

def abrirCamara(url):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("No se pudo conectar")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame no recibido")
            break

        cv2.imshow("Stream", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    url = "http://192.168.93.209:4747/video"
    abrirCamara(url)

if __name__ == '__main__':
    main()