import cv2
from cv2 import aruco
import numpy as np
import math


TAMANHO_MARCADOR_CM = 5.0

cap = cv2.VideoCapture(0)

dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_6X6_50
)

detector = aruco.ArucoDetector(dictionary)


while True:

    # captura frame
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    frame_markers = frame.copy()

    if ids is not None:

        aruco.drawDetectedMarkers(
            frame_markers,
            corners,
            ids
        )
    if ids is not None and len(ids) >= 2:

        centros = []

        for corner in corners:

            pontos = corner[0]

            x = int(np.mean(pontos[:, 0]))
            y = int(np.mean(pontos[:, 1]))

            centros.append((x, y))


            cv2.circle(
                frame_markers,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

        x1, y1 = centros[0]
        x2, y2 = centros[1]

        cv2.line(
            frame_markers,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        distancia_pixels = math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

        largura_pixels = np.linalg.norm(
            corners[0][0][0] - corners[0][0][1]
        )

        escala = TAMANHO_MARCADOR_CM / largura_pixels

        distancia_cm = distancia_pixels * escala

        texto = f"{distancia_cm:.2f} cm"

        meio_x = int((x1 + x2) / 2)
        meio_y = int((y1 + y2) / 2)

        cv2.putText(
            frame_markers,
            texto,
            (meio_x, meio_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Metrologia ArUco", frame_markers)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()