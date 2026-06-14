import cv2
from cv2 import aruco
import numpy as np
import math

# tamanho real do marcador em centímetros
TAMANHO_MARCADOR_CM = 5.0

# abre webcam
cap = cv2.VideoCapture(0)

# dicionário ArUco
dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_6X6_50
)

# detector
detector = aruco.ArucoDetector(dictionary)

while True:

    # captura frame
    ret, frame = cap.read()

    if not ret:
        break

    # converte para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecta marcadores
    corners, ids, rejected = detector.detectMarkers(gray)

    frame_markers = frame.copy()

    # desenha marcadores detectados
    if ids is not None:

        aruco.drawDetectedMarkers(
            frame_markers,
            corners,
            ids
        )

        # transforma ids em vetor simples
        ids = ids.flatten()

        # dicionário para armazenar centros
        centros = {}

        # percorre marcadores detectados
        for corner, marker_id in zip(corners, ids):

            # mostra somente IDs 1 e 2
            if marker_id not in [1, 2]:
                continue

            pontos = corner[0]

            # calcula centro do marcador
            x = int(np.mean(pontos[:, 0]))
            y = int(np.mean(pontos[:, 1]))

            # salva centro e canto
            centros[marker_id] = (x, y, corner)

            # desenha centro
            cv2.circle(
                frame_markers,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )


        # verifica se IDs 1 e 2 foram detectados
        if 1 in centros and 2 in centros:

            x1, y1, corner1 = centros[1]
            x2, y2, corner2 = centros[2]

            # desenha linha entre marcadores
            cv2.line(
                frame_markers,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )

            # distância em pixels
            distancia_pixels = math.sqrt(
                (x2 - x1) ** 2 +
                (y2 - y1) ** 2
            )

            # largura do marcador em pixels
            largura_pixels = np.linalg.norm(
                corner1[0][0] - corner1[0][1]
            )

            # conversão pixel -> cm
            escala = TAMANHO_MARCADOR_CM / largura_pixels

            # distância em cm
            distancia_cm = distancia_pixels * escala

            texto = f"{distancia_cm:.2f} cm"

            # posição do texto
            meio_x = int((x1 + x2) / 2)
            meio_y = int((y1 + y2) / 2)

            # escreve distância
            cv2.putText(
                frame_markers,
                texto,
                (meio_x - 60, meio_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

    # mostra resultado
    cv2.imshow("Metrologia ArUco", frame_markers)

    # tecla q para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# libera recursos
cap.release()
cv2.destroyAllWindows()