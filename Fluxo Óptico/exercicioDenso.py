import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# criar imagem HSV para visualizar o fluxo
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while True:

    ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # calcular fluxo óptico denso
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        0.5,   # escala da pirâmide
        3,     # níveis
        15,    # tamanho da janela
        3,     # iterações
        5,     # tamanho da vizinhança
        1.2,   # sigma
        0
    )

    # converter fluxo para magnitude e direção
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Fluxo Optico Denso", rgb)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    gray1 = gray2

cap.release()
cv2.destroyAllWindows()