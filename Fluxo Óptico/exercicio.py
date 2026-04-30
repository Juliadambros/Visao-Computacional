import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

#Shi-Tomasi / Harris-Shi-Tomasi
pontos = cv2.goodFeaturesToTrack(gray1,
                                 maxCorners=100,
                                 qualityLevel=0.3,
                                 minDistance=7,
                                 blockSize=7)

# Parâmetros do Lucas-Kanade
lk_params = dict(winSize=(15,15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

# Máscara para desenhar o movimento
mask = np.zeros_like(frame1)

while True:

    ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calcular fluxo óptico
    novos_pontos, status, erro = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, pontos, None, **lk_params)

    # Selecionar pontos válidos
    good_new = novos_pontos[status == 1]
    good_old = pontos[status == 1]

    # Desenhar movimento
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()

        mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), (0,255,0), 2)
        frame2 = cv2.circle(frame2, (int(a),int(b)), 5, (0,0,255), -1)

    resultado = cv2.add(frame2, mask)

    cv2.imshow("Fluxo Optico - Lucas Kanade", resultado)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    gray1 = gray2.copy()
    pontos = good_new.reshape(-1,1,2)

cap.release()
cv2.destroyAllWindows()