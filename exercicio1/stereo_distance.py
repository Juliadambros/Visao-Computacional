import cv2
import numpy as np

# ==============================
# PARÂMETROS DA CÂMERA
# ==============================

focal_length = 721.0   # pixels
baseline = 0.54        # metros

# ==============================
# CARREGAR IMAGENS
# ==============================

imgL = cv2.imread(r"C:\Users\julia\OneDrive\Documentos\visao computacional\exercicio1\Art\view1.png")
imgR = cv2.imread(r"C:\Users\julia\OneDrive\Documentos\visao computacional\exercicio1\Art\view5.png")

grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# ==============================
# StereoBM (mais simples)
# ==============================

stereoBM = cv2.StereoBM_create(
    numDisparities=16*6,
    blockSize=15
)

disparity_bm = stereoBM.compute(grayL, grayR)

dispBM = cv2.normalize(disparity_bm, None, 0, 255, cv2.NORM_MINMAX)
dispBM = np.uint8(dispBM)

# ==============================
# StereoSGBM (melhor resultado)
# ==============================

stereoSGBM = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*6,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

disparity = stereoSGBM.compute(grayL, grayR).astype(np.float32) / 16.0

# evitar divisão por zero
disparity[disparity <= 0] = 0.1

# ==============================
# CALCULAR PROFUNDIDADE
# ==============================

depth = (focal_length * baseline) / disparity

dispSGBM = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
dispSGBM = np.uint8(dispSGBM)

# ==============================
# CLIQUE PARA VER DISTÂNCIA
# ==============================

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        distancia = depth[y, x]

        print(f"Pixel: ({x},{y})")
        print(f"Distância aproximada: {distancia:.2f} metros\n")

        cv2.circle(imgL, (x,y), 5, (0,0,255), -1)
        cv2.imshow("Imagem Esquerda", imgL)

# ==============================
# MOSTRAR RESULTADOS
# ==============================

cv2.imshow("Disparidade - StereoBM", dispBM)
cv2.imshow("Disparidade - StereoSGBM", dispSGBM)
cv2.imshow("Imagem Esquerda", imgL)

cv2.setMouseCallback("Imagem Esquerda", mouse_click)

cv2.waitKey(0)
cv2.destroyAllWindows()