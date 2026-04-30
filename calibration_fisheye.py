# Para gerar o tabuleiro: https://calib.io/pages/camera-calibration-pattern-generator
# Configurações: ChArUco / 297 / 210 / 5 / 7 / 40 / DICT_6X6 / 0

import cv2
import numpy as np

def generate_full_frame_fisheye_map(width, height, k1, k2):
    # 1. Definir uma Matriz de Câmera "Fictícia" (K) que preenche o quadro
    f_gen = width * 0.7
    mtx_gen = np.array([[f_gen, 0, width / 2],
                               [0, f_gen, height / 2],
                               [0, 0, 1]], dtype=float)

    # 2. Definir os coeficientes de distorção de Brown (apenas radial)
    dist_gen = np.array([k1, k2, 0, 0, 0], dtype=float)

    # 3. Gerar o mapa de distorção reverso (Mapeamento Inverso)
    # initUndistortRectifyMap gera mapx e mapy para o remap
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx_gen, dist_gen, None, mtx_gen, (width, height), cv2.CV_32FC1
    )
    return mapx, mapy

def main():
    cap = cv2.VideoCapture(0)

    # CONFIGURAÇÃO FISHEYE
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]

        # Gerar o mapa apenas uma vez para desempenho
        # Tente k1 entre -0.1 (leve) e -0.6 (extremo)
        mapx, mapy = generate_full_frame_fisheye_map(w, h, k1=-0.45, k2=0.08)

    # 1. CONFIGURAÇÃO DO TABULEIRO
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.03, aruco_dict)

    # 2. CONFIGURAÇÃO DO DETECTOR
    charuco_detector = cv2.aruco.CharucoDetector(board)

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    fisheye = False

    print("Comandos: 's' para capturar frame | 'q' para finalizar e calcular | 'f' para fisheye")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Aplicar o remapeamento
        if fisheye:
            frame = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        # 3. DETECÇÃO
        # O detector já faz a detecção dos marcadores e a interpolação das quinas
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

        frame_copy = frame.copy()

        # Se detectar marcadores suficientes para formar quinas do ChArUco
        if charuco_ids is not None and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame_copy, charuco_corners, charuco_ids)

        frame_copy = cv2.resize(frame_copy, (800, 600))
        cv2.imshow("Calibracao - ChArUco", frame_copy)

        key = cv2.waitKey(1)
        if key == ord('s') and charuco_ids is not None:
            if len(charuco_ids) > 4:  # Garantir que temos pontos suficientes neste frame
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                print(f"Frame {len(all_charuco_corners)} capturado!")
            else:
                print("Poucos marcadores detectados. Tente outro ângulo.")
        elif key == ord('q'):
            break
        elif key == ord('f'):
            fisheye = not fisheye

    cap.release()
    cv2.destroyAllWindows()

    # 4. CÁLCULO DA CALIBRAÇÃO
    if len(all_charuco_corners) > 10:
        print("\nCalculando parâmetros... isso pode levar alguns segundos.")

        try:
            ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=all_charuco_corners,
                charucoIds=all_charuco_ids,
                board=board,
                imageSize=image_size,
                cameraMatrix=None,
                distCoeffs=None
            )

            # Salvar os resultados
            np.savez("calibracao_webcam_casa.npz", mtx=mtx, dist=dist)

            print("\n--- Calibração Concluída ---")
            print(f"Erro de Re-projeção RMS: {ret:.4f}")
            print(f"Matriz Intrínseca K:\n{mtx}")
            print(f"Vetor de Distorção D:\n{dist}")

        except AttributeError:
            print("Erro: Verifique se o opencv-contrib-python está instalado corretamente.")
    else:
        print("Capturas insuficientes para um resultado preciso. Capturar mais que 10.")



if __name__ == "__main__":
    main()