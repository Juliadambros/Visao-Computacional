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

    # Configuração fisheye
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]

        # Gerar o mapa apenas uma vez para desempenho
        # Tente k1 entre -0.1 (leve) e -0.6 (extremo)
        mapx, mapy = generate_full_frame_fisheye_map(w, h, k1=-0.45, k2=0.08)

    # 1. Tentar carregar a calibração real
    try:
        data = np.load('calibracao_webcam_casa.npz')
        mtx_real = data['mtx']
        dist_real = data['dist']
        print("Calibração real carregada com sucesso!")
    except:
        print("Arquivo de calibração não encontrado. Usando apenas o 'chute'.")
        mtx_real = None

    # 2. Configurações do ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    use_real_calibration = False
    fisheye = False

    while True:
        ret, frame = cap.read()
        h, w = frame.shape[:2]

        # Aplicar o remapeamento
        if fisheye:
            frame = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)

        # --- DEFINIÇÃO DA MATRIZ GENÉRICA (O "CHUTE") ---
        # Assume que focal_length = largura da imagem e centro = meio da imagem
        focal_gen = w
        mtx_gen = np.array([[focal_gen, 0, w / 2], [0, focal_gen, h / 2], [0, 0, 1]], dtype=float)
        dist_gen = np.zeros((5, 1))  # Assume distorção zero

        # Alternar entre matrizes
        current_mtx = mtx_real if (use_real_calibration and mtx_real is not None) else mtx_gen
        current_dist = dist_real if (use_real_calibration and mtx_real is not None) else dist_gen

        # Detectar marcadores
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            # Pontos 3D do objeto
            # Metade do tamanho do marcador (2cm para um marcador de 4cm)
            s = 0.02
            # Altura do cubo (4cm)
            h = 0.04

            obj_pts = np.float32([
                # BASE (Z=0) - No plano do papel
                [-s, -s, 0],  # 0: Superior-Esquerdo (Top-Left)
                [s, -s, 0],  # 1: Superior-Direito (Top-Right)
                [s, s, 0],  # 2: Inferior-Direito (Bottom-Right)
                [-s, s, 0],  # 3: Inferior-Esquerdo (Bottom-Left)

                # TOPO (Z negativo "sai" do papel em direção à lente)
                [-s, -s, h],  # 4: Superior-Esquerdo Topo
                [s, -s, h],  # 5: Superior-Direito Topo
                [s, s, h],  # 6: Inferior-Direito Topo
                [-s, s, h]  # 7: Inferior-Esquerdo Topo
            ])

            # Pontos de referência do marcador
            ref_3d = np.array([[-0.02, 0.02, 0], [0.02, 0.02, 0], [0.02, -0.02, 0], [-0.02, -0.02, 0]],
                              dtype=np.float32)

            for i in range(len(ids)):
                # Estimativa de Pose
                _, rvec, tvec = cv2.solvePnP(ref_3d, corners[i], current_mtx, current_dist)

                # Projeção
                imgpts, _ = cv2.projectPoints(obj_pts, rvec, tvec, current_mtx, current_dist)
                imgpts = np.int32(imgpts).reshape(-1, 2)

                # Desenhar Cubo (Base verde, Topo azul)
                cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
                for j in range(4): cv2.line(frame, tuple(imgpts[j]), tuple(imgpts[j + 4]), (255, 0, 0), 2)
                cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)

        # UI
        status = "REAL (CALIBRADO)" if use_real_calibration else "GENERICO (CHUTE)"
        cv2.putText(frame, f"Modo: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        frame = cv2.resize(frame, (800, 600))
        cv2.imshow("AR - Pressione 'c' para alternar calibracao | 'f' para fisheye | 'q' para sair", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'): use_real_calibration = not use_real_calibration
        if key == ord('q'): break
        if key == ord('f'): fisheye = not fisheye

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()