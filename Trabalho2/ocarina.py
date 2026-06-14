import cv2
from cv2 import aruco
import numpy as np
import time
import threading

try:
    import winsound
    TEM_WINSOUND = True
except ImportError:
    TEM_WINSOUND = False


# ================= CONFIGURAÇÕES =================

# Pela sua imagem da metrologia:
# ArUco maior = ID 11
# ArUcos menores = IDs 1, 2, 3 e 4
ID_REFERENCIA = 11
IDS_FUROS = [1, 2, 3, 4]

SONS = {
    1: 523,   # Dó
    2: 587,   # Ré
    3: 659,   # Mi
    4: 784    # Sol
}

DURACAO_SOM_MS = 250
COOLDOWN_SOM = 0.6

# ==================================================


def tocar_som(frequencia):
    if TEM_WINSOUND:
        winsound.Beep(frequencia, DURACAO_SOM_MS)
    else:
        print(f"Som: {frequencia} Hz")


def tocar_som_thread(frequencia):
    thread = threading.Thread(target=tocar_som, args=(frequencia,))
    thread.daemon = True
    thread.start()


def projetar_ponto(ponto, homografia):
    ponto_np = np.array([[[ponto[0], ponto[1]]]], dtype=np.float32)
    ponto_proj = cv2.perspectiveTransform(ponto_np, homografia)
    x, y = ponto_proj[0][0]
    return int(x), int(y)


def desenhar_ocarina_3d(frame, marker_corners, furos_ativos):
    pontos_marker = marker_corners.reshape((4, 2)).astype(np.float32)

    cx = int(np.mean(pontos_marker[:, 0]))
    cy = int(np.mean(pontos_marker[:, 1]))

    largura_marker = np.linalg.norm(pontos_marker[0] - pontos_marker[1])

    escala_x = int(largura_marker * 2.2)
    escala_y = int(largura_marker * 1.05)

    centro_x = cx + int(largura_marker * 0.35)
    centro_y = cy - int(largura_marker * 1.15)

    overlay = frame.copy()

    # ================= SOMBRA =================
    cv2.ellipse(
        overlay,
        (centro_x + 10, centro_y + 25),
        (escala_x // 2, escala_y // 2),
        -8,
        0,
        360,
        (35, 35, 35),
        -1
    )

    # ================= CORPO PRINCIPAL =================
    cv2.ellipse(
        overlay,
        (centro_x, centro_y),
        (escala_x // 2, escala_y // 2),
        -8,
        0,
        360,
        (140, 85, 45),
        -1
    )

    cv2.ellipse(
        overlay,
        (centro_x, centro_y),
        (escala_x // 2, escala_y // 2),
        -8,
        0,
        360,
        (255, 255, 255),
        3
    )

    # Parte interna para dar volume
    cv2.ellipse(
        overlay,
        (centro_x - 15, centro_y - 10),
        (int(escala_x * 0.38), int(escala_y * 0.35)),
        -8,
        0,
        360,
        (180, 120, 65),
        2
    )

    # Brilho superior
    cv2.ellipse(
        overlay,
        (centro_x - 35, centro_y - 30),
        (int(escala_x * 0.25), int(escala_y * 0.12)),
        -10,
        200,
        340,
        (230, 190, 120),
        3
    )

    # ================= BOCAL =================
    bocal_x = centro_x + escala_x // 2 - 10
    bocal_y = centro_y + 5

    pts_bocal = np.array([
        [bocal_x - 15, bocal_y - 25],
        [bocal_x + int(largura_marker * 1.0), bocal_y - 18],
        [bocal_x + int(largura_marker * 1.05), bocal_y + 22],
        [bocal_x - 10, bocal_y + 30]
    ], dtype=np.int32)

    cv2.fillPoly(overlay, [pts_bocal], (115, 70, 40))
    cv2.polylines(overlay, [pts_bocal], True, (255, 255, 255), 3)

    # Boca do bocal
    boca_centro = (
        bocal_x + int(largura_marker * 1.0),
        bocal_y + 2
    )

    cv2.ellipse(
        overlay,
        boca_centro,
        (int(largura_marker * 0.22), int(largura_marker * 0.32)),
        5,
        0,
        360,
        (20, 20, 20),
        -1
    )

    cv2.ellipse(
        overlay,
        boca_centro,
        (int(largura_marker * 0.22), int(largura_marker * 0.32)),
        5,
        0,
        360,
        (230, 230, 230),
        2
    )

    # Linhas de profundidade do bocal
    cv2.line(
        overlay,
        (bocal_x - 15, bocal_y - 25),
        (bocal_x + int(largura_marker * 1.0), bocal_y - 18),
        (70, 45, 30),
        2
    )

    cv2.line(
        overlay,
        (bocal_x - 10, bocal_y + 30),
        (bocal_x + int(largura_marker * 1.05), bocal_y + 22),
        (70, 45, 30),
        2
    )

    # ================= LINHAS 3D / WIREFRAME =================
    linhas_3d = [
        ((centro_x - escala_x // 2, centro_y), (centro_x - escala_x // 2 - 35, centro_y + 40)),
        ((centro_x + escala_x // 2, centro_y), (centro_x + escala_x // 2 - 10, centro_y + 40)),
        ((centro_x, centro_y - escala_y // 2), (centro_x - 20, centro_y - escala_y // 2 + 35)),
        ((centro_x, centro_y + escala_y // 2), (centro_x - 15, centro_y + escala_y // 2 + 35)),
    ]

    for p1, p2 in linhas_3d:
        cv2.line(overlay, p1, p2, (90, 90, 90), 2)

    cv2.ellipse(
        overlay,
        (centro_x - 20, centro_y + 35),
        (escala_x // 2, escala_y // 2),
        -8,
        0,
        360,
        (90, 90, 90),
        2
    )

    # ================= FUROS =================
    posicoes_furos = {
        1: (centro_x - int(escala_x * 0.22), centro_y - int(escala_y * 0.05)),
        2: (centro_x, centro_y - int(escala_y * 0.20)),
        3: (centro_x + int(escala_x * 0.22), centro_y - int(escala_y * 0.03)),
        4: (centro_x - int(escala_x * 0.02), centro_y + int(escala_y * 0.23)),
    }

    nomes_notas = {
        1: "DO",
        2: "RE",
        3: "MI",
        4: "SOL"
    }

    for id_furo, (x, y) in posicoes_furos.items():
        ativo = id_furo in furos_ativos

        if ativo:
            cor_furo = (0, 0, 255)
            cor_borda = (255, 255, 255)
            raio = int(largura_marker * 0.18)
        else:
            cor_furo = (10, 10, 10)
            cor_borda = (230, 230, 230)
            raio = int(largura_marker * 0.14)

        # sombra do furo
        cv2.circle(overlay, (x + 5, y + 5), raio + 3, (45, 30, 20), -1)

        # furo
        cv2.circle(overlay, (x, y), raio, cor_furo, -1)
        cv2.circle(overlay, (x, y), raio + 4, cor_borda, 2)

        # brilho do furo
        cv2.circle(
            overlay,
            (x - raio // 3, y - raio // 3),
            max(3, raio // 4),
            (90, 90, 90),
            -1
        )

        if ativo:
            cv2.circle(overlay, (x, y), raio + 9, (0, 0, 255), 2)

        cv2.putText(
            overlay,
            nomes_notas[id_furo],
            (x - 18, y + raio + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2
        )

    # ================= LIGAÇÃO COM O ARUCO =================
    cv2.line(
        overlay,
        (cx, cy),
        (centro_x, centro_y + escala_y // 2),
        (0, 255, 255),
        2
    )

    cv2.circle(overlay, (cx, cy), 6, (0, 255, 255), -1)

    cv2.putText(
        overlay,
        "Instrumento 3D",
        (centro_x - int(escala_x * 0.45), centro_y - escala_y // 2 - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

def iniciar_ocarina():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    detector = aruco.ArucoDetector(dictionary)

    marcadores_ja_vistos = set()
    ultimo_som = {id_furo: 0 for id_furo in IDS_FUROS}

    print("Ocarina iniciada.")
    print("ArUco maior: ID 11")
    print("Furos/sons: IDs 1, 2, 3 e 4")
    print("Pressione ESC para sair.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        ids_detectados = set()
        corner_referencia = None

        if ids is not None:
            ids_lista = ids.flatten()

            aruco.drawDetectedMarkers(frame, corners, ids)

            for i, id_atual in enumerate(ids_lista):
                id_atual = int(id_atual)
                ids_detectados.add(id_atual)

                if id_atual == ID_REFERENCIA:
                    corner_referencia = corners[i]

                pontos = corners[i][0]
                cx = int(np.mean(pontos[:, 0]))
                cy = int(np.mean(pontos[:, 1]))

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                cv2.putText(
                    frame,
                    f"id={id_atual}",
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

        # Só funciona se o ArUco maior estiver aparecendo
        if corner_referencia is not None:

            for id_furo in IDS_FUROS:
                if id_furo in ids_detectados:
                    marcadores_ja_vistos.add(id_furo)

            # Só começa a tocar depois que TODOS os furos já foram vistos pelo menos uma vez
            todos_furos_calibrados = all(
                id_furo in marcadores_ja_vistos for id_furo in IDS_FUROS
            )

            furos_cobertos = []

            if todos_furos_calibrados:
                for id_furo in IDS_FUROS:
                    if id_furo not in ids_detectados:
                        furos_cobertos.append(id_furo)

            agora = time.time()

            for id_furo in furos_cobertos:
                if agora - ultimo_som[id_furo] > COOLDOWN_SOM:
                    tocar_som_thread(SONS[id_furo])
                    ultimo_som[id_furo] = agora

            desenhar_ocarina_3d(frame, corner_referencia, furos_cobertos)

            cv2.putText(
                frame,
                "Ocarina detectada - cubra os ArUcos menores para tocar",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        else:
            # Se o marcador maior sumiu, zera o estado para não tocar sozinho
            marcadores_ja_vistos.clear()

            cv2.putText(
                frame,
                "Mostre o ArUco maior ID 11 para posicionar a ocarina",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        y = 70

        for id_furo in IDS_FUROS:
            if corner_referencia is None:
                status = "aguardando ocarina"
                cor = (0, 255, 255)
            elif id_furo in ids_detectados:
                status = "livre"
                cor = (0, 255, 0)
            elif id_furo in marcadores_ja_vistos:
                status = "coberto / tocando"
                cor = (0, 0, 255)
            else:
                status = "aguardando calibrar"
                cor = (0, 255, 255)

            cv2.putText(
                frame,
                f"Furo ID {id_furo}: {status}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                cor,
                2
            )

            y += 28

        cv2.imshow("Ocarina Virtual com ArUco", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def menu():
    while True:
        print("\n===== MENU OCARINA =====")
        print("1 - Iniciar ocarina virtual")
        print("2 - Sair")

        opcao = input("Escolha: ")

        if opcao == "1":
            iniciar_ocarina()

        elif opcao == "2":
            print("Encerrando...")
            break

        else:
            print("Opcao invalida.")


if __name__ == "__main__":
    menu()