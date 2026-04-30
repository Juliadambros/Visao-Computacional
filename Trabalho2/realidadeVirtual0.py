import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

def calcular_centro_palma(hand_landmarks, largura, altura):

    pontos = [0, 5, 9, 13, 17]

    xs = []
    ys = []

    for p in pontos:
        lm = hand_landmarks.landmark[p]

        xs.append(int(lm.x * largura))
        ys.append(int(lm.y * altura))

    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))

    return cx, cy

def calcular_profundidade(hand_landmarks, largura, altura):

    p5 = hand_landmarks.landmark[5]
    p17 = hand_landmarks.landmark[17]

    x1 = int(p5.x * largura)
    y1 = int(p5.y * altura)

    x2 = int(p17.x * largura)
    y2 = int(p17.y * altura)

    distancia = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    tamanho = int(distancia * 0.45)
    tamanho = max(20, min(tamanho, 70))

    return tamanho

def desenhar_cubo_3d(frame, cx, cy, tamanho):

    metade = tamanho // 2

    offset = int(tamanho * 0.25)

    # face frontal
    frente = np.array([
        [cx - metade, cy - metade],
        [cx + metade, cy - metade],
        [cx + metade, cy + metade],
        [cx - metade, cy + metade]
    ])

    # face traseira
    tras = np.array([
        [cx - metade + offset, cy - metade - offset],
        [cx + metade + offset, cy - metade - offset],
        [cx + metade + offset, cy + metade - offset],
        [cx - metade + offset, cy + metade - offset]
    ])

    frente = frente.astype(int)
    tras = tras.astype(int)

    # desenhar quadrados
    cv2.polylines(frame, [frente], True, (0, 255, 0), 3)
    cv2.polylines(frame, [tras], True, (0, 200, 255), 3)

    # conectar vértices
    for i in range(4):
        cv2.line(
            frame,
            tuple(frente[i]),
            tuple(tras[i]),
            (255, 255, 255),
            2
        )

    # ponto central
    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
def iniciar_camera():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir câmera")
        return

    print("Pressione ESC para sair")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        altura, largura, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resultados = hands.process(rgb)

        if resultados.multi_hand_landmarks:

            for hand_landmarks in resultados.multi_hand_landmarks:

                # desenhar mão
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # centro da palma
                cx, cy = calcular_centro_palma(
                    hand_landmarks,
                    largura,
                    altura
                )

                # profundidade
                tamanho = calcular_profundidade(
                    hand_landmarks,
                    largura,
                    altura
                )

                # desenhar cubo
                desenhar_cubo_3d(frame, cx, cy, tamanho)

        cv2.imshow("AR - Cubo 3D", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def menu():

    while True:

        print("\n===== MENU =====")
        print("1 - Iniciar realidade aumentada")
        print("2 - Sair")

        opcao = input("Escolha: ")

        if opcao == "1":
            iniciar_camera()

        elif opcao == "2":
            print("Encerrando...")
            break

        else:
            print("Opção inválida")

if __name__ == "__main__":
    menu()