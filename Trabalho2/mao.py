import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def criar_octaedro():
    vertices = np.array([
        [0,  1,  0],   # topo
        [1,  0,  0],   # direita
        [0,  0,  1],   # frente
        [-1, 0,  0],   # esquerda
        [0,  0, -1],   # trás
        [0, -1,  0],   # baixo
    ], dtype=np.float32)

    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [5, 2, 1],
        [5, 3, 2],
        [5, 4, 3],
        [5, 1, 4],
    ]

    return vertices, faces

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

    tamanho = int(distancia * 0.6)
    tamanho = max(25, min(tamanho, 70))

    return tamanho

def matriz_rotacao_y(angulo):
    return np.array([
        [math.cos(angulo), 0, math.sin(angulo)],
        [0, 1, 0],
        [-math.sin(angulo), 0, math.cos(angulo)]
    ], dtype=np.float32)


def matriz_rotacao_x(angulo):
    return np.array([
        [1, 0, 0],
        [0, math.cos(angulo), -math.sin(angulo)],
        [0, math.sin(angulo), math.cos(angulo)]
    ], dtype=np.float32)

def desenhar_octaedro(frame, vertices, faces, cx, cy, tamanho, angulo):

    cor = (0, 0, 0)
    esp = 2

    escala = tamanho

    rot_y = matriz_rotacao_y(angulo)
    rot_x = matriz_rotacao_x(-0.5)

    vertices_rot = vertices @ rot_y.T
    vertices_rot = vertices_rot @ rot_x.T

    pontos_2d = []

    for v in vertices_rot:
        x = int(cx + v[0] * escala)
        y = int(cy - v[1] * escala)
        pontos_2d.append((x, y))

    # desenhar arestas
    for face in faces:
        for i in range(len(face)):
            p1 = pontos_2d[face[i]]
            p2 = pontos_2d[face[(i + 1) % len(face)]]

            cv2.line(frame, p1, p2, cor, esp)

    # centro da palma
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)


def iniciar_camera():

    vertices, faces = criar_octaedro()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir câmera")
        return

    print("Pressione ESC para sair")

    angulo = 0

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

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                cx, cy = calcular_centro_palma(
                    hand_landmarks,
                    largura,
                    altura
                )

                tamanho = calcular_profundidade(
                    hand_landmarks,
                    largura,
                    altura
                )

                desenhar_octaedro(
                    frame,
                    vertices,
                    faces,
                    cx,
                    cy,
                    tamanho,
                    angulo
                )

                cv2.putText(
                    frame,
                    f"Profundidade: {tamanho}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )

        angulo += 0.03

        cv2.imshow("AR - Octaedro 3D", frame)

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