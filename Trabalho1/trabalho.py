import os
import csv
import cv2
import time
import numpy as np
import pyautogui
import tkinter as tk
import glob
from tkinter import filedialog, messagebox, ttk
from datetime import datetime

MIN_MATCH_COUNT = 10 #mínimo de correspondências para tentar calcular homografia
RATIO_TEST = 0.75 #filtra matches ambíguos
GESTURE_THRESHOLD = 2.5         # menor = mais sensível, sensibilidade mínima por frame
GESTURE_ACCUM_THRESHOLD = 20    # confirmação do gesto só depois de movimento suficiente
GESTURE_COOLDOWN = 0.8          # segundos entre comandos,evita múltiplos comandos consecutivos
QUALITY_LEVEL = 0.2
MIN_DISTANCE = 7
BLOCK_SIZE = 7
MAX_CORNERS = 150 #quantidade máxima de pontos rastreados na mão
MAX_MATCHES_DRAW = 80
DISPLAY_HEIGHT = 500

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "resultados_trabalho")
PANORAMA_DIR = os.path.join(BASE_OUTPUT_DIR, "panoramicas")
GESTURE_DIR = os.path.join(BASE_OUTPUT_DIR, "gestos")
REPORTS_DIR = os.path.join(BASE_OUTPUT_DIR, "relatorios")


def garantir_pastas_saida():
    for pasta in [BASE_OUTPUT_DIR, PANORAMA_DIR, GESTURE_DIR, REPORTS_DIR]:
        os.makedirs(pasta, exist_ok=True)

def redimensionar_para_altura(img, altura=600):
    h, w = img.shape[:2]
    escala = altura / h
    nova_largura = int(w * escala)
    return cv2.resize(img, (nova_largura, altura))


def redimensionar_para_exibicao(img, altura=DISPLAY_HEIGHT):
    h, w = img.shape[:2]
    if h <= altura:
        return img
    escala = altura / h
    nova_largura = int(w * escala)
    return cv2.resize(img, (nova_largura, altura))


def recortar_bordas_pretas(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y + h, x:x + w]


def ajustar_cor(img_base, img_ajustar):
    #ajustando média e desvio padrão de cada canal RGB
    base = img_base.astype(np.float32)
    ajustar = img_ajustar.astype(np.float32)

    for i in range(3):
        media_base, std_base = base[:, :, i].mean(), base[:, :, i].std()
        media_aj, std_aj = ajustar[:, :, i].mean(), ajustar[:, :, i].std()

        if std_aj > 0:
            ajustar[:, :, i] = (ajustar[:, :, i] - media_aj) * (std_base / std_aj) + media_base

    return np.clip(ajustar, 0, 255).astype(np.uint8)


def estimar_qualidade_panorama(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    #A nitidez foi estimada pela variância do Laplaciano. O Laplaciano destaca bordas,
    #então, quando a imagem tem mais detalhes e menos borramento, a variância tende a ser maior.
    contraste = gray.std()
    #O contraste foi medido pelo desvio padrão dos níveis de cinza. 
    #Quanto maior a variação entre pixels claros e escuros, maior o contraste
    return {
        "nitidez_laplaciano": float(lap_var),
        "contraste_std": float(contraste),
        "largura": int(img.shape[1]),
        "altura": int(img.shape[0])
    }


def salvar_resumo_csv(resumo, caminho_csv):
    if not resumo:
        return

    campos = [
        "algoritmos", "tempo_s", "kp_img1", "kp_img2", "matches_bons",
        "nitidez_laplaciano", "contraste_std", "largura", "altura",
        "arquivo_panorama", "arquivo_matches"
    ]

    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for item in resumo:
            writer.writerow(item)

#panorâmica
def criar_detector(nome):
    nome = nome.upper()

    if nome == "ORB":
        return cv2.ORB_create(nfeatures=4000)

    if nome == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "SIFT não está disponível nesta instalação do OpenCV. "
                "Instale: pip install opencv-contrib-python"
            )
        return cv2.SIFT_create()

    raise ValueError("Detector inválido. Use ORB ou SIFT.")



def criar_matcher(nome_detector, nome_matcher):
    nome_detector = nome_detector.upper()
    nome_matcher = nome_matcher.upper()

    if nome_matcher == "BF":
        if nome_detector == "ORB":
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    if nome_matcher == "FLANN":
        if nome_detector == "ORB":
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)

    raise ValueError("Matcher inválido. Use BF ou FLANN.")



def carregar_imagens(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem 1: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem 2: {img2_path}")

    img1 = redimensionar_para_altura(img1, 600)
    img2 = redimensionar_para_altura(img2, 600)
    return img1, img2

def detectar_e_descrever(img, detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(gray, None)
    return kp, des

def encontrar_boas_correspondencias(des1, des2, matcher):
    if des1 is None or des2 is None:
        return []

    matches = matcher.knnMatch(des1, des2, k=2)
    good = []

    for par in matches:
        if len(par) < 2:
            continue
        m, n = par
        if m.distance < RATIO_TEST * n.distance:
            good.append(m)

    return good

def desenhar_matches(img1, kp1, img2, kp2, good_matches, max_matches=MAX_MATCHES_DRAW):
    draw_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]
    img = cv2.drawMatches(
        img1, kp1, img2, kp2, draw_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return img



def criar_panorama(img1, img2, detector_name="ORB", matcher_name="BF"):
    img2 = ajustar_cor(img1, img2)
    detector = criar_detector(detector_name)
    matcher = criar_matcher(detector_name, matcher_name)

    inicio = time.time()

    kp1, des1 = detectar_e_descrever(img1, detector)
    kp2, des2 = detectar_e_descrever(img2, detector)
    good_matches = encontrar_boas_correspondencias(des1, des2, matcher)

    if len(good_matches) < MIN_MATCH_COUNT:
        raise RuntimeError(
            f"Poucas correspondências encontradas: {len(good_matches)}. "
            f"Mínimo necessário: {MIN_MATCH_COUNT}."
        )

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask_h = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Não foi possível calcular a homografia.")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    cantos_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    cantos_img2_transformados = cv2.perspectiveTransform(cantos_img2, H)
    cantos_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    todos_cantos = np.concatenate((cantos_img1, cantos_img2_transformados), axis=0)
    [xmin, ymin] = np.int32(todos_cantos.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(todos_cantos.max(axis=0).ravel() + 0.5)

    translacao = [-xmin, -ymin]
    T = np.array([
        [1, 0, translacao[0]],
        [0, 1, translacao[1]],
        [0, 0, 1]
    ])

    panorama = cv2.warpPerspective(img2, T @ H, (xmax - xmin, ymax - ymin))
    panorama[translacao[1]:h1 + translacao[1], translacao[0]:w1 + translacao[0]] = img1
    panorama = recortar_bordas_pretas(panorama)

    img_matches = desenhar_matches(img1, kp1, img2, kp2, good_matches)
    tempo_total = time.time() - inicio
    qualidade = estimar_qualidade_panorama(panorama)

    return {
        "panorama": panorama,
        "matches_img": img_matches,
        "tempo": tempo_total,
        "num_kp1": len(kp1),
        "num_kp2": len(kp2),
        "num_matches": len(good_matches),
        "qualidade": qualidade,
    }



def salvar_resultado_panorama(resultado, detector_name, matcher_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_base = f"{detector_name.upper()}_{matcher_name.upper()}_{timestamp}"

    panorama_path = os.path.join(PANORAMA_DIR, f"panorama_{nome_base}.jpg")
    matches_path = os.path.join(PANORAMA_DIR, f"matches_{nome_base}.jpg")

    cv2.imwrite(panorama_path, resultado["panorama"])
    cv2.imwrite(matches_path, resultado["matches_img"])

    return panorama_path, matches_path



def executar_panorama(img1_path, img2_path, detector_name, matcher_name, mostrar_janelas=True):
    img1, img2 = carregar_imagens(img1_path, img2_path)
    resultado = criar_panorama(img1, img2, detector_name, matcher_name)
    panorama_path, matches_path = salvar_resultado_panorama(resultado, detector_name, matcher_name)

    resumo = {
        "algoritmos": f"{detector_name.upper()} + {matcher_name.upper()}",
        "tempo_s": round(resultado["tempo"], 4),
        "kp_img1": resultado["num_kp1"],
        "kp_img2": resultado["num_kp2"],
        "matches_bons": resultado["num_matches"],
        "nitidez_laplaciano": round(resultado["qualidade"]["nitidez_laplaciano"], 4),
        "contraste_std": round(resultado["qualidade"]["contraste_std"], 4),
        "largura": resultado["qualidade"]["largura"],
        "altura": resultado["qualidade"]["altura"],
        "arquivo_panorama": panorama_path,
        "arquivo_matches": matches_path,
    }

    if mostrar_janelas:
        cv2.imshow(f"Matches - {detector_name}_{matcher_name}", redimensionar_para_exibicao(resultado["matches_img"]))
        cv2.imshow(f"Panorama - {detector_name}_{matcher_name}", redimensionar_para_exibicao(resultado["panorama"]))

    return resumo, resultado



def executar_todas_combinacoes(img1_path, img2_path):
    garantir_pastas_saida()

    combinacoes = [
        ("ORB", "BF"),
        ("ORB", "FLANN"),
        ("SIFT", "BF"),
        ("SIFT", "FLANN"),
    ]

    resumo_geral = []
    erros = []

    for detector_name, matcher_name in combinacoes:
        try:
            resumo, _ = executar_panorama(img1_path, img2_path, detector_name, matcher_name, mostrar_janelas=False)
            resumo_geral.append(resumo)
        except Exception as e:
            erros.append(f"{detector_name} + {matcher_name}: {e}")

    csv_path = os.path.join(REPORTS_DIR, f"resumo_panoramas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    salvar_resumo_csv(resumo_geral, csv_path)

    return resumo_geral, erros, csv_path



def criar_mascara_central(gray):
    h, w = gray.shape[:2]
    mascara = np.zeros_like(gray)

    # região central onde a mão deve aparecer
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)
    y1 = int(h * 0.20)
    y2 = int(h * 0.85)
    mascara[y1:y2, x1:x2] = 255
    return mascara, (x1, y1, x2, y2)


def inicializar_pontos(gray):
    mascara, _ = criar_mascara_central(gray)
    return cv2.goodFeaturesToTrack(
        gray,
        mask=mascara,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE
    )



def interface_gestual(status_callback=None):
    garantir_pastas_saida()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível acessar a câmera.")

    ret, old_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Não foi possível capturar a imagem inicial da câmera.")

    old_frame = cv2.flip(old_frame, 1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = inicializar_pontos(old_gray)
    trilha = np.zeros_like(old_frame)
    ultimo_comando = 0
    ultimo_texto = "Aguardando gesto..."
    acumulado_dx = 0.0

    log_path = os.path.join(GESTURE_DIR, f"gestos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gesto", "deslocamento_medio_x", "deslocamento_acumulado_x"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mascara_roi, (x1, y1, x2, y2) = criar_mascara_central(frame_gray)

        if p0 is None or len(p0) == 0:
            p0 = inicializar_pontos(frame_gray)
            old_gray = frame_gray.copy()
            trilha = np.zeros_like(frame)
            acumulado_dx = 0.0

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        deslocamento_medio_x = 0.0
        combinado = frame.copy()
        cv2.rectangle(combinado, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(combinado, "Mova a mao dentro da caixa", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if p1 is not None and st is not None:
            pontos_novos = p1[st == 1]
            pontos_antigos = p0[st == 1]

            bons_novos = []
            bons_antigos = []

            for new, old in zip(pontos_novos, pontos_antigos):
                a, b = new.ravel()
                c, d = old.ravel()

                # considera só pontos dentro da região central
                if x1 <= a <= x2 and y1 <= b <= y2:
                    bons_novos.append([a, b])
                    bons_antigos.append([c, d])

            if len(bons_novos) > 0 and len(bons_antigos) > 0:
                good_new = np.array(bons_novos, dtype=np.float32)
                good_old = np.array(bons_antigos, dtype=np.float32)

                deslocamentos_x = good_new[:, 0] - good_old[:, 0]
                deslocamento_medio_x = float(np.mean(deslocamentos_x))

                if abs(deslocamento_medio_x) >= GESTURE_THRESHOLD:
                    acumulado_dx += deslocamento_medio_x
                else:
                    acumulado_dx *= 0.6

                for new, old in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a, b, c, d = int(a), int(b), int(c), int(d)
                    trilha = cv2.line(trilha, (c, d), (a, b), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (a, b), 4, (0, 0, 255), -1)

                agora = time.time()
                gesto_detectado = None
                acumulado_registro = acumulado_dx

                if agora - ultimo_comando > GESTURE_COOLDOWN:
                    if acumulado_dx >= GESTURE_ACCUM_THRESHOLD:
                        pyautogui.press("right")
                        gesto_detectado = "direita"
                        ultimo_texto = "GESTO: DIREITA -> AVANCAR SLIDE"
                        ultimo_comando = agora
                        acumulado_registro = acumulado_dx
                        acumulado_dx = 0.0
                    elif acumulado_dx <= -GESTURE_ACCUM_THRESHOLD:
                        pyautogui.press("left")
                        gesto_detectado = "esquerda"
                        ultimo_texto = "GESTO: ESQUERDA -> VOLTAR SLIDE"
                        ultimo_comando = agora
                        acumulado_registro = acumulado_dx
                        acumulado_dx = 0.0

                if gesto_detectado is not None:
                    with open(log_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            gesto_detectado,
                            round(deslocamento_medio_x, 4),
                            round(acumulado_registro, 4)
                        ])
                    if status_callback:
                        status_callback(f"Gesto detectado: {gesto_detectado}")

                combinado = cv2.add(frame, trilha)
                cv2.rectangle(combinado, (x1, y1), (x2, y2), (255, 255, 0), 2)
            else:
                p0 = inicializar_pontos(frame_gray)
                trilha = np.zeros_like(frame)
                acumulado_dx = 0.0
                combinado = frame.copy()
                cv2.rectangle(combinado, (x1, y1), (x2, y2), (255, 255, 0), 2)

            old_gray = frame_gray.copy()
            if len(bons_novos) > 0:
                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = inicializar_pontos(frame_gray)
        else:
            p0 = inicializar_pontos(frame_gray)
            old_gray = frame_gray.copy()
            acumulado_dx = 0.0

        cv2.putText(combinado, "q = sair | r = recalibrar", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combinado, f"DX medio: {deslocamento_medio_x:.2f}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(combinado, f"DX acumulado: {acumulado_dx:.2f}", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(combinado, ultimo_texto, (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Interface Gestual - Lucas Kanade", combinado)
        tecla = cv2.waitKey(30) & 0xFF

        if tecla == ord('r'):
            old_gray = frame_gray.copy()
            p0 = inicializar_pontos(old_gray)
            trilha = np.zeros_like(frame)
            acumulado_dx = 0.0
            ultimo_texto = "Pontos recalibrados"
            if status_callback:
                status_callback("Pontos recalibrados")

        elif tecla == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if status_callback:
        status_callback(f"Log salvo em: {log_path}")

    return log_path


SLIDES_WINDOW_NAME = "Slides por Gestos"

def listar_imagens_pasta(pasta):
    extensoes = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    arquivos = []
    for ext in extensoes:
        arquivos.extend(glob.glob(os.path.join(pasta, ext)))
        arquivos.extend(glob.glob(os.path.join(pasta, ext.upper())))
    return sorted(arquivos)


def preparar_slide(img, largura=1200, altura=700):
    canvas = np.zeros((altura, largura, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    escala = min(largura / w, altura / h)
    novo_w = max(1, int(w * escala))
    novo_h = max(1, int(h * escala))
    redim = cv2.resize(img, (novo_w, novo_h))
    x = (largura - novo_w) // 2
    y = (altura - novo_h) // 2
    canvas[y:y + novo_h, x:x + novo_w] = redim
    return canvas


def mostrar_slide(lista_imagens, indice):
    if not lista_imagens:
        return
    img = cv2.imread(lista_imagens[indice])
    if img is None:
        return
    slide = preparar_slide(img)
    texto = f"Slide {indice + 1}/{len(lista_imagens)}"
    nome_arquivo = os.path.basename(lista_imagens[indice])
    cv2.putText(slide, texto, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(slide, nome_arquivo, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow(SLIDES_WINDOW_NAME, slide)


def interface_gestual_slides(pasta_slides, status_callback=None):
    garantir_pastas_saida()
    lista_imagens = listar_imagens_pasta(pasta_slides)
    if not lista_imagens:
        raise RuntimeError("Nenhuma imagem encontrada na pasta selecionada.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível acessar a câmera.")

    ret, old_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Não foi possível capturar a imagem inicial da câmera.")

    indice_slide = 0
    mostrar_slide(lista_imagens, indice_slide)

    old_frame = cv2.flip(old_frame, 1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = inicializar_pontos(old_gray)
    trilha = np.zeros_like(old_frame)
    ultimo_comando = 0
    ultimo_texto = "Aguardando gesto..."
    acumulado_dx = 0.0

    log_path = os.path.join(GESTURE_DIR, f"gestos_slides_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gesto", "deslocamento_medio_x", "deslocamento_acumulado_x", "slide_atual"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, (x1, y1, x2, y2) = criar_mascara_central(frame_gray)

        if p0 is None or len(p0) == 0:
            p0 = inicializar_pontos(frame_gray)
            old_gray = frame_gray.copy()
            trilha = np.zeros_like(frame)
            acumulado_dx = 0.0

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        deslocamento_medio_x = 0.0
        combinado = frame.copy()
        cv2.rectangle(combinado, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(combinado, "Mova a mao dentro da caixa", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if p1 is not None and st is not None:
            pontos_novos = p1[st == 1]
            pontos_antigos = p0[st == 1]

            bons_novos = []
            bons_antigos = []
            for new, old in zip(pontos_novos, pontos_antigos):
                a, b = new.ravel()
                c, d = old.ravel()
                if x1 <= a <= x2 and y1 <= b <= y2:
                    bons_novos.append([a, b])
                    bons_antigos.append([c, d])

            if len(bons_novos) > 0 and len(bons_antigos) > 0:
                good_new = np.array(bons_novos, dtype=np.float32)
                good_old = np.array(bons_antigos, dtype=np.float32)
                deslocamentos_x = good_new[:, 0] - good_old[:, 0]
                deslocamento_medio_x = float(np.mean(deslocamentos_x))

                if abs(deslocamento_medio_x) >= GESTURE_THRESHOLD:
                    acumulado_dx += deslocamento_medio_x
                else:
                    acumulado_dx *= 0.6

                for new, old in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a, b, c, d = int(a), int(b), int(c), int(d)
                    trilha = cv2.line(trilha, (c, d), (a, b), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (a, b), 4, (0, 0, 255), -1)

                agora = time.time()
                gesto_detectado = None
                acumulado_registro = acumulado_dx

                if agora - ultimo_comando > GESTURE_COOLDOWN:
                    if acumulado_dx >= GESTURE_ACCUM_THRESHOLD:
                        pyautogui.press("right")
                        gesto_detectado = "direita"
                        ultimo_texto = "GESTO: DIREITA -> PROXIMO SLIDE"
                        ultimo_comando = agora
                        acumulado_registro = acumulado_dx
                        acumulado_dx = 0.0
                        indice_slide = min(indice_slide + 1, len(lista_imagens) - 1)
                        mostrar_slide(lista_imagens, indice_slide)
                    elif acumulado_dx <= -GESTURE_ACCUM_THRESHOLD:
                        pyautogui.press("left")
                        gesto_detectado = "esquerda"
                        ultimo_texto = "GESTO: ESQUERDA -> SLIDE ANTERIOR"
                        ultimo_comando = agora
                        acumulado_registro = acumulado_dx
                        acumulado_dx = 0.0
                        indice_slide = max(indice_slide - 1, 0)
                        mostrar_slide(lista_imagens, indice_slide)

                if gesto_detectado is not None:
                    with open(log_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            gesto_detectado,
                            round(deslocamento_medio_x, 4),
                            round(acumulado_registro, 4),
                            indice_slide + 1
                        ])
                    if status_callback:
                        status_callback(f"Gesto detectado: {gesto_detectado} | Slide {indice_slide + 1}")

                combinado = cv2.add(frame, trilha)
                cv2.rectangle(combinado, (x1, y1), (x2, y2), (255, 255, 0), 2)
            else:
                p0 = inicializar_pontos(frame_gray)
                trilha = np.zeros_like(frame)
                acumulado_dx = 0.0
                combinado = frame.copy()
                cv2.rectangle(combinado, (x1, y1), (x2, y2), (255, 255, 0), 2)

            old_gray = frame_gray.copy()
            if len(bons_novos) > 0:
                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = inicializar_pontos(frame_gray)
        else:
            p0 = inicializar_pontos(frame_gray)
            old_gray = frame_gray.copy()
            acumulado_dx = 0.0

        cv2.putText(combinado, "q = sair | r = recalibrar", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combinado, f"DX medio: {deslocamento_medio_x:.2f}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(combinado, f"DX acumulado: {acumulado_dx:.2f}", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(combinado, ultimo_texto, (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combinado, f"Slide atual: {indice_slide + 1}/{len(lista_imagens)}", (15, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Interface Gestual - Lucas Kanade", combinado)
        tecla = cv2.waitKey(30) & 0xFF

        if tecla == ord('r'):
            old_gray = frame_gray.copy()
            p0 = inicializar_pontos(old_gray)
            trilha = np.zeros_like(frame)
            acumulado_dx = 0.0
            ultimo_texto = "Pontos recalibrados"
            if status_callback:
                status_callback("Pontos recalibrados")
        elif tecla == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if status_callback:
        status_callback(f"Log salvo em: {log_path}")
    return log_path


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Trabalho de Visão Computacional - OpenCV")
        self.root.geometry("900x620")
        self.root.resizable(False, False)

        garantir_pastas_saida()

        self.img1_path = tk.StringVar()
        self.img2_path = tk.StringVar()
        self.algoritmo_var = tk.StringVar(value="ORB + BF")
        self.status_var = tk.StringVar(value="Selecione duas imagens para gerar a panorâmica.")
        self.pasta_slides_var = tk.StringVar()

        self.criar_interface()

    def criar_interface(self):
        titulo = tk.Label(
            self.root,
            text="Trabalho Prático - Panorâmica e Interface Gestual",
            font=("Arial", 16, "bold")
        )
        titulo.pack(pady=10)

        subtitulo = tk.Label(
            self.root,
            text="Selecione as imagens, escolha o algoritmo e execute.\nOs resultados serão salvos automaticamente na pasta resultados_trabalho.",
            font=("Arial", 10),
            justify="center"
        )
        subtitulo.pack(pady=5)

        frame_arquivos = tk.LabelFrame(self.root, text="1) Seleção das imagens", padx=10, pady=10)
        frame_arquivos.pack(fill="x", padx=15, pady=10)

        tk.Label(frame_arquivos, text="Imagem 1:").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(frame_arquivos, textvariable=self.img1_path, width=75).grid(row=0, column=1, padx=5)
        tk.Button(frame_arquivos, text="Procurar...", command=self.selecionar_img1).grid(row=0, column=2, padx=5)

        tk.Label(frame_arquivos, text="Imagem 2:").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(frame_arquivos, textvariable=self.img2_path, width=75).grid(row=1, column=1, padx=5)
        tk.Button(frame_arquivos, text="Procurar...", command=self.selecionar_img2).grid(row=1, column=2, padx=5)

        frame_alg = tk.LabelFrame(self.root, text="2) Escolha do algoritmo", padx=10, pady=10)
        frame_alg.pack(fill="x", padx=15, pady=10)

        ttk.Combobox(
            frame_alg,
            textvariable=self.algoritmo_var,
            values=["ORB + BF", "ORB + FLANN", "SIFT + BF", "SIFT + FLANN"],
            state="readonly",
            width=25
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        frame_botoes = tk.LabelFrame(self.root, text="3) Execução", padx=10, pady=10)
        frame_botoes.pack(fill="x", padx=15, pady=10)

        tk.Button(frame_botoes, text="Gerar panorâmica selecionada", width=28, command=self.gerar_panorama_selecionada).grid(row=0, column=0, padx=8, pady=8)
        tk.Button(frame_botoes, text="Executar as 4 combinações", width=28, command=self.executar_todas).grid(row=0, column=1, padx=8, pady=8)
        tk.Button(frame_botoes, text="Abrir interface gestual", width=28, command=self.abrir_interface_gestual).grid(row=0, column=2, padx=8, pady=8)

        frame_slides = tk.LabelFrame(self.root, text="4) Slides por gestos", padx=10, pady=10)
        frame_slides.pack(fill="x", padx=15, pady=10)

        tk.Label(frame_slides, text="Pasta com imagens dos slides:").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(frame_slides, textvariable=self.pasta_slides_var, width=65).grid(row=0, column=1, padx=5)
        tk.Button(frame_slides, text="Procurar pasta...", command=self.selecionar_pasta_slides).grid(row=0, column=2, padx=5)
        tk.Button(frame_slides, text="Iniciar slides por gestos", width=24, command=self.abrir_interface_gestual_slides).grid(row=1, column=1, pady=8)

        frame_info = tk.LabelFrame(self.root, text="5) Informações para o relatório", padx=10, pady=10)
        frame_info.pack(fill="both", expand=True, padx=15, pady=10)

        self.texto_info = tk.Text(frame_info, height=18, wrap="word", font=("Consolas", 10))
        self.texto_info.pack(fill="both", expand=True)
        self.texto_info.insert("end", "Os resultados aparecerão aqui.\n")
        self.texto_info.config(state="disabled")

        frame_status = tk.Frame(self.root)
        frame_status.pack(fill="x", padx=15, pady=10)
        tk.Label(frame_status, textvariable=self.status_var, anchor="w", fg="blue").pack(fill="x")

    def adicionar_texto(self, texto):
        self.texto_info.config(state="normal")
        self.texto_info.insert("end", texto + "\n")
        self.texto_info.see("end")
        self.texto_info.config(state="disabled")

    def selecionar_img1(self):
        path = filedialog.askopenfilename(
            title="Selecione a primeira imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
        )
        if path:
            self.img1_path.set(path)
            self.status_var.set("Imagem 1 selecionada.")

    def selecionar_img2(self):
        path = filedialog.askopenfilename(
            title="Selecione a segunda imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
        )
        if path:
            self.img2_path.set(path)
            self.status_var.set("Imagem 2 selecionada.")

    def selecionar_pasta_slides(self):
        path = filedialog.askdirectory(title="Selecione a pasta com as imagens dos slides")
        if path:
            self.pasta_slides_var.set(path)
            self.status_var.set("Pasta de slides selecionada.")

    def validar_imagens(self):
        if not self.img1_path.get() or not self.img2_path.get():
            messagebox.showwarning("Aviso", "Selecione as duas imagens antes de executar.")
            return False
        return True

    def obter_detector_matcher(self):
        detector, matcher = self.algoritmo_var.get().split(" + ")
        return detector.strip(), matcher.strip()

    def gerar_panorama_selecionada(self):
        if not self.validar_imagens():
            return

        detector, matcher = self.obter_detector_matcher()

        try:
            self.status_var.set(f"Executando {detector} + {matcher}...")
            self.root.update_idletasks()

            resumo, resultado = executar_panorama(self.img1_path.get(), self.img2_path.get(), detector, matcher, mostrar_janelas=True)

            self.adicionar_texto("=" * 80)
            self.adicionar_texto(f"Algoritmos: {resumo['algoritmos']}")
            self.adicionar_texto(f"Tempo de processamento: {resumo['tempo_s']} s")
            self.adicionar_texto(f"Keypoints imagem 1: {resumo['kp_img1']}")
            self.adicionar_texto(f"Keypoints imagem 2: {resumo['kp_img2']}")
            self.adicionar_texto(f"Boas correspondências: {resumo['matches_bons']}")
            self.adicionar_texto(f"Nitidez (variância do Laplaciano): {resumo['nitidez_laplaciano']}")
            self.adicionar_texto(f"Contraste (desvio padrão): {resumo['contraste_std']}")
            self.adicionar_texto(f"Tamanho panorama: {resumo['largura']} x {resumo['altura']}")
            self.adicionar_texto(f"Panorama salvo em: {resumo['arquivo_panorama']}")
            self.adicionar_texto(f"Matches salvos em: {resumo['arquivo_matches']}")
            self.adicionar_texto("Janela aberta com matches e panorama. Feche as janelas do OpenCV para continuar.\n")

            self.status_var.set("Panorâmica gerada com sucesso.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            messagebox.showerror("Erro", str(e))
            self.status_var.set("Erro ao gerar a panorâmica.")

    def executar_todas(self):
        if not self.validar_imagens():
            return

        try:
            self.status_var.set("Executando as 4 combinações...")
            self.root.update_idletasks()

            resumo_geral, erros, csv_path = executar_todas_combinacoes(self.img1_path.get(), self.img2_path.get())

            self.adicionar_texto("=" * 80)
            self.adicionar_texto("RESUMO DAS 4 COMBINAÇÕES")
            for item in resumo_geral:
                self.adicionar_texto(
                    f"{item['algoritmos']} | tempo={item['tempo_s']} s | "
                    f"matches={item['matches_bons']} | nitidez={item['nitidez_laplaciano']} | "
                    f"contraste={item['contraste_std']}"
                )
                self.adicionar_texto(f"Panorama: {item['arquivo_panorama']}")
                self.adicionar_texto(f"Matches: {item['arquivo_matches']}")
                self.adicionar_texto("-" * 80)

            if erros:
                self.adicionar_texto("ERROS ENCONTRADOS:")
                for erro in erros:
                    self.adicionar_texto(erro)

            self.adicionar_texto(f"Resumo CSV salvo em: {csv_path}\n")
            self.status_var.set("As 4 combinações foram executadas.")
            messagebox.showinfo("Concluído", f"Execução finalizada.\nResumo salvo em:\n{csv_path}")

        except Exception as e:
            messagebox.showerror("Erro", str(e))
            self.status_var.set("Erro ao executar as combinações.")

    def abrir_interface_gestual(self):
        try:
            self.status_var.set("Interface gestual iniciada. Use q para sair e r para recalibrar.")
            self.root.update_idletasks()
            log_path = interface_gestual(status_callback=self.status_var.set)
            self.adicionar_texto("=" * 80)
            self.adicionar_texto("INTERFACE GESTUAL FINALIZADA")
            self.adicionar_texto(f"Log de gestos salvo em: {log_path}\n")
            messagebox.showinfo("Finalizado", f"Interface gestual finalizada.\nLog salvo em:\n{log_path}")
        except Exception as e:
            messagebox.showerror("Erro na interface gestual", str(e))
            self.status_var.set("Erro na interface gestual.")

    def abrir_interface_gestual_slides(self):
        if not self.pasta_slides_var.get():
            messagebox.showwarning("Aviso", "Selecione uma pasta com imagens para usar como slides.")
            return

        try:
            self.status_var.set("Slides por gestos iniciados. Use q para sair e r para recalibrar.")
            self.root.update_idletasks()
            log_path = interface_gestual_slides(self.pasta_slides_var.get(), status_callback=self.status_var.set)
            self.adicionar_texto("=" * 80)
            self.adicionar_texto("SLIDES POR GESTOS FINALIZADO")
            self.adicionar_texto(f"Pasta de slides: {self.pasta_slides_var.get()}")
            self.adicionar_texto(f"Log salvo em: {log_path}\n")
            messagebox.showinfo("Finalizado", f"Slides por gestos finalizados.\nLog salvo em:\n{log_path}")
        except Exception as e:
            messagebox.showerror("Erro nos slides por gestos", str(e))
            self.status_var.set("Erro nos slides por gestos.")

def main():
    pyautogui.FAILSAFE = False
    garantir_pastas_saida()

    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()