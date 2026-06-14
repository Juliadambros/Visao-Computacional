import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

# ==========================================================
# INTERFACE GRÁFICA DO TRABALHO 2 - VISÃO COMPUTACIONAL
# Permite escolher qual parte do trabalho será executada.
# ==========================================================

PASTA_BASE = os.path.dirname(os.path.abspath(__file__))

PROGRAMAS = {
    "Metrologia com ArUco": {
        "arquivo": "aruco1.py",
        "descricao": "Mede distância entre marcadores ArUco."
    },
    "Objeto 3D na mão": {
        "arquivo": "mao.py",
        "descricao": "Detecta a mão e posiciona o objeto 3D virtual."
    },
    "Ocarina virtual": {
        "arquivo": "ocarina.py",
        "descricao": "Usa ArUco e modelo .OBJ para tocar a ocarina virtual."
    }
}

processo_atual = None
programa_atual = None


def caminho_script(nome_arquivo):
    return os.path.join(PASTA_BASE, nome_arquivo)


def atualizar_status(texto, cor="#333333"):
    lbl_status.config(text=texto, fg=cor)


def iniciar_programa(nome_programa):
    global processo_atual, programa_atual

    if processo_atual is not None and processo_atual.poll() is None:
        messagebox.showwarning(
            "Programa em execução",
            "Já existe uma parte do trabalho em execução.\n\nFeche ou pare o programa atual antes de abrir outro."
        )
        return

    info = PROGRAMAS[nome_programa]
    arquivo = info["arquivo"]
    caminho = caminho_script(arquivo)

    if not os.path.exists(caminho):
        messagebox.showerror(
            "Arquivo não encontrado",
            f"Não encontrei o arquivo:\n{arquivo}\n\nVerifique se ele está na mesma pasta da interface."
        )
        return

    try:
        # Usa o mesmo Python que abriu a interface.
        processo_atual = subprocess.Popen(
            [sys.executable, caminho],
            cwd=PASTA_BASE
        )
        programa_atual = nome_programa
        atualizar_status(f"Executando: {nome_programa}", "#1b7f2a")
        btn_parar.config(state="normal")

    except Exception as erro:
        messagebox.showerror(
            "Erro ao iniciar",
            f"Não foi possível iniciar {nome_programa}.\n\nErro: {erro}"
        )
        atualizar_status("Erro ao iniciar programa", "#a00000")


def parar_programa():
    global processo_atual, programa_atual

    if processo_atual is None or processo_atual.poll() is not None:
        atualizar_status("Nenhum programa em execução", "#333333")
        btn_parar.config(state="disabled")
        processo_atual = None
        programa_atual = None
        return

    try:
        processo_atual.terminate()
        processo_atual = None
        programa_atual = None
        atualizar_status("Programa encerrado", "#a06000")
        btn_parar.config(state="disabled")

    except Exception as erro:
        messagebox.showerror(
            "Erro ao parar",
            f"Não foi possível encerrar o programa.\n\nErro: {erro}"
        )


def verificar_processo():
    global processo_atual, programa_atual

    if processo_atual is not None:
        if processo_atual.poll() is not None:
            processo_atual = None
            programa_atual = None
            atualizar_status("Nenhum programa em execução", "#333333")
            btn_parar.config(state="disabled")

    janela.after(800, verificar_processo)


def ao_fechar():
    global processo_atual

    if processo_atual is not None and processo_atual.poll() is None:
        resposta = messagebox.askyesno(
            "Sair",
            "Existe um programa em execução. Deseja encerrá-lo e fechar a interface?"
        )

        if not resposta:
            return

        try:
            processo_atual.terminate()
        except Exception:
            pass

    janela.destroy()


# ================= INTERFACE =================

janela = tk.Tk()
janela.title("Trabalho 2 - Visão Computacional")
janela.geometry("620x520")
janela.resizable(False, False)
janela.configure(bg="#f2f4f8")

janela.protocol("WM_DELETE_WINDOW", ao_fechar)

fonte_titulo = ("Arial", 18, "bold")
fonte_subtitulo = ("Arial", 11)
fonte_botao = ("Arial", 12, "bold")
fonte_texto = ("Arial", 10)

frame_principal = tk.Frame(janela, bg="#f2f4f8", padx=25, pady=20)
frame_principal.pack(fill="both", expand=True)

lbl_titulo = tk.Label(
    frame_principal,
    text="Trabalho 2 - Visão Computacional",
    font=fonte_titulo,
    bg="#f2f4f8",
    fg="#1f2937"
)
lbl_titulo.pack(pady=(5, 5))

lbl_subtitulo = tk.Label(
    frame_principal,
    text="Selecione qual demonstração deseja executar.",
    font=fonte_subtitulo,
    bg="#f2f4f8",
    fg="#4b5563"
)
lbl_subtitulo.pack(pady=(0, 20))

frame_cards = tk.Frame(frame_principal, bg="#f2f4f8")
frame_cards.pack(fill="x")


def criar_card(nome_programa):
    info = PROGRAMAS[nome_programa]

    card = tk.Frame(
        frame_cards,
        bg="#ffffff",
        highlightbackground="#d1d5db",
        highlightthickness=1,
        padx=15,
        pady=12
    )
    card.pack(fill="x", pady=8)

    lbl_nome = tk.Label(
        card,
        text=nome_programa,
        font=("Arial", 13, "bold"),
        bg="#ffffff",
        fg="#111827",
        anchor="w"
    )
    lbl_nome.pack(fill="x")

    lbl_desc = tk.Label(
        card,
        text=info["descricao"],
        font=fonte_texto,
        bg="#ffffff",
        fg="#4b5563",
        anchor="w"
    )
    lbl_desc.pack(fill="x", pady=(3, 8))

    btn = tk.Button(
        card,
        text="Iniciar",
        font=fonte_botao,
        bg="#2563eb",
        fg="#ffffff",
        activebackground="#1d4ed8",
        activeforeground="#ffffff",
        relief="flat",
        padx=15,
        pady=6,
        command=lambda: iniciar_programa(nome_programa)
    )
    btn.pack(anchor="e")


for nome in PROGRAMAS.keys():
    criar_card(nome)

frame_controle = tk.Frame(frame_principal, bg="#f2f4f8")
frame_controle.pack(fill="x", pady=(20, 10))

btn_parar = tk.Button(
    frame_controle,
    text="Parar programa atual",
    font=fonte_botao,
    bg="#dc2626",
    fg="#ffffff",
    activebackground="#991b1b",
    activeforeground="#ffffff",
    relief="flat",
    padx=15,
    pady=8,
    state="disabled",
    command=parar_programa
)
btn_parar.pack()

lbl_status = tk.Label(
    frame_principal,
    text="Nenhum programa em execução",
    font=("Arial", 11, "bold"),
    bg="#f2f4f8",
    fg="#333333"
)
lbl_status.pack(pady=(8, 8))

lbl_instrucao = tk.Label(
    frame_principal,
    text="Observação: para fechar uma demonstração, pressione ESC na janela da câmera ou use o botão acima.",
    font=("Arial", 9),
    bg="#f2f4f8",
    fg="#6b7280",
    wraplength=560,
    justify="center"
)
lbl_instrucao.pack(pady=(5, 0))

verificar_processo()
janela.mainloop()
