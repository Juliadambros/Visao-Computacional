import tkinter as tk
from tkinter import ttk

from main import (
    pre_processar,
    dvr,
    gerar_isosuperficie,
    mostrar_isosuperficie,
    gerar_esqueleto,
    mostrar_esqueleto_e_iso,
    calcular_metricas,
    janela_dividida
)


class Aplicacao:

    def __init__(self, root):

        self.root = root
        self.root.title("Análise 3D de Sistemas Radiculares")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1e1e1e")

        self.PASTA1 = "b0207/b0207"
        self.PASTA2 = "b0309/b0309"

        self.vol1 = None
        self.vol2 = None

        self.mesh1 = None
        self.mesh2 = None

        self.skel1 = None
        self.skel2 = None

        self.configurar_estilo()
        self.criar_interface()

        self.status.config(
            text="🔄 Carregando volumes...",
            fg="orange"
        )

        self.root.update()

        self.vol1 = pre_processar(self.PASTA1)
        self.vol2 = pre_processar(self.PASTA2)

        self.status.config(
            text="🟢 Volumes carregados",
            fg="lightgreen"
        )

    def configurar_estilo(self):

        style = ttk.Style()

        style.theme_use("clam")

        style.configure(
            "TButton",
            font=("Segoe UI", 11, "bold"),
            padding=10
        )

        style.configure(
            "Titulo.TLabel",
            background="#1e1e1e",
            foreground="white",
            font=("Segoe UI", 20, "bold")
        )

        style.configure(
            "Subtitulo.TLabel",
            background="#1e1e1e",
            foreground="#cccccc",
            font=("Segoe UI", 10)
        )

    def criar_interface(self):

        topo = tk.Frame(
            self.root,
            bg="#1e1e1e"
        )

        topo.pack(fill="x", pady=15)

        ttk.Label(
            topo,
            text="🌱 Análise 3D de Sistemas Radiculares",
            style="Titulo.TLabel"
        ).pack()

        ttk.Label(
            topo,
            text="Trabalho Prático 3 - Visão Computacional",
            style="Subtitulo.TLabel"
        ).pack()

        self.status = tk.Label(
            self.root,
            text="",
            bg="#1e1e1e",
            fg="white",
            font=("Segoe UI", 10, "bold")
        )

        self.status.pack(pady=5)

        principal = tk.Frame(
            self.root,
            bg="#1e1e1e"
        )

        principal.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

        visual_frame = tk.LabelFrame(
            principal,
            text=" Visualizações ",
            bg="#2d2d2d",
            fg="white",
            font=("Segoe UI", 12, "bold")
        )

        visual_frame.pack(
            side="left",
            fill="y",
            padx=10
        )

        ttk.Button(
            visual_frame,
            text="🦷 DVR",
            command=self.mostrar_dvr
        ).pack(fill="x", padx=15, pady=8)

        ttk.Button(
            visual_frame,
            text="🧩 Isosuperfície",
            command=self.mostrar_iso
        ).pack(fill="x", padx=15, pady=8)

        ttk.Button(
            visual_frame,
            text="🌱 Esqueleto + Isosuperfície",
            command=self.mostrar_esqueleto
        ).pack(fill="x", padx=15, pady=8)

        ttk.Button(
            visual_frame,
            text="🔄 Comparação Sincronizada",
            command=self.mostrar_dividida
        ).pack(fill="x", padx=15, pady=8)

        ttk.Button(
            visual_frame,
            text="📊 Calcular Métricas",
            command=self.mostrar_metricas
        ).pack(fill="x", padx=15, pady=8)

        colunas = (
            "Métrica",
            "b0207",
            "b0309"
        )

        self.tabela = ttk.Treeview(
            principal,
            columns=colunas,
            show="headings",
            height=20
        )

        for coluna in colunas:

            self.tabela.heading(
                coluna,
                text=coluna
            )

            self.tabela.column(
                coluna,
                width=220,
                anchor="center"
            )

        self.tabela.pack(
            side="right",
            fill="both",
            expand=True,
            padx=10
        )

    def mostrar_dvr(self):

        dvr(self.vol1, "DVR - b0207")
        dvr(self.vol2, "DVR - b0309")

    def mostrar_iso(self):

        self.mesh1 = gerar_isosuperficie(self.vol1)
        self.mesh2 = gerar_isosuperficie(self.vol2)

        mostrar_isosuperficie(
            self.mesh1,
            "b0207"
        )

        mostrar_isosuperficie(
            self.mesh2,
            "b0309"
        )

    def mostrar_esqueleto(self):

        if self.mesh1 is None:

            self.mesh1 = gerar_isosuperficie(self.vol1)
            self.mesh2 = gerar_isosuperficie(self.vol2)

        self.skel1 = gerar_esqueleto(self.vol1)
        self.skel2 = gerar_esqueleto(self.vol2)

        mostrar_esqueleto_e_iso(
            self.mesh1,
            self.skel1,
            "b0207"
        )

        mostrar_esqueleto_e_iso(
            self.mesh2,
            self.skel2,
            "b0309"
        )

    def mostrar_metricas(self):

        for item in self.tabela.get_children():
            self.tabela.delete(item)

        if self.mesh1 is None:
            self.mesh1 = gerar_isosuperficie(self.vol1)
            self.mesh2 = gerar_isosuperficie(self.vol2)

        if self.skel1 is None:
            self.skel1 = gerar_esqueleto(self.vol1)
            self.skel2 = gerar_esqueleto(self.vol2)

        m1 = calcular_metricas(
            self.mesh1,
            self.vol1,
            self.skel1
        )

        m2 = calcular_metricas(
            self.mesh2,
            self.vol2,
            self.skel2
        )

        for chave in m1.keys():

            self.tabela.insert(
                "",
                "end",
                values=(
                    chave,
                    round(float(m1[chave]), 4),
                    round(float(m2[chave]), 4)
                )
            )

    def mostrar_dividida(self):

        if self.mesh1 is None:

            self.mesh1 = gerar_isosuperficie(self.vol1)
            self.mesh2 = gerar_isosuperficie(self.vol2)

        janela_dividida(
            self.vol1,
            self.mesh1,
            self.vol2,
            self.mesh2,
            ("b0207", "b0309")
        )


if __name__ == "__main__":

    root = tk.Tk()

    app = Aplicacao(root)

    root.mainloop()