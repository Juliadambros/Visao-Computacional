import numpy as np
import pyvista as pv
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree


def skel(volume):
    ### 1. Geração da Isosuperfície e do Esqueleto 3D

    # Suavização Gaussiana do volume para reduzir ruído
    volume = gaussian_filter(volume, sigma=3)

    # Convertendo o array NumPy para a estrutura PyVista
    grade = pv.ImageData()
    grade.dimensions = volume.shape
    grade.spacing = (1.0, 1.0, 1.0)
    grade.origin = (0.0, 0.0, 0.0)

    # O VTK exige que a memória seja achatada no padrão Fortran
    grade.point_data["intensidades"] = volume.flatten(order="F")

    # Extração da Isosuperfície
    malha_externa = grade.contour(isosurfaces=[50])

    # Extração do Esqueleto 3D
    volume = volume > 50 # Threshold
    esqueleto = skeletonize(volume)

    # Extrai as coordenadas (X, Y, Z) dos voxels que formam o esqueleto
    coordenadas_esqueleto = np.argwhere(esqueleto).astype(np.float32)

    ### 2. Organização das coordenadas do esqueleto para
    ### interligação sequencial na renderização

    # Indexação espacial da nuvem de pontos com KDTree
    arvore_espacial = KDTree(coordenadas_esqueleto)

    # query_pairs varre a árvore e devolve os pares de pontos
    # que estão a uma distância máxima 'r' uns dos outros.
    # Foi usado r=1.8 para pegar os vizinhos de face (d=1),
    # de aresta (d=1.41) e de vértice/diagonal (d=1.73).
    pares_vizinhos = arvore_espacial.query_pairs(r=1.8)

    # O PyVista exige o formato de array 1D:
    # [NumPontosNaLinha, PontoA, PontoB, NumPontosNaLinha...]
    linhas_pyvista = []
    for indice_a, indice_b in pares_vizinhos:
        linhas_pyvista.extend([2, indice_a, indice_b])

    ### 3. Renderização Geométrica (PyVista)

    # Cria a instância da malha apenas com as coordenadas
    grafo_3d = pv.PolyData(coordenadas_esqueleto)

    # Injeta as arestas formatadas
    grafo_3d.lines = np.array(linhas_pyvista)

    # Prepara a janela de visualização
    p = pv.Plotter(window_size=[1000, 800])

    # Adiciona a isosuperfície (transparente)
    p.add_mesh(malha_externa, color="tan", opacity=0.3, show_edges=False)

    # Desenha as arestas engrossando as linhas com o filtro 'tube'
    tubos = grafo_3d.tube(radius=1)
    p.add_mesh(tubos, color="red",smooth_shading=True)

    p.add_axes()
    p.show()


# ==========================================
# Execução
# ==========================================
if __name__ == "__main__":
    volume = np.load("vasos_sanguineos.npy")
    skel(volume)