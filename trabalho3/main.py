import os
import numpy as np
import imageio.v3 as iio
import pyvista as pv
from skimage import measure, morphology, filters
from skimage.morphology import skeletonize
from scipy import ndimage
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText



def carregar_pasta(caminho: str) -> np.ndarray:
    """Carrega sequência de imagens de uma pasta e retorna volume 3D (Z,Y,X)."""
    extensoes = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    arquivos = sorted(
        [
            os.path.join(caminho, f)
            for f in os.listdir(caminho)
            if f.lower().endswith(extensoes)
        ]
    )
    if not arquivos:
        raise ValueError(f"Nenhuma imagem encontrada em: {caminho}")

    fatias = []
    for arq in arquivos:
        img = iio.imread(arq)
        if img.ndim == 3:
            img = img.mean(axis=2)  # RGB -> cinza
        fatias.append(img.astype(np.float32))

    volume = np.stack(fatias, axis=0)  # (Z, Y, X)
    print(f"  Carregado: {len(fatias)} fatias, shape={volume.shape}")
    return volume


def limpar_bordas(
    volume: np.ndarray, margem_z: int = 15, margem_xy: int = 50
) -> np.ndarray:
    """Zera margem nas 6 faces do volume. XY precisam de margem maior por artefatos de scanner."""
    vol = volume.copy()
    vol[:margem_z] = 0
    vol[-margem_z:] = 0
    vol[:, :margem_xy] = 0
    vol[:, -margem_xy:] = 0
    vol[:, :, :margem_xy] = 0
    vol[:, :, -margem_xy:] = 0
    return vol


def normalizar(volume: np.ndarray) -> np.ndarray:
    mn, mx = volume.min(), volume.max()
    normalizado = ((volume - mn) / (mx - mn) * 255).astype(np.float32)
    return 255.0 - normalizado  # inverte: raiz escura vira brilhante, fundo vira preto


def pre_processar(caminho: str, margem_z: int = 15, margem_xy: int = 50) -> np.ndarray:
    print(f"Carregando '{caminho}'...")
    vol = carregar_pasta(caminho)
    vol = limpar_bordas(vol, margem_z, margem_xy)
    vol = normalizar(vol)
    vol = limpar_bordas(vol, margem_z, margem_xy)
    print(f"  Pré-processado: min={vol.min():.1f}, max={vol.max():.1f}")
    return vol


def para_pyvista(volume: np.ndarray) -> pv.ImageData:
    """
    Converte array (Z,Y,X) para ImageData com point_data.
    point_data (e não cell_data) é necessário para o add_volume funcionar
    corretamente e permitir transparência real no DVR.
    """
    nz, ny, nx = volume.shape
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)  # dimensões em pontos
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (0.0, 0.0, 0.0)
    # Ordem de flatten: o PyVista espera Fortran-order para (X,Y,Z)
    grid.point_data["values"] = volume.transpose(2, 1, 0).flatten(order="F")
    return grid


def dvr(volume: np.ndarray, titulo: str = "DVR"):
    """
    DVR correto:
    - Usa point_data (não cell_data)
    - Função de opacidade sigmoidal: fundo (escuro) = transparente,
      raiz (brilhante) = opaco
    - cmap 'bone' para aparência radiográfica
    """
    grid = para_pyvista(volume)

    # Opacidade: os primeiros valores (fundo escuro) são 0,
    # aumenta progressivamente para os voxels claros (raiz)
    opacity = [0.0, 0.0, 0.0, 0.02, 0.05, 0.2, 0.5, 0.8, 1.0]

    pl = pv.Plotter(title=titulo)
    pl.background_color = "black"
    pl.add_volume(
        grid,
        scalars="values",
        cmap="bone",
        opacity=opacity,
        opacity_unit_distance=1.0,
        shade=True,
        ambient=0.3,
        diffuse=0.6,
        specular=0.2,
    )
    pl.add_text(titulo, font_size=12, color="white")
    pl.show()



def gerar_isosuperficie(volume: np.ndarray, percentil_iso: int = 30, n_suavizacao: int = 20, fator_decimacao: float = 0.5) -> pv.PolyData:
    """
    Marching Cubes -> mantém só maior componente -> suavização -> decimação.
    Usa threshold de Otsu sobre os voxels não-nulos para separar
    raiz do fundo de forma automática e robusta.
    """
    from skimage.filters import threshold_otsu
    voxels_nao_nulos = volume[volume > 0]
    iso_val = float(threshold_otsu(voxels_nao_nulos))
    print(f"  iso_value = {iso_val:.2f}  (Otsu)")

    verts, faces, normals, _ = measure.marching_cubes(
        volume, level=iso_val, allow_degenerate=False
    )
    # Marching cubes retorna (Z,Y,X) — converter para (X,Y,Z)
    verts_xyz = verts[:, [2, 1, 0]]

    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()
    mesh = pv.PolyData(verts_xyz, faces_pv)

    # Remove fragmentos soltos — mantém só o maior componente conectado
    bodies = mesh.split_bodies()
    if len(bodies) > 1:
        maior = max(bodies, key=lambda b: b.n_cells)
        mesh = maior.extract_surface()
        print(f"  Fragmentos removidos: {len(bodies)-1} componentes descartados")

    if n_suavizacao > 0:
        mesh = mesh.smooth(n_iter=n_suavizacao, relaxation_factor=0.1)

    if 0 < fator_decimacao < 1:
        mesh = mesh.decimate(fator_decimacao)

    print(f"  Mesh: {mesh.n_points} vértices, {mesh.n_cells} faces")
    return mesh


def mostrar_isosuperficie(mesh: pv.PolyData, titulo: str = "Isosuperfície"):
    pl = pv.Plotter(title=titulo)
    pl.background_color = "#1a1a2e"
    pl.add_mesh(
        mesh, color="#c8a96e", smooth_shading=True, specular=0.5, specular_power=15
    )
    pl.add_text(titulo, font_size=12, color="white")
    pl.show()


def gerar_esqueleto(volume: np.ndarray, percentil_iso: int = 30) -> np.ndarray:
    from skimage.filters import threshold_otsu

    iso_val = float(threshold_otsu(volume[volume > 0]))
    print(f"  Esqueleto threshold = {iso_val:.2f} (Otsu)")
    binario = volume > iso_val
    binario = ndimage.binary_fill_holes(binario)
    binario = morphology.remove_small_objects(binario, min_size=64)
    skel = skeletonize(binario)
    print(f"  Esqueleto: {int(skel.sum())} pontos")
    return skel.astype(np.uint8)


def mostrar_esqueleto_e_iso(
    mesh: pv.PolyData, skel: np.ndarray, titulo: str = "Isosuperfície + Esqueleto"
):
    coords = np.argwhere(skel > 0).astype(float)
    # Converter (Z,Y,X) -> (X,Y,Z)
    coords_xyz = coords[:, [2, 1, 0]]

    pl = pv.Plotter(title=titulo)
    pl.background_color = "#0d0d0d"
    pl.add_mesh(mesh, color="#c8a96e", smooth_shading=True, opacity=0.4, specular=0.3)
    if len(coords_xyz) > 0:
        skel_cloud = pv.PolyData(coords_xyz)
        pl.add_points(
            skel_cloud, color="dodgerblue", point_size=2, render_points_as_spheres=True
        )
    pl.add_text(titulo, font_size=12, color="white")
    pl.show()


def calcular_metricas(
    mesh: pv.PolyData, volume: np.ndarray, skel: np.ndarray, percentil_iso: int = 30
) -> dict:
    from skimage.filters import threshold_otsu

    iso_val = float(threshold_otsu(volume[volume > 0]))
    binario = (volume > iso_val).astype(np.uint8)

    # Volume e área
    vol_voxel = float(binario.sum())
    area = float(mesh.area)

    # Compacidade: 36π·V² / A³  (= 1 para esfera perfeita)
    compacidade = (36 * np.pi * vol_voxel**2) / (area**3) if area > 0 else 0.0

    # Excentricidade via PCA dos voxels
    coords = np.argwhere(binario).astype(float)
    if len(coords) > 3:
        cov = np.cov(coords.T)
        eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
        excentricidade = float(np.sqrt(1 - eigs[-1] / eigs[0])) if eigs[0] > 0 else 0.0
    else:
        excentricidade = 0.0

    # Métricas de esqueleto
    n_pontos = int(skel.sum())
    comprimento_total = float(n_pontos)  # em voxels

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    vizinhos = ndimage.convolve(skel, kernel, mode="constant", cval=0)
    graus = vizinhos[skel > 0]

    grau_medio = float(graus.mean()) if len(graus) > 0 else 0.0
    n_bifurcacoes = int((graus >= 3).sum())
    n_pontas = int((graus == 1).sum())
    grau_max = int(graus.max()) if len(graus) > 0 else 0
    n_ramos = (n_bifurcacoes + n_pontas) // 2 if (n_bifurcacoes + n_pontas) > 0 else 0
    densidade = comprimento_total / vol_voxel if vol_voxel > 0 else 0.0
    # Tortuosidade estimada: comprimento_esqueleto / distância_euclidiana_extremos
    if n_pontas >= 2:
        pontas_coords = np.argwhere((skel > 0) & (vizinhos == 1)).astype(float)
        if len(pontas_coords) >= 2:
            dist_max = np.linalg.norm(pontas_coords[0] - pontas_coords[-1])
            tortuosidade = comprimento_total / dist_max if dist_max > 0 else 1.0
        else:
            tortuosidade = 1.0
    else:
        tortuosidade = 1.0

    return {
        "volume_voxel": vol_voxel,
        "area_superficie": area,
        "compacidade": compacidade,
        "excentricidade": excentricidade,
        "skel_n_pontos": n_pontos,
        "skel_comprimento": comprimento_total,
        "skel_grau_medio": grau_medio,
        "skel_n_bifurcacoes": n_bifurcacoes,
        "skel_n_pontas": n_pontas,
        "skel_grau_max": grau_max,
        "skel_n_ramos": n_ramos,
        "skel_densidade": densidade,
        "skel_tortuosidade": tortuosidade,
    }


def imprimir_metricas(nome: str, m: dict):
    print(f"""
{'='*50}
  MÉTRICAS: {nome}
{'='*50}
  Volume (voxels)        : {m['volume_voxel']:.0f}
  Área de Superfície     : {m['area_superficie']:.2f}
  Compacidade            : {m['compacidade']:.8f}
  Excentricidade         : {m['excentricidade']:.4f}
{'─'*50}
  ESQUELETO
{'─'*50}
  Nº de Pontos           : {m['skel_n_pontos']}
  Comprimento Total      : {m['skel_comprimento']:.1f}
  Grau Médio             : {m['skel_grau_medio']:.2f}
  Nº de Bifurcações      : {m['skel_n_bifurcacoes']}
  Nº de Pontas           : {m['skel_n_pontas']}
  Grau Máximo            : {m['skel_grau_max']}
  Nº de Ramos (est.)     : {m['skel_n_ramos']}
  Densidade do Esqueleto : {m['skel_densidade']:.6f}
  Tortuosidade           : {m['skel_tortuosidade']:.4f}
{'='*50}""")


def janela_dividida(
    vol1: np.ndarray,
    mesh1: pv.PolyData,
    vol2: np.ndarray,
    mesh2: pv.PolyData,
    nomes=("b0207", "b0309"),
):
    """
    Janela 2x2:
      [0,0] DVR raiz 1       [0,1] DVR raiz 2
      [1,0] Isosup. raiz 1   [1,1] Isosup. raiz 2
    Câmeras vinculadas com pl.link_views().
    """
    opacity = [0.0, 0.0, 0.0, 0.02, 0.05, 0.2, 0.5, 0.8, 1.0]

    pl = pv.Plotter(shape=(2, 2), title="Visualização Dividida e Sincronizada")
    pl.background_color = "black"

    pl.subplot(0, 0)
    pl.add_volume(para_pyvista(vol1), scalars="values", cmap="bone", opacity=opacity)
    pl.add_text(f"DVR – {nomes[0]}", font_size=10, color="white")

    pl.subplot(0, 1)
    pl.add_volume(para_pyvista(vol2), scalars="values", cmap="bone", opacity=opacity)
    pl.add_text(f"DVR – {nomes[1]}", font_size=10, color="white")

    pl.subplot(1, 0)
    pl.add_mesh(mesh1, color="#c8a96e", smooth_shading=True, specular=0.5)
    pl.add_text(f"Isosuperfície – {nomes[0]}", font_size=10, color="white")

    pl.subplot(1, 1)
    pl.add_mesh(mesh2, color="#c8a96e", smooth_shading=True, specular=0.5)
    pl.add_text(f"Isosuperfície – {nomes[1]}", font_size=10, color="white")

    pl.link_views()  # sincroniza câmeras de todas as subjanelas
    pl.show()

 


