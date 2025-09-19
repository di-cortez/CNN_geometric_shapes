# utils.py (VERSÃO CORRIGIDA E LIMPA)

import os
import glob
import numpy as np
import torch.nn as nn
from PIL import Image, ImageTk

# --- Funções de Ajuda (Sistema de Arquivos e Modelo) ---

def find_datasets(base_path="."):
    """Varre o diretório atual em busca de pastas de dataset."""
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.')]

def find_models_in_dataset(dataset_path):
    """Encontra todos os arquivos .pth dentro de uma pasta de dataset."""
    return glob.glob(os.path.join(dataset_path, "*.pth"))

def load_class_map(dataset_path, filename="shape_ids.txt"):
    """
    Carrega o mapeamento de ID para nome. Por padrão, procura por 'shape_ids.txt'.
    ESTA É A CORREÇÃO PRINCIPAL.
    """
    class_map = {}
    path = os.path.join(dataset_path, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    class_map[int(parts[0])] = parts[1].strip()
    return class_map

def get_model_layers(model):
    """Inspeciona um modelo e retorna os nomes das camadas CNN e FC."""
    layers = {'CNN': [], 'FC': []}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.BatchNorm2d)):
            if name: layers['CNN'].append(f"{name} ({type(module).__name__})")
        elif isinstance(module, (nn.Linear, nn.Dropout)):
            if name: layers['FC'].append(f"{name} ({type(module).__name__})")
    return layers

def format_weight(w):
    """Formata um peso para exibição consistente."""
    return f"{w:>7.4f}"

# --- Funções de Visualização (Desenho nos Canvas) ---

def visualize_feature_maps(canvas, feature_maps, tk_images_list):
    """
    Desenha os feature maps de camadas CNN em um tamanho fixo e adiciona o 
    índice do filtro sobre cada miniatura.
    Esta função NÃO usa a variável 'original_shape'.
    """
    canvas.delete("all")
    tk_images_list.clear()
    n_maps = feature_maps.shape[0]
    if n_maps == 0: return

    img_size, pad, cols = 64, 5, 4
    canvas.viz_info = {'maps': feature_maps, 'cols': cols, 'img_w': img_size, 'img_h': img_size, 'pad': pad}

    for i, fm in enumerate(feature_maps):
        row, col = divmod(i, cols)
        x1, y1 = col * (img_size + pad) + pad, row * (img_size + pad) + pad
        
        fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        fm_img_data = (fm_norm * 255).byte().cpu().numpy()
        pil_img = Image.fromarray(fm_img_data).resize((img_size, img_size), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(pil_img)
        tk_images_list.append(tk_img)
        
        # 1. Desenha a imagem da ativação
        canvas.create_image(x1, y1, image=tk_img, anchor='nw')

        # 2. Prepara e desenha o texto do índice sobre a imagem
        text_label = f"{i + 1}/{n_maps}"
        text_x = x1 + 3
        text_y = y1 + 3

        # Desenha uma "sombra" preta para o texto
        canvas.create_text(text_x + 1, text_y + 1, text=text_label, font=("Arial", 8, "bold"), fill="black", anchor='nw')
        # Desenha o texto principal em branco por cima da sombra
        canvas.create_text(text_x, text_y, text=text_label, font=("Arial", 8, "bold"), fill="white", anchor='nw')


def visualize_vector(canvas, vector, original_shape):
    """
    Visualiza um vetor de ativação como uma grade de retângulos coloridos (heatmap).
    Azul representa baixa ativação, Vermelho representa alta ativação.
    Também exibe as dimensões originais do tensor.
    """
    canvas.delete("all")
    if vector is None: return
    canvas.update_idletasks()

    canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
    if canvas_w < 10 or canvas_h < 10: return

    # Exibe as dimensões do vetor no topo do canvas
    shape_text = f"Shape: {tuple(original_shape)}"
    canvas.create_text(5, 5, text=shape_text, anchor='nw', fill='white', font=("Arial", 9))
    
    # Prepara o vetor para visualização em grade
    vector = vector.flatten()
    v_min, v_max = vector.min(), vector.max()
    
    # Normaliza o vetor para o intervalo [0, 1] para mapear para cores
    if v_max == v_min:
        norm_vector = np.zeros_like(vector)
    else:
        norm_vector = (vector - v_min) / (v_max - v_min)

    # Define um número de linhas para a grade e calcula as colunas
    rows = 32
    if len(vector) == 0: return
    cols = int(np.ceil(len(vector) / rows))
    
    cell_w, cell_h = canvas_w / cols, (canvas_h - 25) / rows # Deixa espaço para o texto
    
    for i, val in enumerate(norm_vector):
        row, col = divmod(i, cols)
        
        # Mapeia o valor normalizado para uma cor (gradiente de azul para vermelho)
        r = int(val * 255)
        b = int((1 - val) * 255)
        color = f'#{r:02x}00{b:02x}'
        
        # Calcula a posição da célula, considerando o espaço para o texto
        x0, y0 = col * cell_w, row * cell_h + 25
        x1, y1 = x0 + cell_w, y0 + cell_h
        canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")


def draw_pixel_grid(canvas, original_img_size, canvas_size, offset):
    """Desenha um grid de pixels sobre a imagem de entrada."""
    canvas.delete("pixel_grid")
    offset_x, offset_y = offset
    canvas_w, canvas_h = canvas_size
    step_x = canvas_w / original_img_size[0]
    step_y = canvas_h / original_img_size[1]
    for i in range(1, original_img_size[0]):
        x = offset_x + i * step_x
        canvas.create_line(x, offset_y, x, offset_y + canvas_h, fill='gray50', tags="pixel_grid", width=0.5)
    for i in range(1, original_img_size[1]):
        y = offset_y + i * step_y
        canvas.create_line(offset_x, y, offset_x + canvas_w, y, fill='gray50', tags="pixel_grid", width=0.5)

def draw_detail_pixel_grid(canvas, original_shape, resized_size, offset):
    """Desenha um grid de pixels sobre a imagem de detalhe da ativação."""
    canvas.delete("detail_grid")
    offset_x, offset_y = offset
    resized_w, resized_h = resized_size
    original_h, original_w = original_shape
    if original_w == 0 or original_h == 0: return
    step_x, step_y = resized_w / original_w, resized_h / original_h
    for i in range(1, original_w):
        x = offset_x + i * step_x
        canvas.create_line(x, offset_y, x, offset_y + resized_h, fill='gray50', tags="detail_grid", width=0.5)
    for i in range(1, original_h):
        y = offset_y + i * step_y
        canvas.create_line(offset_x, y, offset_x + resized_w, y, fill='gray50', tags="detail_grid", width=0.5)