import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import zipfile
import io

import app_state as state
from model import SimpleCNN
from utils import (load_class_map, find_models_in_dataset, get_model_layers, 
                   visualize_feature_maps, visualize_vector, draw_pixel_grid, 
                   draw_detail_pixel_grid, format_weight)

# --- Funções de Callback (Lógica da Aplicação) ---

def on_dataset_selected(app, event=None):
    dataset_path = app.dataset_selector.get()
    if not dataset_path: return
    app.model_selector.set(''); app.layer_selector.set('')
    try:
        models_found = find_models_in_dataset(dataset_path)
        app.model_selector['values'] = models_found
        if models_found:
            app.model_selector.set(models_found[0])
            load_model_and_data(app)
        else:
            app.status_var.set(f"Dataset selected, but no model (.pth) found in '{dataset_path}'.")
            enable_controls(app, False)
    except Exception as e:
        messagebox.showerror("Error", f"Could not process dataset directory: {e}")

def load_model_and_data(app):
    pth_path = app.model_selector.get(); dataset_path = app.dataset_selector.get()
    if not pth_path or not dataset_path: 
        enable_controls(app, False)
        return
    try:
        images_zip_path = os.path.join(dataset_path, "images.zip")
        labels_zip_path = os.path.join(dataset_path, "labels.zip")
        state.images_zip = zipfile.ZipFile(images_zip_path, 'r')
        state.labels_zip = zipfile.ZipFile(labels_zip_path, 'r')
        state.file_names = sorted([name for name in state.images_zip.namelist() if name.endswith(".png")])
        state.class_map = load_class_map(dataset_path)
        with state.images_zip.open(state.file_names[0]) as f: 
            img_size = Image.open(f).size[0]
        num_classes = len(state.class_map)
        state.model = SimpleCNN(dropout=0.4, img_size=img_size, num_classes=num_classes)
        state.model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
        state.model.eval()
        state.model_layers = get_model_layers(state.model)
        populate_layer_selector(app)
        app.status_var.set(f"Model '{os.path.basename(pth_path)}' loaded. Ready to explore.")
        enable_controls(app, True)
        update_all_visuals(app)
    except Exception as e: 
        messagebox.showerror("Loading Error", f"An error occurred: {e}")
        enable_controls(app, False)

def populate_layer_selector(app):
    layer_type = app.layer_type_var.get()
    layers = state.model_layers.get(layer_type, [])
    app.layer_selector['values'] = layers
    if layers: app.layer_selector.set(layers[0])
    else: app.layer_selector.set('')
    update_all_visuals(app)

def update_all_visuals(app):
    if not all([state.model, state.images_zip, state.file_names]): return
    update_input_panel(app)
    update_activation_panels(app)

def update_input_panel(app):
    try:
        idx = int(app.index_var.get())
        if not (0 <= idx < len(state.file_names)): 
            app.status_var.set("Invalid index.")
            return

        img_name = state.file_names[idx]
        img_bytes = state.images_zip.read(img_name)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        original_size = img.size
        
        app.image_canvas.update_idletasks()
        canvas_width = app.image_canvas.winfo_width()
        canvas_height = app.image_canvas.winfo_height()
        if canvas_width < 20 or canvas_height < 20: canvas_width, canvas_height = 280, 280
        new_size = min(canvas_width, canvas_height)
        canvas_size = (new_size, new_size)
        img_resized = img.resize(canvas_size, Image.NEAREST)
        
        tk_img = ImageTk.PhotoImage(img_resized)
        app.image_canvas.delete("all")
        x_offset = (canvas_width - new_size) // 2
        y_offset = (canvas_height - new_size) // 2
        app.image_canvas.create_image(x_offset, y_offset, image=tk_img, anchor='nw')
        app.image_canvas.image = tk_img 
        
        if app.show_grid_var.get():
            draw_pixel_grid(app.image_canvas, original_size, canvas_size, (x_offset, y_offset))

        input_tensor = transforms.ToTensor()(img).unsqueeze(0)
        with torch.no_grad(): outputs = state.model(input_tensor)
        _, pred_id = torch.max(outputs, 1); pred_id = pred_id.item()
        label_id = int(state.labels_zip.read(img_name.replace('.png', '.txt')).decode("utf-8").strip())
        
        app.label_var.set(f"True Label: {label_id} ({state.class_map.get(label_id, '?')})")
        app.pred_var.set(f"Prediction: {pred_id} ({state.class_map.get(pred_id, '?')})")
        app.status_var.set(f"Showing image {idx}/{len(state.file_names)-1}")
    except Exception as e:
        app.status_var.set(f"Error displaying image: {e}")

def update_activation_panels(app):
    update_activation_detail(app, None, -1); update_kernel_panel(app, None, -1)
    selected_layer_str = app.layer_selector.get()
    if not selected_layer_str or not state.model: 
        app.activation_grid_canvas.delete("all"); return
    layer_name = selected_layer_str.split(' ')[0]
    activation_tensor = state.model.activations.get(layer_name)
    if activation_tensor is None: 
        app.activation_grid_canvas.delete("all"); return
    
    state.tk_grid_images.clear()
    if 'conv' in layer_name or 'pool' in layer_name:
        visualize_feature_maps(app.activation_grid_canvas, activation_tensor.squeeze(0), state.tk_grid_images)
    elif 'fc' in layer_name or 'relu' in layer_name or 'dropout' in layer_name:
        # 1. Captura o shape original ANTES de achatar o tensor
        original_shape = activation_tensor.shape
        # 2. Prepara os dados do vetor
        vector_data = activation_tensor.squeeze().numpy()
        # 3. Chama a função de visualização, AGORA PASSANDO o shape como terceiro argumento
        visualize_vector(app.activation_grid_canvas, vector_data, original_shape)

def on_grid_click(app, event):
    canvas = event.widget
    if not hasattr(canvas, 'viz_info'): return
    info = canvas.viz_info; x, y = event.x, event.y
    col = x // (info['img_w'] + info['pad']); row = y // (info['img_h'] + info['pad'])
    filter_index = row * info['cols'] + col
    if 0 <= filter_index < info['maps'].shape[0]:
        update_activation_detail(app, info['maps'], filter_index)
        update_kernel_panel(app, app.layer_selector.get().split(' ')[0], filter_index)

def update_activation_detail(app, feature_maps, filter_index):
    state.last_detail_maps, state.last_detail_index = feature_maps, filter_index
    app.detail_canvas.delete("all")
    if feature_maps is None or filter_index < 0:
        app.detail_frame.config(text="Activation Detail")
        app.stats_var.set("Stats: (select a filter)"); return
    
    app.detail_frame.config(text=f"Activation Detail (Filter {filter_index + 1} of {feature_maps.shape[0]})")
    selected_map = feature_maps[filter_index]
    min_val, max_val, mean_val = selected_map.min().item(), selected_map.max().item(), selected_map.mean().item()
    app.stats_var.set(f"Stats: Min={min_val:.4f} | Max={max_val:.4f} | Mean={mean_val:.4f}")
    fm_norm = (selected_map - min_val) / (max_val - min_val + 1e-8)
    fm_img_data = (fm_norm * 255).byte().numpy()
    
    app.detail_canvas.update_idletasks()
    canvas_w = app.detail_canvas.winfo_width(); canvas_h = app.detail_canvas.winfo_height()
    size = max(16, min(canvas_w, canvas_h) - 10)
    if size < 16: return
    
    pil_img = Image.fromarray(fm_img_data).resize((size, size), Image.NEAREST)
    state.tk_detail_image = ImageTk.PhotoImage(pil_img)
    app.detail_canvas.create_image(5, 5, image=state.tk_detail_image, anchor='nw')
    
    if app.show_detail_grid_var.get():
        draw_detail_pixel_grid(app.detail_canvas, selected_map.shape, (size, size), (5, 5))

def update_kernel_panel(app, layer_name, filter_index):
    """Atualiza o Kernel Panel, criando dois visualizadores de canal para comparação."""
    
    # Limpa o painel de qualquer conteúdo antigo
    for widget in app.kernel_panel_frame.winfo_children(): 
        widget.destroy()

    # Se não for uma camada convolucional válida, mostra a mensagem padrão
    if not layer_name or 'conv' not in layer_name or filter_index < 0:
        ttk.Label(app.kernel_panel_frame, text="Select a CNN filter to see its kernel.").pack(pady=20)
        return

    # Pega a camada e os pesos do modelo
    layer = getattr(state.model, layer_name.split(' ')[0], None)
    if not layer: return
    
    weights = layer.weight.data[filter_index]
    bias = layer.bias.data[filter_index]
    in_channels = weights.shape[0]

    # Mostra informações gerais que se aplicam a ambos os viewers
    ttk.Label(app.kernel_panel_frame, text=f"Input Channels: {in_channels}", font=("", 10)).pack(pady=(5,0))
    ttk.Label(app.kernel_panel_frame, text=f"Bias: b = {format_weight(bias.item())}", font=("", 10, "bold")).pack(pady=(0,5))

    # 1. Cria o primeiro visualizador
    create_channel_viewer(app.kernel_panel_frame, "Viewer A", weights, in_channels)

    # 2. Adiciona um separador visual
    ttk.Separator(app.kernel_panel_frame, orient='horizontal').pack(fill='x', pady=10, padx=5)

    # 3. Cria o segundo visualizador
    create_channel_viewer(app.kernel_panel_frame, "Viewer B", weights, in_channels)



def on_canvas_configure(app, event):
    app.winfo_toplevel().after_idle(lambda: app.activation_grid_canvas.configure(scrollregion=app.activation_grid_canvas.bbox("all")))

def on_detail_canvas_resize(app, event):
    update_activation_detail(app, state.last_detail_maps, state.last_detail_index)

def _on_mousewheel(app, event):
    if event.num == 5 or event.delta < 0: direction = 1
    else: direction = -1
    app.activation_grid_canvas.yview_scroll(direction, "units")

def enable_controls(app, is_enabled):
    state = 'normal' if is_enabled else 'disabled'
    for child in app.control_frame.winfo_children(): child.configure(state=state)
    # Supondo que o right_top_frame foi salvo como atributo em ui_layout
    # Se não, a lógica de encontrar os widgets precisaria ser ajustada.
    # Por simplicidade, vamos focar nos botões de navegação.
    
# --- Conexão dos Callbacks com a UI ---
def setup_callbacks(app):
    """Conecta todos os widgets da UI às suas funções de callback."""
    app.dataset_selector.bind("<<ComboboxSelected>>", lambda e: on_dataset_selected(app, e))
    app.model_selector.bind("<<ComboboxSelected>>", lambda e: load_model_and_data(app))
    
    app.cnn_radio.config(command=lambda: populate_layer_selector(app))
    app.fc_radio.config(command=lambda: populate_layer_selector(app))
    app.layer_selector.bind("<<ComboboxSelected>>", lambda e: update_all_visuals(app))

    def prev_image(app_instance):
        app_instance.index_var.set(str(max(0, int(app_instance.index_var.get()) - 1)))
        update_all_visuals(app_instance)
    def next_image(app_instance):
        app_instance.index_var.set(str(int(app_instance.index_var.get()) + 1))
        update_all_visuals(app_instance)

    app.prev_button.config(command=lambda: prev_image(app))
    app.next_button.config(command=lambda: next_image(app))
    app.index_entry.bind("<Return>", lambda e: update_all_visuals(app))
    
    app.grid_checkbutton.config(command=lambda: update_input_panel(app))
    
    app.activation_grid_canvas.bind("<Button-1>", lambda e: on_grid_click(app, e))
    app.activation_grid_canvas.bind("<Configure>", lambda e: on_canvas_configure(app, e))
    app.activation_grid_canvas.bind("<MouseWheel>", lambda e: _on_mousewheel(app, e))
    app.activation_grid_canvas.bind("<Button-4>", lambda e: _on_mousewheel(app, e))
    app.activation_grid_canvas.bind("<Button-5>", lambda e: _on_mousewheel(app, e))
    
    app.detail_canvas.bind("<Configure>", lambda e: on_detail_canvas_resize(app, e))
    app.detail_grid_checkbutton.config(command=lambda: on_detail_canvas_resize(app, None))

    enable_controls(app, False)

def create_channel_viewer(parent_frame, title, weights, in_channels):
    """Cria um visualizador interativo para um canal de um kernel, com um título."""
    
    frame = ttk.Frame(parent_frame)
    frame.pack(pady=5, padx=5, fill='x')

    # Adiciona o título para diferenciar os viewers (ex: "Viewer A")
    ttk.Label(frame, text=title, font=("", 10, "bold")).pack()

    nav_frame = ttk.Frame(frame)
    nav_frame.pack()
    
    matrix_frame = ttk.Frame(frame, relief="sunken", borderwidth=2)
    matrix_frame.pack(padx=10, pady=5)

    k_size = weights.shape[1]
    channel_var = tk.IntVar(value=0)
    
    weight_labels = [[ttk.Label(matrix_frame, width=10, anchor="center") for _ in range(k_size)] for _ in range(k_size)]
    for r in range(k_size):
        for c in range(k_size): 
            weight_labels[r][c].grid(row=r, column=c, padx=1, pady=1)

    def update_matrix():
        channel_idx = channel_var.get()
        channel_label_var.set(f"{channel_idx + 1}/{in_channels}")
        kernel_slice = weights[channel_idx]
        for r in range(k_size):
            for c in range(k_size): 
                weight_labels[r][c].config(text=format_weight(kernel_slice[r,c].item()))

    def nav(direction):
        current_idx = channel_var.get()
        new_idx = (current_idx + direction) % in_channels
        channel_var.set(new_idx)
        update_matrix()

    channel_label_var = tk.StringVar()
    ttk.Button(nav_frame, text="< Prev", command=lambda: nav(-1)).pack(side="left")
    ttk.Label(nav_frame, textvariable=channel_label_var, width=6, anchor="center").pack(side="left")
    ttk.Button(nav_frame, text="Next >", command=lambda: nav(1)).pack(side="left")
    
    update_matrix()