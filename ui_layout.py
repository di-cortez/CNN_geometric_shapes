# ui_layout.py
import tkinter as tk
from tkinter import ttk
from callbacks import setup_callbacks
from utils import find_datasets

class Application(ttk.Frame):
    """
    Classe principal da interface gráfica.
    Responsável por criar e organizar todos os widgets na tela.
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        # Configurações da janela principal (root)
        parent.title("Neural Network Analyzer")
        parent.geometry("1300x800")
        
        # Variáveis do Tkinter que estão diretamente ligadas aos widgets
        self.index_var = tk.StringVar(value="0")
        self.show_grid_var = tk.BooleanVar(value=False)
        self.show_detail_grid_var = tk.BooleanVar(value=False)
        self.layer_type_var = tk.StringVar(value='CNN')
        self.label_var = tk.StringVar(value="True Label:")
        self.pred_var = tk.StringVar(value="Prediction:")
        self.stats_var = tk.StringVar(value="Stats: (select a filter)")
        self.status_var = tk.StringVar(value="Ready. Please select a dataset to begin.")
        
        # Cria todos os widgets da interface
        self._create_widgets()
        
        # Conecta os widgets às suas funções de callback
        setup_callbacks(self)

    def _create_widgets(self):
        """Cria e posiciona todos os widgets na janela."""
        
        # --- Top Frame (Seleção de Dataset e Modelo) ---
        top_frame = ttk.Frame(self, padding=(0, 0, 0, 10)); top_frame.pack(fill='x')
        top_frame.columnconfigure(1, weight=1)

        left_top_frame = ttk.Frame(top_frame); left_top_frame.grid(row=0, column=0, sticky='w')
        ttk.Label(left_top_frame, text="Select Dataset:").pack(anchor='w')
        self.dataset_selector = ttk.Combobox(left_top_frame, values=find_datasets(), width=50, state="readonly")
        self.dataset_selector.pack(fill='x')
        ttk.Label(left_top_frame, text="Select Model:").pack(anchor='w', pady=(5,0))
        self.model_selector = ttk.Combobox(left_top_frame, width=50, state="readonly")
        self.model_selector.pack(fill='x')

        right_top_frame = ttk.Frame(top_frame); right_top_frame.grid(row=0, column=2, sticky='e')
        self.cnn_radio = ttk.Radiobutton(right_top_frame, text="CNN", variable=self.layer_type_var, value='CNN')
        self.cnn_radio.pack(side='left')
        self.fc_radio = ttk.Radiobutton(right_top_frame, text="FC", variable=self.layer_type_var, value='FC')
        self.fc_radio.pack(side='left', padx=(5,15))
        ttk.Label(right_top_frame, text="Select Layer:").pack(anchor='w')
        self.layer_selector = ttk.Combobox(right_top_frame, width=35, state="readonly")
        self.layer_selector.pack(fill='x')

        # --- Main Frame (Layout principal de 3 colunas) ---
        main_frame = ttk.Frame(self); main_frame.pack(fill='both', expand=True)
        main_frame.columnconfigure(0, weight=35); main_frame.columnconfigure(1, weight=40); main_frame.columnconfigure(2, weight=25)
        main_frame.rowconfigure(0, weight=1)

        # --- Left Panel (Imagem de Entrada) ---
        left_panel = ttk.LabelFrame(main_frame, text="Input Image", padding=10)
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        self.control_frame = ttk.Frame(left_panel); self.control_frame.pack(pady=(0, 10))
        self.prev_button = ttk.Button(self.control_frame, text="< Prev"); self.prev_button.pack(side="left")
        self.index_entry = ttk.Entry(self.control_frame, textvariable=self.index_var, width=8)
        self.index_entry.pack(side="left", padx=5)
        self.next_button = ttk.Button(self.control_frame, text="Next >"); self.next_button.pack(side="left")
        
        self.image_canvas = tk.Canvas(left_panel, borderwidth=2, relief="solid", bg="black")
        self.image_canvas.pack(pady=5, fill='both', expand=True)
        
        options_frame = ttk.Frame(left_panel); options_frame.pack(anchor='w', pady=(5,0))
        self.grid_checkbutton = ttk.Checkbutton(options_frame, text="Grid", variable=self.show_grid_var)
        self.grid_checkbutton.pack(side='left')
        
        ttk.Label(left_panel, textvariable=self.label_var).pack(anchor='w', pady=(5,0))
        ttk.Label(left_panel, textvariable=self.pred_var).pack(anchor='w')

        # --- Center Panel (Grid de Ativação e Kernel) ---
        center_panel = ttk.Frame(main_frame); center_panel.grid(row=0, column=1, sticky='nsew', padx=5)
        center_panel.rowconfigure(0, weight=50); center_panel.rowconfigure(1, weight=50)

        activation_grid_frame = ttk.LabelFrame(center_panel, text="Activation Grid", padding=5)
        activation_grid_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5))
        activation_grid_frame.grid_rowconfigure(0, weight=1); activation_grid_frame.grid_columnconfigure(0, weight=1)
        self.activation_grid_canvas = tk.Canvas(activation_grid_frame, bg='gray20')
        self.activation_grid_canvas.grid(row=0, column=0, sticky='nsew')
        v_scrollbar = ttk.Scrollbar(activation_grid_frame, orient="vertical", command=self.activation_grid_canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar = ttk.Scrollbar(activation_grid_frame, orient="horizontal", command=self.activation_grid_canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        self.activation_grid_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        self.kernel_panel_frame = ttk.LabelFrame(center_panel, text="Kernel Panel", padding=5)
        self.kernel_panel_frame.grid(row=1, column=0, sticky='nsew')
        ttk.Label(self.kernel_panel_frame, text="Select a CNN filter to see its kernel.").pack(pady=20)

        # --- Right Panel (Detalhe da Ativação) ---
        right_panel = ttk.Frame(main_frame); right_panel.grid(row=0, column=2, sticky='nsew', padx=(5, 0))
        self.detail_frame = ttk.LabelFrame(right_panel, text="Activation Detail", padding=5)
        self.detail_frame.pack(fill='both', expand=True, pady=0)
        self.detail_canvas = tk.Canvas(self.detail_frame, bg='gray10')
        self.detail_canvas.pack(fill='both', expand=True, pady=(0,5))
        
        detail_controls_frame = ttk.Frame(self.detail_frame); detail_controls_frame.pack(fill='x', anchor='w')
        self.detail_grid_checkbutton = ttk.Checkbutton(detail_controls_frame, text="Grid", variable=self.show_detail_grid_var)
        self.detail_grid_checkbutton.pack(side='left')
        ttk.Label(detail_controls_frame, textvariable=self.stats_var).pack(side='left', padx=10)

        # --- Status Bar ---
        status_bar = ttk.Label(self, textvariable=self.status_var, relief='sunken', anchor='w', padding=5)
        status_bar.pack(side='bottom', fill='x')