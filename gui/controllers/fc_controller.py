"""Controllers for the Fully Connected (FC) layer visualization view."""
from __future__ import annotations


def clear_panels(app) -> None:
    """Clears all visual elements related to the FC view."""
    app.fc_panel.update_view([])

def update_fc_view(app) -> None:
    """Gathers data and updates the FC visualization panel."""
    state = app.state
    fc_layer_info = state.model_layers.get("FC", [])
    if not fc_layer_info:
        clear_panels(app)
        return

    layers_data = []
    activations = state.model.activations
    all_layer_names = [name for name, _ in state.model.named_modules()]

    # Find the true input to the first FC layer
    first_fc_name = fc_layer_info[0].split(" ")[0]
    input_tensor = None
    try:
        first_fc_idx = all_layer_names.index(first_fc_name)
        if first_fc_idx > 0:
            input_source_name = all_layer_names[first_fc_idx - 1]
            input_tensor = activations.get(input_source_name)
    except (ValueError, IndexError):
        pass

    # Add the "Input" layer as the first element
    layers_data.append({
        "name": "Input",
        "activation": input_tensor.view(-1) if input_tensor is not None else None,
        "weight": None, "bias": None,
    })

    # Loop through all detected FC layers
    for info_str in fc_layer_info:
        layer_name = info_str.split(" ")[0]
        layer_module = getattr(state.model, layer_name, None)
        
        if layer_module and hasattr(layer_module, 'weight'):
            layers_data.append({
                "name": layer_name,
                "activation": activations.get(layer_name),
                "weight": layer_module.weight.data,
                "bias": layer_module.bias.data if layer_module.bias is not None else None,
            })
    
    app.fc_panel.update_view(layers_data)