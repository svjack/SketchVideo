import importlib

# Mapping from control name to module name
name_to_module = {
    'full': 'controlnet.controlnet_cross_frame',
}

def import_controlnet_module(name):
    module_name = name_to_module.get(name)
    if module_name:
        module = importlib.import_module(module_name)
        CogVideoControlNetModel = module.CogVideoControlNetModel
        return CogVideoControlNetModel
    else:
        raise ValueError(f"Invalid control module type: {name}")

