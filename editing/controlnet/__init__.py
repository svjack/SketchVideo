import importlib

# 字符和子文件的对应关系
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
        raise ValueError(f"无效的控制模块: {name}")

