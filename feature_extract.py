import torch


class FeatureExtractor:

    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = set(layer_names)
        self.handles = []
        self._features = {}

        for name, module in model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(self._hook(name))
                self.handles.append(handle)

    def _hook(self, name):
        def hook_fn(module, input, output):
            self._features[name] = output.detach()
        return hook_fn

    def get_features(self, inputs):
        self._features.clear()
        with torch.no_grad():
            device = next(self.model.parameters()).device

            # 确保输入在模型设备上
            inputs = {
                'data': {k: v.to(device) for k, v in inputs['data'].items()},
                'mask': {k: v.to(device) for k, v in inputs['mask'].items()}
            }

            _ = self.model(inputs)
        return self._features

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __del__(self):
        self.remove_hooks()
