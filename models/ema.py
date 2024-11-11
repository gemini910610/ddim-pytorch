from torch import nn

class EMAHelper:
    def __init__(self, mu):
        self.mu = mu
        self.weights = {}

    def get_module(self, model):
        if isinstance(model, nn.DataParallel):
            return model.module
        return model

    def register(self, model):
        module = self.get_module(model)
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                self.weights[name] = parameter.data.clone()
    
    def update(self, model):
        # W' = mu * W + (1 - mu) * W_model
        module = self.get_module(model)
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                self.weights[name] = self.mu * self.weights[name] + (1 - self.mu) * parameter.data
    
    def ema(self, model):
        module = self.get_module(model)
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                parameter.data.copy_(self.weights[name])
    
    def load_state_dict(self, state_dict):
        self.weights = state_dict
    
    def state_dict(self):
        return self.weights
