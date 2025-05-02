import importlib
import inspect
from pathlib import Path
from typing import Dict, Type
import torch.nn as nn

class ModelFactory:
    _instance = None
    _initialized = False
    
    def __new__(cls, models_dir='models'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, models_dir='models'):
        if not self._initialized:
            self.models_dir = models_dir
            self.models_dict: Dict[str, Type[nn.Module]] = {}
            self._discover_models()
            self._initialized = True
    
    def _discover_models(self):
        """Discover and register all models in the models directory"""
        model_files = [
            f.stem for f in Path(self.models_dir).glob('*.py') 
            if not f.name.startswith('_') and f.name != 'base_model.py'
        ]
        
        for module_name in model_files:
            try:
                module = importlib.import_module(f"{self.models_dir}.{module_name}")
                self._register_module_models(module)
            except Exception as e:
                print(f"Failed to load models from {module_name}: {e}")
    
    def _register_module_models(self, module):
        """Register all models in a module"""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, nn.Module) and 
                obj.__module__ == module.__name__ and
                not name.startswith('_')):
                
                self.models_dict[name] = obj
    
    def get_model(self, model_name: str, *args, **kwargs) -> nn.Module:
        """Instantiate a model by name"""
        if model_name not in self.models_dict:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models_dict.keys())}")
        return self.models_dict[model_name](*args, **kwargs)
    
    def list_models(self) -> list:
        """Return a list of available model names"""
        return list(self.models_dict.keys())

def register_model(cls):
    """Decorator to automatically register models with the factory"""
    ModelFactory().models_dict[cls.__name__] = cls
    return cls