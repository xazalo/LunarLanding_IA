import torch
import os

def load_model(model, path="models/qn_lufind.pth"):
    """
    Load model weights from file / Cargar pesos del modelo desde archivo
    
    Args/Argumentos:
        model: PyTorch model instance / Instancia del modelo PyTorch
        path: Model file path / Ruta del archivo del modelo
    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()  # Set to evaluation mode / Poner en modo evaluación
        print("Model loaded successfully / Modelo cargado correctamente")
    else:
        print("Model not found, training from scratch / No se encontró el modelo, se entrenará desde cero")

def load_metadata(path="models/metadata.pth"):
    """
    Load training metadata from file / Cargar metadatos de entrenamiento desde archivo
    
    Args/Argumentos:
        path: Metadata file path / Ruta del archivo de metadatos
    
    Returns/Retorna:
        tuple: (epsilon, landings, crashes, soft_crashes)
    """
    if os.path.exists(path):
        data = torch.load(path, weights_only=False)
        print("Metadata loaded successfully / Metadatos cargados correctamente")
        return (
            data.get("epsilon", 1.0),  # Default 1.0 if missing / Valor por defecto 1.0 si no existe
            data.get("landings", 0),    # Default 0 if missing / Valor por defecto 0 si no existe
            data.get("crashes", 0),     # Default 0 if missing / Valor por defecto 0 si no existe
            data.get("soft_crashes", 0)  # Default 0 if missing / Valor por defecto 0 si no existe
        )
    else:
        print("Metadata not found, using defaults / No se encontró metadata, se usará por defecto")
        return 1.0, 0, 0, 0  # Default values / Valores por defecto

def initialize_state(model, model_path="models/qn_lufind.pth", metadata_path="models/metadata.pth"):
    """
    Load model and metadata if they exist, otherwise return default values
    Carga el modelo y la metadata si existen, de lo contrario retorna valores por defecto
    
    Args/Argumentos:
        model: DQN model instance / Instancia del modelo DQN
        model_path: Model file path / Ruta del archivo del modelo
        metadata_path: Metadata file path / Ruta del archivo de metadatos
    
    Returns/Retorna:
        tuple: (epsilon, landings, crashes, soft_crashes)
    """
    # Load model weights / Cargar pesos del modelo
    load_model(model, path=model_path)
    
    # Load training metadata / Cargar metadatos de entrenamiento
    epsilon, landings, crashes, soft_crashes = load_metadata(path=metadata_path)
    
    return epsilon, landings, crashes, soft_crashes