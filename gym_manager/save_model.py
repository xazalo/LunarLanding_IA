import torch
import os

def save_model(model, base_path="models", filename="qn_lufind.pth"):
    """
    Save model state dictionary to file
    Guarda el diccionario de estado del modelo en archivo
    
    Args/Argumentos:
        model: PyTorch model to save / Modelo PyTorch a guardar
        base_path: Directory path / Ruta del directorio
        filename: Model filename / Nombre del archivo del modelo
    """
    os.makedirs(base_path, exist_ok=True)  # Create dir if not exists / Crear directorio si no existe
    full_path = os.path.join(base_path, filename)
    torch.save(model.state_dict(), full_path)
    print(f"Model saved successfully to {full_path}")
    print(f"Modelo guardado exitosamente en {full_path}")

def load_model(model, base_path="models", filename="qn_lufind.pth"):
    """
    Load model state dictionary from file
    Carga el diccionario de estado del modelo desde archivo
    
    Args/Argumentos:
        model: PyTorch model to load into / Modelo PyTorch a cargar
        base_path: Directory path / Ruta del directorio
        filename: Model filename / Nombre del archivo del modelo
    
    Returns/Retorna:
        None (modifies model in-place) / None (modifica el modelo directamente)
    """
    full_path = os.path.join(base_path, filename)
    if os.path.exists(full_path):
        model.load_state_dict(torch.load(full_path))
        model.eval()  # Set to evaluation mode / Poner en modo evaluación
        print("Model loaded successfully / Modelo cargado correctamente")
    else:
        print("Model not found, will train from scratch / No se encontró el modelo, se entrenará desde cero")

def save_metadata(epsilon, landings, crashes, soft_crashes, base_path="models", filename="metadata.pth"):
    """
    Save training metadata to file
    Guarda metadatos de entrenamiento en archivo
    
    Args/Argumentos:
        epsilon: Current exploration rate / Tasa actual de exploración
        landings: Successful landings count / Contador de aterrizajes exitosos
        crashes: Crash landings count / Contador de aterrizajes fallidos
        soft_crashes: Soft crash count / Contador de aterrizajes semi-fallidos
        base_path: Directory path / Ruta del directorio
        filename: Metadata filename / Nombre del archivo de metadatos
    """
    os.makedirs(base_path, exist_ok=True)  # Create dir if not exists / Crear directorio si no existe
    full_path = os.path.join(base_path, filename)
    data = {
        "epsilon": epsilon,
        "landings": landings,
        "crashes": crashes,
        "soft_crashes": soft_crashes
    }
    torch.save(data, full_path)
    print(f"Metadata saved successfully to {full_path}")
    print(f"Metadatos guardados exitosamente en {full_path}")

def load_metadata(base_path="models", filename="metadata.pth"):
    """
    Load training metadata from file
    Carga metadatos de entrenamiento desde archivo
    
    Args/Argumentos:
        base_path: Directory path / Ruta del directorio
        filename: Metadata filename / Nombre del archivo de metadatos
    
    Returns/Retorna:
        tuple: (epsilon, landings, crashes, soft_crashes)
    """
    full_path = os.path.join(base_path, filename)
    if os.path.exists(full_path):
        data = torch.load(full_path)
        print("Metadata loaded successfully / Metadatos cargados correctamente")
        return data["epsilon"], data["landings"], data["crashes"], data["soft_crashes"]
    else:
        print("Metadata not found, using defaults / No se encontró metadata, se usará por defecto")
        return 1.0, 0, 0, 0  # Default values / Valores por defecto