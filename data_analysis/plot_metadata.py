import torch
import matplotlib.pyplot as plt

def plot_metadata(filepath, selected_fields):
    try:
        # Español: Carga el archivo de metadatos usando PyTorch
        # English: Load metadata file using PyTorch
        metadata = torch.load(filepath, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Error al cargar el archivo / Error loading file: {e}")
        return

    # Español: Valores por defecto para last_avg y last_crash
    # English: Default values for last_avg and last_crash
    last_avg = None
    last_crash = None

    # Español: Acceso seguro a metadata[1][-1] (últimas estadísticas)
    # English: Safe access to metadata[1][-1] (last stats)
    if isinstance(metadata, list) and len(metadata) > 1:
        second_list = metadata[1]
        if isinstance(second_list, list) and second_list:
            last_stats = second_list[-1]
            if isinstance(last_stats, dict):
                last_avg = last_stats.get("average_score", None)
                last_crash = last_stats.get("crash_ratio", None)

    # Español: Aplanar lista de listas a lista de diccionarios (episodios)
    # English: Flatten list of lists into a list of dictionaries (episodes)
    episodes = []
    for sublist in metadata:
        if not isinstance(sublist, list):
            print("⚠️ Estructura inesperada / Unexpected structure")
            return
        episodes.extend(sublist)

    # Español: Validación de estructura de datos
    # English: Data structure validation
    if not (episodes and isinstance(episodes[0], dict)):
        print("⚠️ Formato incorrecto / Invalid format")
        return

    # Español: Extrae datos para los campos seleccionados
    # English: Extract data for selected fields
    data_per_field = {field: [] for field in selected_fields}
    for ep in episodes:
        for field in selected_fields:
            data_per_field[field].append(ep.get(field, None))

    # Español: Configuración del gráfico
    # English: Plot setup
    n_fields = len(selected_fields)
    plt.figure(figsize=(12, 4 * n_fields))

    # Español: Crear subgráficos para cada campo
    # English: Create subplots for each field
    for i, field in enumerate(selected_fields, start=1):
        plt.subplot(n_fields, 1, i)
        values = data_per_field[field]
        plt.plot(range(1, len(values) + 1), values, marker='o', label=field)
        plt.xlabel("Episodio / Episode")
        plt.ylabel(field)
        plt.title(f"Evolución de {field} / {field} Trend")
        plt.grid(True)
        plt.legend()

    plt.subplots_adjust(top=0.9)
    plt.show()