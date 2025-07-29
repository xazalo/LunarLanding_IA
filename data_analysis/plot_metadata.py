import torch
import matplotlib.pyplot as plt
import os

def plot_metadata(filepath, selected_fields, selected_file):
    try:
        metadata = torch.load(filepath, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Error al cargar el archivo / Error loading file: {e}")
        return

    episodes = []
    for sublist in metadata:
        if not isinstance(sublist, list):
            print("⚠️ Estructura inesperada / Unexpected structure")
            return
        episodes.extend(sublist)

    if not (episodes and isinstance(episodes[0], dict)):
        print("⚠️ Formato incorrecto / Invalid format")
        return

    data_per_field = {field: [] for field in selected_fields}
    for ep in episodes:
        for field in selected_fields:
            data_per_field[field].append(ep.get(field, None))

    # Obtener nombre base (sin prefijo y sin extensión)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    if base_name.startswith("metadata_"):
        base_name = base_name[len("metadata_"):]

    # Crear carpeta con nombre de la IA
    output_dir = os.path.join("screenshoots", selected_file)
    os.makedirs(output_dir, exist_ok=True)

    for field in selected_fields:
        values = data_per_field[field]

        plt.figure(figsize=(12, 4))
        plt.plot(range(1, len(values) + 1), values, marker='o', label=field)
        plt.xlabel("Episodio / Episode")
        plt.ylabel(field)
        plt.title(f"Evolución de {field} / {field} Trend")
        plt.grid(True)
        plt.legend()

        # Guardar el gráfico dentro de la carpeta de la IA
        filename = os.path.join("screenshoots", selected_file, f"{field}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ Gráficos guardados en '{selected_file}' con nombres '<campo>.png'.")
