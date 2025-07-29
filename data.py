import os
import torch
import tempfile
import data_analysis.plot_metadata

def show_files(directory, prefix="metadata_"):
    files = [
        f for f in os.listdir(directory)
        if f.startswith(prefix)
    ]

    if not files:
        print("⚠️  No se encontraron archivos de metadata en el directorio. | No metadata files found in the directory.")
        return None

    files = sorted(files)
    clean_names = [os.path.splitext(f[len(prefix):])[0] for f in files]

    print("\n🗂️  Archivos de metadata disponibles | Available metadata files:\n")
    for i, name in enumerate(clean_names):
        print(f"  [{i + 1}] {name}")

    while True:
        try:
            selection = int(input("\nIngrese el número del archivo | Enter file number: "))
            if 1 <= selection <= len(files):
                selected_file = files[selection - 1]
                print(f"\n✅ Archivo seleccionado | Selected file: {selected_file}")
                return selected_file
            else:
                print("❌ Número fuera de rango. Intente de nuevo. | Number out of range. Try again.")
        except ValueError:
            print("❌ Entrada inválida. Ingrese un número. | Invalid input. Please enter a number.")


def show_metadata_fields(filepath):
    try:
        metadata = torch.load(filepath, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Error al cargar el archivo: {e} | Error loading file: {e}")
        return []

    if not (isinstance(metadata, list) and metadata and
            isinstance(metadata[0], list) and metadata[0] and
            isinstance(metadata[0][0], dict)):
        print("⚠️  El archivo no tiene el formato esperado (lista de listas de diccionarios). | File does not match expected structure (list of lists of dicts).")
        return []

    fields = list(metadata[0][0].keys())

    print("\n📊 Campos disponibles | Available fields:\n")
    for i, field in enumerate(fields):
        print(f"  [{i+1}] {field}")
    print(f"  [13] Todos los campos / All fields")

    print("\nSeleccione campos (números o nombres, separados por comas) | Select fields (numbers or names, comma-separated):")

    while True:
        user_input = input("\nCampos seleccionados | Selected fields: ").strip()
        if not user_input:
            print("❌ No se ingresaron campos. Intente de nuevo. | No fields entered. Try again.")
            continue

        if user_input.strip() == "13":
            print("\n✅ Todos los campos seleccionados. | All fields selected.")
            return fields

        candidates = []
        for part in user_input.replace(',', ' ').split():
            candidates.append(part.strip())

        selected_fields = []
        invalid = []

        for item in candidates:
            if item.isdigit():
                idx = int(item)
                if 1 <= idx <= len(fields):
                    selected_fields.append(fields[idx-1])
                else:
                    invalid.append(item)
            else:
                if item in fields:
                    selected_fields.append(item)
                else:
                    invalid.append(item)

        if invalid:
            print(f"❌ Campos inválidos: {invalid} | Invalid fields: {invalid}")
            continue

        if len(selected_fields) == 0:
            print("❌ Debes seleccionar al menos un campo. | Select at least one field.")
            continue

        break

    print(f"\n✅ Campos seleccionados | Selected fields: {selected_fields}")
    return selected_fields


def load_episodes(filepath):
    """Carga y aplana los episodios de metadata."""
    metadata = torch.load(filepath, map_location="cpu", weights_only=False)
    episodes = []
    for sublist in metadata:
        if not isinstance(sublist, list):
            print("⚠️ Estructura inesperada / Unexpected structure")
            return []
        episodes.extend(sublist)
    if not (episodes and isinstance(episodes[0], dict)):
        print("⚠️ Formato incorrecto / Invalid format")
        return []
    return episodes


def truncate_episodes_by_epsilon(episodes, low_threshold=0.4, reset_value=1.0):
    """
    Detecta el primer punto donde epsilon sube desde < low_threshold hasta >= reset_value,
    y devuelve sólo episodios desde ese punto hacia adelante.
    Si no encuentra, devuelve la lista original.
    """
    epsilons = [ep.get("epsilon", None) for ep in episodes]
    for i in range(1, len(epsilons)):
        prev = epsilons[i-1]
        curr = epsilons[i]
        if prev is not None and curr is not None:
            if prev < low_threshold and curr >= reset_value:
                print(f"🔁 Reinicio detectado en episodio {i+1}. Se descartan episodios anteriores.")
                return episodes[i:]
    return episodes


if __name__ == "__main__":
    selected_file = show_files("./models")
    if selected_file:
        filepath = f"./models/{selected_file}"
        print(f"\n📂 Archivo seleccionado | File selected: {selected_file}")

        # Cargamos una vez todos los episodios para la detección y recorte
        episodes = load_episodes(filepath)
        if not episodes:
            exit(1)

        while True:
            selected_fields = show_metadata_fields(filepath)
            if not selected_fields:
                print("❌ No se seleccionaron campos válidos.")
                continue

            # Si epsilon está entre campos seleccionados, filtramos los episodios
            if any(f.lower() == "epsilon" for f in selected_fields):
                episodes_filtered = truncate_episodes_by_epsilon(episodes)
            else:
                episodes_filtered = episodes

            # Guardamos temporalmente los episodios filtrados para que plot_metadata los lea
            # Reconstruimos la estructura original (lista de listas)
            # Aquí asumo que los episodios están todos en un solo sublista para simplificar:
            temp_metadata = [episodes_filtered]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
                torch.save(temp_metadata, tmp_file.name)
                temp_path = tmp_file.name

            # Llamamos a la función de plotting con el archivo temporal y campos seleccionados
            data_analysis.plot_metadata.plot_metadata(temp_path, selected_fields, selected_file)

            # Borramos el archivo temporal
            try:
                os.remove(temp_path)
            except Exception:
                pass

            choice = input("\n🔁 ¿Quieres seleccionar otros campos? (s/n) | Select other fields? (y/n): ").strip().lower()
            if choice not in ("y", "yes", "s", "si"):
                print("👋 Saliendo... | Exiting...")
                break
