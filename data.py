import os
import torch
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
        print(metadata)
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

    print("\nSeleccione hasta 2 campos (números o nombres, separados por comas) | Select up to 2 fields (numbers or names, comma-separated):")

    while True:
        user_input = input("\nCampos seleccionados | Selected fields: ").strip()
        if not user_input:
            print("❌ No se ingresaron campos. Intente de nuevo. | No fields entered. Try again.")
            continue

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

        if len(selected_fields) > 2:
            print("❌ Máximo 2 campos permitidos. | Maximum 2 fields allowed.")
            continue

        break

    print(f"\n✅ Campos seleccionados | Selected fields: {selected_fields}")
    return selected_fields


if __name__ == "__main__":
    selected_file = show_files("./models")
    if selected_file:
        filepath = f"./models/{selected_file}"
        print(f"\n📂 Archivo seleccionado | File selected: {selected_file}")

        while True:
            selected_fields = show_metadata_fields(filepath)
            data_analysis.plot_metadata.plot_metadata(filepath, selected_fields)

            choice = input("\n🔁 ¿Quieres seleccionar otros campos? (s/n) | Select other fields? (y/n): ").strip().lower()
            if choice not in ("y", "yes", "s", "si"):
                print("👋 Saliendo... | Exiting...")
                break
