import pickle

def load_preprocessed_images(file_path):
    """Load preprocessed images from a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print("Archivo cargado correctamente.")
            
            return data
    except Exception as e:
        print(f"Error al deserializar: {e}")
        return None

# Cargar el archivo correctamente
fm3_images = load_preprocessed_images("C:/Users/UX450FDX/Documents/Projects/PLATO_CNN/data/prueba/FM3_Plateaux_images.pkl")

# Diccionario para almacenar listas de cada temperatura
em_70, em_75, em_80, em_85, em_90 = [], [], [], [], []
