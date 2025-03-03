#config.py

# Rutas base
BASE_DIR = 'data/'
OBSID_DIR = BASE_DIR + 'OBSID/'
IMAGES_DIR = BASE_DIR + 'images/' 
PREPROCESSED_IMAGES_DIR = BASE_DIR + 'processed/'
PRUEBA = BASE_DIR + 'prueba'
ANALYSIS_DIR = BASE_DIR + 'analysis/'

# Prefijos de archivos según el modelo de vuelo
PREFIXES = {
    'FM3': 'INTA_duvel',
    'FM6': 'INTA_gueuze',
    'FM10': 'INTA_karmeliet',
    'FM13': 'INTA_lupulus',
    'FM16': 'INTA_orval',
    'EM': 'INTA_em'
}

# Listas de OBSIDs según el tipo de imagen
OBSID_LISTS = {
    'FM3_Plateaux': [1024, 1035, 1039, 1044, 1048],
    'FM3_BFT': [1052],
    'EM_Plateaux': [670, 683, 701, 711, 727],
    'EM_BFT': [734],
    'FM6_Plateaux': [1391, 1397, 1399, 1401],
    'FM6_BFT': [1406],
    'FM10_Plateaux': [1653, 1663, 1665, 1667, 1669],
    'FM10_BFT': [1673],
    'FM13_Transients_bajada': [1945, 1946, 1947, 1948, 1949],
    'FM13_Transients_subida': [1989, 1991, 1992],
    'FM16_Plateaux': [2356, 2362]
}