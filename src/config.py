#config.py

# Rutas base
BASE_DIR = 'data/'
OBSID_DIR = BASE_DIR + 'OBSID/'
IMAGES_DIR = BASE_DIR + 'images/' 
PREPROCESSED_IMAGES_DIR = BASE_DIR + 'preprocessed/'
PROCESSED_IMAGES_DIR = BASE_DIR + 'processed/'
PRUEBA = BASE_DIR + 'prueba'
ANALYSIS_DIR = BASE_DIR + 'analysis/'

# Prefijos de archivos según el modelo de vuelo
PREFIXES = {
    'FM3': 'INTA_duvel',
    'FM6': 'INTA_gueuze',
    'FM10': 'INTA_karmeliet',
    'FM13': 'INTA_lupulus',
    'FM16': 'INTA_orval',
    'FM18': 'INTA_quintine',
    'FM20': 'INTA_science',
    'FM22': 'INTA_trappe',    
    'EM': 'INTA_em',
    'FMfast1': 'INTA_valdieu'
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
    'FM16_Plateaux': [2356, 2362, 2364, 2366, 2368],
    'FM16_BFT': [2371],
    'FM22_Plateaux': [2978, 2985, 2989, 2993, 2997],
    'FM22_BFT': [3001],
    'FM20_Plateaux': [3236, 3243, 3247, 3251, 3255],
    'FM20_BFT': [3259],
    'FM18_BFT': [2701],
    'FMfast1_Plateaux' : [3679]
}