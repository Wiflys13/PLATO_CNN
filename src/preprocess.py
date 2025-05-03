import os
import time
import pickle
import pandas as pd
from config import OBSID_DIR, PREFIXES, OBSID_LISTS, PREPROCESSED_IMAGES_DIR, PRUEBA  # Import necessary configurations
from classes import ImagePLATO  

def preprocess_single_obsid(obsid: int, model_key: str, image_type: str) -> list:
    """
    Preprocess a single OBSID by performing several steps such as:
    - Processing CSV files to obtain temperature data.
    - Loading FITS images.
    - Merging temperature data with FITS images.
    - Removing background from the images.
    - Cropping the images.
    - Adding EEF (Extra Electric Field) to the cropped images.

    Args:
        obsid (int): The OBSID to be processed.
        model_key (str): The model key used to get the prefix from `config.py`.
        image_type (str): The type of image (e.g., 'Plateaux' or 'BFT').
    
    Returns:
        list: A list of processed images for the given OBSID.
    """
    # Instantiate the ImagePLATO class
    image_plato = ImagePLATO(model_key, image_type)

    temperature_dataframes = []
    fits_dataframes = []

    # Format the OBSID with leading zeros
    obsid_str = str(obsid).zfill(5)

    print(f"Loading OBSID: {obsid_str}")

    # Define the folder path for the OBSID
    folder = os.path.join(OBSID_DIR, f"{obsid_str}_{model_key}")

    # Process the CSV file for temperature data
    temperature_df, process_message = image_plato.process_csv_file(obsid_str)
    print(process_message)

    if temperature_df is not None:
        temperature_dataframes.append(temperature_df)

    # Load the FITS files for the current OBSID
    fits_images, load_message = image_plato.load_fits(obsid_str)
    print(load_message)

    fits_dataframes.append(fits_images)

    # Merge the temperature data and FITS images
    combined_results, merge_message = image_plato.combine_data(temperature_dataframes[0], fits_dataframes[0])
    print(merge_message)

    # Remove background from the merged images
    background_removed_results, background_message = image_plato.remove_background(combined_results)
    print(background_message)

    # Crop the images
    cropped_images, crop_message = image_plato.crop_images(background_removed_results)
    print(crop_message)

    # Add EEF (Extra Electric Field) to the cropped images
    cropped_images_with_eef, eef_message = image_plato.add_eef_to_cropped_images(cropped_images)
    print(eef_message)

    # Free variables no longer needed
    del temperature_dataframes, fits_dataframes, combined_results, cropped_images

    return cropped_images_with_eef


def process_all_obsids(model_key: str, image_type: str) -> None:
    """
    Processes all OBSIDs for a given model and image type by calling `preprocess_single_obsid` for each OBSID.
    
    Args:
        model_key (str): The model key used to get the prefix from `config.py`.
        image_type (str): The type of image (e.g., 'Plateaux' or 'BFT').
    
    Returns:
        None: The function saves the results in a pickle file but does not return anything.
    """
    # Retrieve the OBSID list and prefix based on the provided model key and image type
    obsid_list = OBSID_LISTS.get(f'{model_key}_{image_type}', [])
    prefix = PREFIXES.get(model_key, '')

    # Check if both the OBSID list and prefix were successfully retrieved
    if not obsid_list or not prefix:
        raise ValueError(f"Invalid model or image type: {model_key}, {image_type}. Please ensure the model and type exist in both OBSID_LISTS and PREFIXES.")

    # List to store the final results
    total_list = []

    # Start time for processing
    start_time = time.time()

    # Iterate over each OBSID in the list
    for obsid in obsid_list:
        # Process a single OBSID
        processed_images = preprocess_single_obsid(obsid, model_key, image_type)
        total_list.extend(processed_images)

        # Free memory after each OBSID processing
        del processed_images
        print(f"Processed OBSID {obsid}.\n")

    # Convert the results into a DataFrame (if needed for further processing)
    df_total = pd.DataFrame(total_list)

    # Calculate the total execution time
    total_execution_time = time.time() - start_time
    execution_minutes = int(total_execution_time // 60)
    execution_seconds = int(total_execution_time % 60)

    print(f"Total execution time: {execution_minutes} minutes, {execution_seconds} seconds.")

    # Verifica si la carpeta 'prueba' existe, y si no, créala
    if not os.path.exists(PRUEBA):
        os.makedirs(PRUEBA)

    # Ahora puedes guardar el archivo pickle sin problema
    output_filename = f"images_{model_key}_{image_type}.pkl"
    PICKLE_SAVE_PATH = os.path.join(PREPROCESSED_IMAGES_DIR, output_filename)

    with open(PICKLE_SAVE_PATH, "wb") as f:
        pickle.dump(total_list, f)

    print(f"Pickle file saved at: {PICKLE_SAVE_PATH}")
