import os
import time
import pickle
import pandas as pd
from .config import OBSID_DIR, PREFIXES, OBSID_LISTS, PREPROCESSED_IMAGES_DIR, PRUEBA  # Import necessary configurations
from .classes import ImagePLATO  

def preprocess_obsids(model_key: str, image_type: str) -> None:
    """
    Preprocesses a list of OBSIDs for a given model and image type by performing several steps such as:
    - Processing CSV files to obtain temperature data.
    - Loading FITS images.
    - Merging temperature data with FITS images.
    - Removing background from the images.
    - Cropping the images.
    - Adding EEF (Extra Electric Field) to the cropped images.
    
    The results are saved as a pickle file containing the processed images.

    Args:
        model_key (str): The model key used to get the prefix from `config.py` (e.g., 'FM16').
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

    # Instantiate the ImagePLATO class
    image_plato = ImagePLATO(model_key, image_type)

    # Iterate over each OBSID in the list
    for obsid in obsid_list:
        temperature_dataframes = []
        fits_dataframes = []

        # Format the OBSID with leading zeros
        obsid_str = str(obsid).zfill(5)

        print("-----------------------------------")    
        print(f"Loading OBSID: {obsid_str}\n")

        # Define the folder path for the OBSID
        folder = os.path.join(OBSID_DIR, f"{obsid_str}_{prefix}")

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

        # Free up memory by deleting the intermediate dataframes
        del temperature_dataframes, fits_dataframes

        # Remove background from the merged images
        background_removed_results, background_message = image_plato.remove_background(combined_results)
        print(background_message)

        # Crop the images
        cropped_images, crop_message = image_plato.crop_images(background_removed_results)
        print(crop_message)

        # Add EEF (Extra Electric Field) to the cropped images
        cropped_images_with_eef, eef_message = image_plato.add_eef_to_cropped_images(cropped_images)
        print(eef_message)

        # Append the processed images to the total list
        total_list.extend(cropped_images_with_eef)

        # Free variables no longer needed
        del temperature_df, combined_results, cropped_images

        print("-----------------------------------")
        print("          Next OBSID          ")

    # Convert the results into a DataFrame (if needed for further processing)
    df_total = pd.DataFrame(total_list)

    # Calculate the total execution time
    total_execution_time = time.time() - start_time
    execution_minutes = int(total_execution_time // 60)
    execution_seconds = int(total_execution_time % 60)

    print(f"Total execution time: {execution_minutes} minutes, {execution_seconds} seconds.")

    # Define the dynamic output filename
    output_filename = f"{model_key}_{image_type}_images.pkl"

    # Define the path to save the pickle file
    PICKLE_SAVE_PATH = os.path.join(PRUEBA, output_filename)

    # Save the total list of processed images as a pickle file
    with open(PICKLE_SAVE_PATH, "wb") as f:
        pickle.dump(total_list, f)

    print(f"Pickle file saved at: {PICKLE_SAVE_PATH}")
