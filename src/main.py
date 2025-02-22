# src/main.py

import os
from src.classes import ImageAnalysis
from src.utils import load_preprocessed_images, plot_statistics_distribution
from src.config import PREFIXES, OBSID_LISTS

def analyze_model_and_image_type(model, image_type):
    """
    Analyze a specific model and image type.

    Args:
        model (str): The flight model (e.g., 'FM3', 'FM16').
        image_type (str): The type of images (e.g., 'Plateaux', 'BFT').
    """
    print(f"Analyzing {model} - {image_type}...")

    # Initialize ImageAnalysis
    analysis = ImageAnalysis(model, image_type)

    # Load preprocessed images
    file_path = os.path.join('data', 'processed', f'{model}_images_{image_type}.pkl')
    preprocessed_images = load_preprocessed_images(file_path)

    # Get maximum EEF per FOV
    max_eef_per_fov = analysis.get_max_eef_per_fov(preprocessed_images)

    # Calculate average per OBSID
    avg_per_obsid = analysis.calculate_avg_per_obsid(max_eef_per_fov)

    # Fit and find maximum
    x_max, y_max = analysis.fit_and_find_maximum(avg_per_obsid)

    # Plot statistics distribution
    std_x, std_y = plot_statistics_distribution(max_eef_per_fov, analysis.analysis_dir, model, image_type)

    print(f"BFT for {model} - {image_type}: T = {x_max:.1f} ± {std_x:.1f} °C, EEF = {y_max:.1f} ± {std_y:.1f}%")

def main():
    # Ask the user for models and image types to analyze
    models = input("Enter the models to analyze (comma-separated, e.g., FM3,FM16): ").strip().split(',')
    image_types = input("Enter the image types to analyze (comma-separated, e.g., Plateaux,BFT): ").strip().split(',')

    # Iterate over models and image types
    for model in models:
        for image_type in image_types:
            # Check if the combination exists in OBSID_LISTS
            key = f"{model}_{image_type}"
            if key in OBSID_LISTS:
                analyze_model_and_image_type(model, image_type)
            else:
                print(f"Invalid combination: {model} - {image_type}. Skipping...")

if __name__ == "__main__":
    main()