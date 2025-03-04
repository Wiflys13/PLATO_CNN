# classes.py
import os
import re
import glob
import time
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.io import fits
from .config import OBSID_DIR, PREPROCESSED_IMAGES_DIR, PREFIXES, OBSID_LISTS

class ImagePLATO:
    def __init__(self, model, image_type):
        self.model = model
        self.image_type = image_type
        self.prefix = PREFIXES[model]
        self.obsids = OBSID_LISTS[f"{model}_{image_type}"]
        self.results = []

    def process_csv_file(self, obsid):
        try:
            file_prefix = f'{obsid}_{self.prefix}_TCS-HK'
            csv_files = glob.glob(os.path.join(OBSID_DIR, '*', f'{file_prefix}*.csv'))

            if not csv_files:
                return None, f"No CSV files found for OBSID {obsid}"

            processed_dfs = []
            for csv_path in csv_files:
                df = pd.read_csv(csv_path)
                selected_columns = ['timestamp', 'GTCS_TRP1_T_AVG', 'GTCS_TRP22_T_AVG']
                new_df = df[selected_columns].copy()
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp']).dt.floor('s')
                reduced_df = new_df.groupby('timestamp', as_index=False).first()
                reduced_df['obsid'] = obsid
                processed_dfs.append(reduced_df)

            combined_df = pd.concat(processed_dfs, ignore_index=True)
            return combined_df, f'Temperature data for OBSID {obsid} loaded successfully'

        except Exception as e:
            return None, f"Error processing CSV files for OBSID {obsid}: {e}"

    def load_fits(self, obsid):
        try:
            folder = os.path.join(OBSID_DIR, f'{obsid}_{self.prefix}')
            results = []
            fits_files = sorted([file for file in os.listdir(folder) if file.endswith('cube.fits')])

            for fits_file in fits_files:
                fov_number = re.search(r'CCD_(\d{5})', fits_file).group(1)
                with fits.open(os.path.join(folder, fits_file)) as hdulist:
                    for image_num, extension in enumerate(hdulist, start=1):
                        if 'IMAGE' in extension.name and extension.data is not None:
                            date_time_str = hdulist[0].header['DATE-OBS']
                            date_time_pd = pd.to_datetime(date_time_str)
                            truncated_date_time = date_time_pd.floor('s')
                            image_array = np.array(extension.data)
                            results.append({
                                'obsid': obsid,
                                'file_name': fits_file,
                                'fov_number': fov_number,
                                'date_time': truncated_date_time,
                                'image': image_array
                            })

            df = pd.DataFrame(results)
            return df, f"Loaded {len(df)} images for OBSID {obsid}"

        except Exception as e:
            return None, f"Error loading FITS images for OBSID {obsid}: {e}"

    def combine_data(self, temp_df, fits_images):
        try:
            if temp_df['obsid'].nunique() != 1 or fits_images['obsid'].nunique() != 1:
                return None, "Error: DataFrames have different 'obsid' values"

            if temp_df['obsid'].iloc[0] != fits_images['obsid'].iloc[0]:
                return None, "Error: 'obsid' values do not match in both DataFrames"

            temp_df = temp_df.drop(columns=['obsid'])
            combined_results = pd.merge(fits_images, temp_df, left_on='date_time', right_on='timestamp', how='left')

            if len(combined_results) == len(fits_images):
                return combined_results, f"All data combined for OBSID: {fits_images['obsid'].iloc[0]}"
            else:
                return None, "Not all data was combined"

        except Exception as e:
            return None, f"Error combining data: {e}"

    def remove_background(self, image_list):
        subtracted_results = []
        for i in range(1, len(image_list), 2):
            even_image = image_list.iloc[i]
            odd_image = image_list.iloc[i - 1]
            fov_number = int(even_image['fov_number']) / 2
            subtracted_frame = np.abs(np.array(even_image['image'], dtype=np.int16) - np.array(odd_image['image'][0], dtype=np.int16))
            subtracted_results.append({
                'file_name': even_image['file_name'],
                'fov_number': fov_number,
                'date_time': even_image['date_time'],
                'Temperature_TRP1': even_image['GTCS_TRP1_T_AVG'],
                'GTCS_TRP22_T_AVG': even_image['GTCS_TRP22_T_AVG'],
                'obsid': even_image['obsid'],
                'image_no_background': subtracted_frame
            })
        return subtracted_results, "Background removed from all images"

    def crop_images(self, subtracted_results):
        cropped_results = []
        for result in subtracted_results:
            image_no_bg = result['image_no_background']
            num_frames, _, _ = image_no_bg.shape
            for frame_num in range(num_frames):
                frame = image_no_bg[frame_num]
                max_pos = np.unravel_index(np.argmax(frame), frame.shape)
                row_start = max_pos[0] - 5
                row_end = max_pos[0] + 6
                col_start = max_pos[1] - 5
                col_end = max_pos[1] + 6
                row_start = max(0, row_start)
                row_end = min(frame.shape[0], row_end)
                col_start = max(0, col_start)
                col_end = min(frame.shape[1], col_end)
                cropped_image = np.zeros((11, 11))
                cropped_image[:row_end - row_start, :col_end - col_start] = frame[row_start:row_end, col_start:col_end]
                cropped_image = np.abs(cropped_image)
                cropped_results.append({
                    'file_name': result['file_name'],
                    'fov_number': result['fov_number'],
                    'date_time': result['date_time'],
                    'Temperature_TRP1': result['Temperature_TRP1'],
                    'GTCS_TRP22_T_AVG': result['GTCS_TRP22_T_AVG'],
                    'obsid': result['obsid'],
                    'frame': frame_num + 1,
                    'cropped_image': cropped_image
                })
        return cropped_results, "All images cropped"

    def calculate_eef(self, image):
        abs_image = np.abs(image)
        max_pos_10x10 = np.unravel_index(np.argmax(abs_image), abs_image.shape)
        row_start_10x10 = max(0, max_pos_10x10[0] - 5)
        row_end_10x10 = min(image.shape[0], max_pos_10x10[0] + 5)
        col_start_10x10 = max(0, max_pos_10x10[1] - 5)
        col_end_10x10 = min(image.shape[1], max_pos_10x10[1] + 5)
        area_10x10 = image[row_start_10x10:row_end_10x10, col_start_10x10:col_end_10x10]
        max_energy_2x2 = -np.inf
        for i in range(area_10x10.shape[0] - 1):
            for j in range(area_10x10.shape[1] - 1):
                energy_2x2 = np.sum(area_10x10[i:i+2, j:j+2])
                if energy_2x2 > max_energy_2x2:
                    max_energy_2x2 = energy_2x2
        energy_10x10 = np.sum(area_10x10)
        eef_2x2_10x10 = (max_energy_2x2 / energy_10x10) * 100
        return eef_2x2_10x10

    def add_eef_to_cropped_images(self, cropped_images):
        for cropped_image_info in cropped_images:
            cropped_image = cropped_image_info['cropped_image']
            eef = self.calculate_eef(cropped_image)
            cropped_image_info['eef'] = eef
        return cropped_images, "EEF added to all images."

    def get_max_eef_per_fov(self, cropped_images_with_eef):
        max_eef_per_fov = {}
        temp_sum = {}
        temp_count = {}
        for image in cropped_images_with_eef:
            obsid = image['obsid']
            fov_number = image['fov_number']
            eef = image['eef']
            temperature = image['Temperature_TRP1']
            key = (obsid, fov_number)
            if key not in max_eef_per_fov or eef > max_eef_per_fov[key]['eef']:
                max_eef_per_fov[key] = image
            temp_sum[obsid] = temp_sum.get(obsid, 0) + temperature
            temp_count[obsid] = temp_count.get(obsid, 0) + 1
        avg_temp_per_obsid = {}
        for obsid, total in temp_sum.items():
            avg_temp_per_obsid[obsid] = total / temp_count[obsid]
        for image in max_eef_per_fov.values():
            obsid = image['obsid']
            image['avg_temperature'] = avg_temp_per_obsid[obsid]
        max_eef_list = list(max_eef_per_fov.values())
        return max_eef_list, "Max EEF per FOV and average temperature per OBSID calculated successfully."

    def process_obsids(self):
        start_time = time.time()
        total_list = []
        cropped_images_with_eef = []

        for obsid in self.obsids:
            obsid_str = str(obsid).zfill(5)
            print("-----------------------------------")    
            print("Loading OBSID:", obsid_str)
            print()

            temperature, process_msg = self.process_csv_file(obsid_str)
            print(process_msg)
            if temperature is None:
                continue

            fits_images, load_msg = self.load_fits(obsid_str)
            print(load_msg)
            if fits_images is None:
                continue

            combined_results, combine_msg = self.combine_data(temperature, fits_images)
            print(combine_msg)
            if combined_results is None:
                continue

            subtracted_results, bg_msg = self.remove_background(combined_results)
            print(bg_msg)

            cropped_images, crop_msg = self.crop_images(subtracted_results)
            print(crop_msg)

            cropped_images_with_eef, eef_msg = self.add_eef_to_cropped_images(cropped_images)
            print(eef_msg)

            cropped_images_with_eef.append(cropped_images_with_eef)
            total_list.extend(cropped_images_with_eef)

            print("-----------------------------------")
            print("          Next OBSID          ")

        df_total = pd.DataFrame(total_list)

        total_time_seconds = time.time() - start_time
        total_minutes = int(total_time_seconds // 60)
        total_seconds = int(total_time_seconds % 60)
        print(f"Total execution time: {total_minutes} minutes, {total_seconds} seconds.")

        file_name = f"{self.model}_{self.image_type}_preprocessed_images.pkl"
        save_path = os.path.join(PREPROCESSED_IMAGES_DIR, file_name)
        with open(save_path, "wb") as f:
            pickle.dump(total_list, f)
        print(f"Pickle file saved at: {save_path}")
        
        
class ImageAnalysis:
    def __init__(self, model, image_type):
        """
        Initialize the ImageAnalysis class.

        Args:
            model (str): The flight model (e.g., 'FM3', 'FM16').
            image_type (str): The type of images (e.g., 'Plateaux', 'BFT').
        """
        self.model = model
        self.image_type = image_type
        self.analysis_dir = os.path.join('data', 'analysis', f'{model}_{image_type}')
        os.makedirs(self.analysis_dir, exist_ok=True)

    def get_max_eef_per_fov(self, preprocessed_images):
        """Get the maximum EEF per FOV."""
        max_per_fov = {}
        temp_sum = {}
        temp_count = {}

        for image in preprocessed_images:
            obsid = image['obsid']
            fov_number = image['fov_numero']
            eef = image['eef']
            temperature = image['Temperatura_TRP1']
            key = (obsid, fov_number)

            if key not in max_per_fov or eef > max_per_fov[key]['eef']:
                max_per_fov[key] = image

            temp_sum[obsid] = temp_sum.get(obsid, 0) + temperature
            temp_count[obsid] = temp_count.get(obsid, 0) + 1

        avg_temp_per_obsid = {
            obsid: temp_sum[obsid] / temp_count[obsid]
            for obsid in temp_sum
        }

        for image in max_per_fov.values():
            image['avg_temperature'] = avg_temp_per_obsid[image['obsid']]

        return list(max_per_fov.values())

    def calculate_avg_per_obsid(self, max_eef_per_fov):
        """Calculate the average EEF per OBSID."""
        avg_per_obsid = {}

        for image in max_eef_per_fov:
            obsid = image['obsid']
            eef_max = image['eef']
            avg_temp = image['avg_temperature']

            if obsid not in avg_per_obsid:
                avg_per_obsid[obsid] = {'avg_temperature': avg_temp, 'eef_max_list': []}

            avg_per_obsid[obsid]['eef_max_list'].append(eef_max)

        for obsid, data in avg_per_obsid.items():
            data['avg_eef'] = np.mean(data['eef_max_list'])
            del data['eef_max_list']

        return pd.DataFrame.from_dict(avg_per_obsid, orient='index').reset_index()

    def fit_and_find_maximum(self, df):
        """Fit a polynomial curve and find the maximum."""
        coefficients = np.polyfit(df['avg_temperature'], df['avg_eef'], 4)
        x = np.linspace(df['avg_temperature'].min(), df['avg_temperature'].max(), 100)
        y_fitted = np.polyval(coefficients, x)
        x_max = x[np.argmax(y_fitted)]
        y_max = np.max(y_fitted)

        plt.scatter(df['avg_temperature'], df['avg_eef'], color='red')
        plt.plot(x, y_fitted, color='darkgrey', linestyle='-', label='4th Degree Polynomial Fit')
        plt.xlabel('Temperature [˚C]')
        plt.ylabel('EEF 2x2/10x10')
        plt.xticks(range(-90, -65, 5))
        plt.yticks(range(20, 105, 10))
        plt.axvline(x_max, color='black', linestyle='--', label=f'BFT: T={x_max:.1f} ˚C')
        plt.legend()
        plt.savefig(os.path.join(self.analysis_dir, f'{self.model}_{self.image_type}_BestFocusTemperature_4thDegree.png'))
        plt.show()

        return x_max, y_max
    
    def plot_statistics_distribution(df, analysis_dir, model, image_type):
        """Plot the distribution of statistics."""
        plt.figure(figsize=(8, 6))
        sns.violinplot(y=df['x_max'], color='skyblue')
        median_x = df['x_max'].median()
        ci_x = sns.utils.ci(df['x_max'], which=95)
        plt.axhline(y=median_x, color='skyblue', linestyle='--', label=f'Median: {median_x:.1f}')
        plt.fill_between(x=[-2, 2], y1=ci_x[0], y2=ci_x[1], color='lightgray', alpha=0.3, label=f'95% CI: [{ci_x[0]:.1f}, {ci_x[1]:.1f}]')
        plt.ylabel('Temperature [°C]', fontsize=16)
        plt.legend(fontsize=14)
        plt.xlim(-1, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f'{model}_{image_type}_BFT_Distribution.png'))
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.violinplot(y=df['y_max'], color='salmon')
        median_y = df['y_max'].median()
        ci_y = sns.utils.ci(df['y_max'], which=95)
        plt.axhline(y=median_y, color='salmon', linestyle='--', label=f'Median: {median_y:.1f}')
        plt.fill_between(x=[-2, 2], y1=ci_y[0], y2=ci_y[1], color='lightgray', alpha=0.3, label=f'95% CI: [{ci_y[0]:.1f}, {ci_y[1]:.1f}]')
        plt.ylabel('EEF (%)', fontsize=16)
        plt.legend(fontsize=14)
        plt.xlim(-1, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f'{model}_{image_type}_EEF_Distribution.png'))
        plt.show()

        return df['x_max'].std(), df['y_max'].std()