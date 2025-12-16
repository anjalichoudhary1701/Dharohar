import cv2
import torch
import shutil
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import random
import re
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import Counter
import torchvision.transforms as transforms
# NOTE: The 'ultralytics' library is required for the YOLO model wrappers in a real environment.
from ultralytics import YOLO
from classification_pth import SimpleCNN_Lipi, predict_single_image_by_path,predict_single_image_from_np, GRAYSCALE_MEAN, GRAYSCALE_STD
# --- NEW IMPORTS FOR DESKEWING ---
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate
from deskew import determine_skew
# ---------------------------------
# --- GLOBAL PATHS (Replace with actual paths) ---
Text_YOLO_path = r"C:\Users\AnjaliC\OneDrive - Dharohar\Desktop\test\text_yolov8_model 1.pt"
Line_YOLO_path = r"C:\Users\AnjaliC\OneDrive - Dharohar\Desktop\line_yolo_model\6_yolo_model.pt"
Charac_YOLO_path =r"C:\Users\AnjaliC\OneDrive - Dharohar\Desktop\test\character_yolov8_model 1.pt"
lipi_model_path = r"C:\Users\AnjaliC\OneDrive - Dharohar\Desktop\test\Devanagri_lipi.pth"
cover_info_model_path = r""
folder_path = r"C:\Users\AnjaliC\OneDrive - Dharohar\Desktop\test\test\deskewed_originals"

def Cover_information_page(folder_path,model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cover_info_w = 240
    cover_info_h = 240
    num_classes = 3
    Modal_path_lipi = r""
    model.load_state_dict(torch.load(MODEL_PATH_Lipi, map_location=device))
    print(f"✅ Successfully loaded model weights from {MODEL_PATH_Lipi}")

    classes_name_to_use = SimpleCNN_CoverInfo.CLASS_NAMES
    image_extensions= ('.jpeg','jpg','.png','.tif','.tiff')
    image_files = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(image_extensions):
            full_path = os.path.join(folder_path,filename)
            image_files.append(full_path)

            if len(image_files)>4:
                break
    return image_files
    for image_file in image_files:
        output={}
        predicted_class = predict_single_image(
            model=model,
            image_path=image_file,
            class_names =classes_name_to_use,
            device = device,
            img_h=cover_info_h,
            img_w=cover_info_w
        )
        output.append[image_file] = predicted_class

        class_list = list(output.values())

        final_classes = class_list[:]  # Copy the list to modify it

        for i in range(len(final_classes) - 1):  # Loop up to the second-to-last item
            if final_classes[i] == 'Cover Page' and final_classes[i + 1] == 'Cover Page':
                final_classes[i] = 'Information Sheet'

        final_results = {}
        cover_info_files = []

        for i in range(len(file_list)):
            filename = file_list[i]
            final_class = final_classes[i]

            # Assemble the full list of results
            final_results[filename] = final_class

        # Identify files classified as 'Cover Page' or 'Information Sheet'
        if final_class in ['Cover Page']:
            Cover_page = os.path.basename(filename)
        if final_class in ['Information Page']:
            Information_sheet = os.path.basename(filename)
            # --- 6. Final Return Structure ---
        return {
            "Cover_page": Cover_page,
            "Information_sheet": Information_sheet
        }

# --- HELPER FUNCTIONS FOR EXTERNAL UTILITIES (Stitching) ---

def get_homography(img1, img2):
    # This is the actual SIFT logic from your original plan, kept for context.
    sift = cv2.SIFT_create()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if des1.shape[0] < 2 or des2.shape[0] < 2:
        return None

    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]

    if len(good_matches) < 4:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H

def compensate_exposure(img_base, img_new, H):
    h_base, w_base = img_base.shape[:2]
    img_new_float = img_new.astype(np.float32)
    warped_new_temp = cv2.warpPerspective(img_new_float, H, (w_base, h_base),
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    mask_base = (img_base[:, :, 0] > 0).astype(np.uint8) if len(img_base.shape) == 3 else (img_base > 0).astype(
        np.uint8)
    mask_new_temp = (warped_new_temp[:, :, 0] > 0).astype(np.uint8) if len(warped_new_temp.shape) == 3 else (
            warped_new_temp > 0).astype(np.uint8)

    overlap_mask = cv2.bitwise_and(mask_base, mask_new_temp)

    if len(img_base.shape) == 3:
        overlap_mask_3ch = cv2.cvtColor(overlap_mask, cv2.COLOR_GRAY2BGR)
    else:
        overlap_mask_3ch = np.expand_dims(overlap_mask, axis=-1)

    base_overlap_pixels = img_base[overlap_mask_3ch > 0]
    new_overlap_pixels = warped_new_temp[overlap_mask_3ch > 0]

    if len(base_overlap_pixels) == 0 or len(new_overlap_pixels) == 0:
        return img_new

    mean_base_overlap = np.mean(base_overlap_pixels).astype(np.float32)
    mean_new_overlap = np.mean(new_overlap_pixels).astype(np.float32)

    if mean_new_overlap < 1.0:
        return img_new

    scale_factor = mean_base_overlap / mean_new_overlap
    compensated_new_img = np.clip(img_new.astype(np.float32) * scale_factor, 0, 255).astype(np.uint8)

    return compensated_new_img

def stitch_and_blend(base_img, new_img, H, max_dim=20000):
    """
            Intelligently stitches and blends two images using a new canvas that is only
            as large as necessary. Includes a check to prevent memory errors from bad homographies.

            Returns:
                The stitched image (np.uint8) if successful, or None if the canvas size is unreasonable.
            """
    h_base, w_base = base_img.shape[:2]
    h_new, w_new = new_img.shape[:2]

    # Calculate corners of the new image after warping
    corners_new = np.float32([[0, 0], [0, h_new], [w_new, h_new], [w_new, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_new, H)

    # Calculate dimensions of the final combined canvas
    all_corners = np.concatenate(
        (np.float32([[0, 0], [w_base, 0], [w_base, h_base], [0, h_base]]).reshape(-1, 1, 2), warped_corners),
        axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation matrix to shift the origin to positive coordinates
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

    # New combined dimensions
    new_w = x_max - x_min
    new_h = y_max - y_min

    # --- MEMORY SAFETY CHECK ---
    # Prevents allocation of massive arrays due to erroneous homography calculations
    if new_w > max_dim or new_h > max_dim:
        print(
            f"[ERROR] Stitching attempted to create a canvas of size ({new_h}x{new_w}). Forcing simple stitching.")
        return None

    # Create the new canvas
    # This is the line that previously failed due to the huge calculated dimensions
    result_img = np.zeros((new_h, new_w, 3), dtype=np.float32)

    # Warp both images into the new canvas
    cv2.warpPerspective(base_img.astype(np.float32), translation_matrix, (new_w, new_h),
                        result_img, borderMode=cv2.BORDER_TRANSPARENT)
    cv2.warpPerspective(new_img.astype(np.float32), translation_matrix @ H, (new_w, new_h),
                        result_img, borderMode=cv2.BORDER_TRANSPARENT)

    # Convert to 8-bit image and return
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)

    return result_img

# --- MAIN PIPELINE CLASS ---

class Line_count():
    # ... (__init__, _setup_output_directory, _run_yolo_predict, Lipi_detection, image_selector methods are unchanged) ...
    def __init__(self):
        # --- STATE INITIALIZATION ---
        self.folder_path = None
        self.image_list = []
        self.selected_image_paths = []
        self.image_text_extracted_np = []  # NumPy arrays of cropped text regions
        self.all_line_coordinates = []  # (x, y) coordinates for DBSCAN
        self.all_line_data = []  # Stores (coord, image_np, original_image_name)

        # --- MODEL INITIALIZATION (Loads models once for efficiency) ---
        self.text_yolo_model = None
        self.line_yolo_model = None
        self.charac_yolo_model = None

        try:
            # Attempt to load the models. This requires valid paths in a real environment.
            self.text_yolo_model = YOLO(Text_YOLO_path)
            self.line_yolo_model = YOLO(Line_YOLO_path)
            self.charac_yolo_model = YOLO(Charac_YOLO_path)
            print("YOLO models initialized.")
        except Exception as e:
            # If YOLO/paths fail, models remain None, and the predict function will handle the error.
            print(f"YOLO model initialization failed (Check paths): {e}")

    # --- MODEL HELPER FUNCTIONS ---
    def _setup_output_directory(self, base_folder_path):
        """Creates a new output directory based on the input folder."""
        output_dir_name = os.path.basename(base_folder_path) + "_analysis_output"
        self.output_folder = os.path.join(os.path.dirname(base_folder_path), output_dir_name)
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"✅ Output directory created at: {self.output_folder}")

    def _run_yolo_predict(self, image_np, model_type="text"):
        """Runs the actual YOLO prediction structure."""
        boxes = np.array([])
        model_map = {
            "text": self.text_yolo_model,
            "line": self.line_yolo_model,
            "charac": self.charac_yolo_model
        }

        model = model_map.get(model_type)

        if model is None:
            print(f"[ERROR] {model_type.upper()} model failed to initialize. Returning empty detections.")
            return np.array([])

        # --- REAL INFERENCE STRUCTURE ---
        try:
            # Setting conf_level for better separation
            if model_type == 'charac':
                conf_level = 0.1  # Lower confidence for more granular detection
            elif model_type == 'text':
                conf_level = 0.45
            else:  # line
                conf_level = 0.25

            results = model.predict(image_np, iou=0.25, conf=conf_level, verbose=False)
            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
        except Exception as e:
            print(f"[ERROR] {model_type.upper()} prediction failed: {e}")

        return boxes

    def Lipi_detection(self, line_image_np):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lipi_w = 240
        lipi_h = 60
        num_classes = 2
        MODEL_PATH_Lipi = r"C:\Users\AnjaliC\OneDrive - Dharohar\Desktop\test\Devanagri_lipi.pth"
        model = SimpleCNN_Lipi(num_classes=num_classes, img_w=lipi_w, img_h=lipi_h)
        model.to(device)
        model.eval()
        model.load_state_dict(torch.load(MODEL_PATH_Lipi, map_location=device))
        print(f"✅ Successfully loaded model weights from {MODEL_PATH_Lipi}")

        classes_name_to_use = SimpleCNN_Lipi.CLASS_NAMES

        predicted_class = predict_single_image_from_np(
            model=model,
            image_np=line_image_np,
            img_h=lipi_h,
            img_w=lipi_w,
            device=device,
            class_names=classes_name_to_use
        )
        return predicted_class

    def image_selector(self, folder_path):
        # Step 1: Randomly select up to 3 images
        self.folder_path = folder_path
        self.image_list = [img for img in os.listdir(folder_path)
                           if img.lower().endswith((".jpeg", ".png", ".jpg", ".tiff", ".tif"))]

        num_present = len(self.image_list)
        sample_size = min(3, num_present)

        if sample_size > 0:
            selected_files = random.sample(self.image_list, sample_size)
            selected_image_paths = [os.path.join(self.folder_path, f) for f in selected_files]
            print(f"Selected {sample_size} images.")
            return selected_image_paths
        else:
            print("No images are present.")
            return []

    def text_extract_and_crop(self, image_np):
        """
        Applies Text YOLO and returns a list of cropped text region NumPy arrays.
        """
        cropped_images = []
        if image_np is None:
            return cropped_images

        # NOTE: Using the internal _run_yolo_predict with the TEXT model
        boxes = self._run_yolo_predict(image_np, "text")

        if boxes.size > 0:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]

                # Safety checks
                x1 = max(0, x1);
                y1 = max(0, y1)
                x2 = min(image_np.shape[1], x2);
                y2 = min(image_np.shape[0], y2)

                if x1 < x2 and y1 < y2:
                    cropped_img = image_np[y1:y2, x1:x2]
                    cropped_images.append(cropped_img)

        return cropped_images

    # --- NEW DESKEWING METHOD ---
    def deskew_skimage_page(self, image_path, original_name):
        """
        Applies deskewing using determine_skew and rotate from skimage.
        Returns the deskewed image as a NumPy array (compatible with cv2).
        """
        try:
            # 1. Read the image using skimage (which reads in RGB float format)
            image = io.imread(image_path)

            # 2. Determine skew angle on grayscale image
            grayscale = rgb2gray(image)
            angle = determine_skew(grayscale)

            # 3. Apply rotation based on your custom condition
            if angle is None:
                print(f"[WARN] Deskew failed for {original_name}. Angle is None. Skipping rotation.")
                # Fallback: Read the original image using CV2 (BGR)
                deskewed_image_np = cv2.imread(image_path)
            else:
                if angle < -1.5:
                    # Apply rotation with a slight correction (+1)
                    rotated = rotate(image, angle + 1, resize=True, mode='constant', cval=1.0) * 255
                else:
                    # Apply rotation
                    rotated = rotate(image, angle, resize=True, mode='constant', cval=1.0) * 255

                print(f"📐 Skew angle detected: {angle:.2f}°. Applied rotation.")

                # Convert the rotated image (RGB float 0-255) to a NumPy array (BGR uint8)
                # Skimage loads as RGB, CV2 uses BGR. We need to convert the color order and type.
                deskewed_image_np = rotated.astype(np.uint8)

                # If it's a color image (3 channels), convert RGB to BGR for OpenCV compatibility
                if deskewed_image_np.ndim == 3 and deskewed_image_np.shape[2] == 3:
                    deskewed_image_np = cv2.cvtColor(deskewed_image_np, cv2.COLOR_RGB2BGR)

            return deskewed_image_np

        except Exception as e:
            print(f"[ERROR] Deskewing page {original_name} failed: {e}. Returning original image.")
            # Fallback to loading the original image using CV2
            return cv2.imread(image_path)

    # ... (_plot_clustering_results_per_image, _process_image_parts_and_lines, line_predict_Count, count_char, stitch_line_segments, _sample_lines_and_predict methods are unchanged) ...

    def execution(self, folder_path):
        """
        Main function to run the pipeline: selection, DESKEWING, text extraction,
        and full pipeline execution on the deskewed text regions.
        """
        # --- STATE RESET & SETUP ---
        self.all_line_coordinates = []
        self.all_line_data = []
        self.image_text_extracted_np = []
        text_region_to_original_name = []

        self._setup_output_directory(folder_path)

        # 1. Take 3 random images
        self.selected_image_paths = self.image_selector(folder_path)
        if not self.selected_image_paths:
            return {"status": "No images processed."}

        # --- STEP 1: DESKEWING PHASE ---
        print("\n--- STEP 1: Full Page Deskewing Phase ---")

        deskewed_image_nps = []

        for image_path in self.selected_image_paths:
            original_name = os.path.basename(image_path)

            # 1a. Apply Deskewing
            deskewed_page_np = self.deskew_skimage_page(image_path, original_name)

            if deskewed_page_np is None:
                print(f"[SKIP] Deskewed page was None for {original_name}.")
                continue

            deskewed_image_nps.append((deskewed_page_np, original_name))

            # Save the deskewed page
            deskewed_filename = f"{os.path.splitext(original_name)[0]}_Deskewed_Page.jpg"
            save_path = os.path.join(self.output_folder, deskewed_filename)
            cv2.imwrite(save_path, deskewed_page_np)
            print(f"🖼️ Saved deskewed page to: {save_path}")

            # Also copy the original selected file for reference (if it wasn't the deskewed output)
            # shutil.copy(image_path, self.output_folder) # Already handled implicitly by saving the deskewed version

        # --- STEP 2: Text Extraction (YOLO) on Deskewed Pages ---
        print("\n--- STEP 2: Text Extraction on Deskewed Pages ---")

        for deskewed_page_np, original_name in deskewed_image_nps:

            # 2a. Extract text region from the deskewed image
            cropped_text_regions = self.text_extract_and_crop(deskewed_page_np)

            for i, cropped_region in enumerate(cropped_text_regions):
                region_id = f"R{i + 1}"

                # Store the cropped text region for analysis
                self.image_text_extracted_np.append(cropped_region)
                text_region_to_original_name.append(original_name)

                # Save the cropped text region (which is deskewed)
                text_region_filename = f"{os.path.splitext(original_name)[0]}_TextRegion_{region_id}.jpg"
                save_path = os.path.join(self.output_folder, text_region_filename)

                if cropped_region.dtype != np.uint8:
                    cropped_region = np.clip(cropped_region, 0, 255).astype(np.uint8)

                cv2.imwrite(save_path, cropped_region)
                print(f"🖼️ Saved cropped text region (deskewed) to: {save_path}")

        # --- STEP 3, 4, 5, 6: Full Analysis Pipeline ---
        print("\n--- STEP 3-6: Line and Character Analysis on Deskewed Text Regions ---")

        individual_image_line_counts = []

        # Loop through the deskewed text images for line segmentation
        for i, text_img_np in enumerate(self.image_text_extracted_np):
            original_name = text_region_to_original_name[i]

            # 3. Segment, apply YOLO line, collect coords/segments
            third_part_line_count, image_line_coords = \
                self._process_image_parts_and_lines(text_img_np, original_name)

            self.all_line_coordinates.extend(image_line_coords)

            if third_part_line_count > 0:
                individual_image_line_counts.append(third_part_line_count)

        # Group all line data by the original image name
        data_by_image = {}
        for data in self.all_line_data:
            img_name = data['original_img']
            if img_name not in data_by_image:
                data_by_image[img_name] = []
            data_by_image[img_name].append(data)

        # Plot clustering results for verification
        for img_name, img_data in data_by_image.items():
            # 4. DBSCAN Clustering and Plotting
            self._plot_clustering_results_per_image(img_data, img_name)

        # Calculate final line count
        final_line_count = int(np.mean(individual_image_line_counts)) if individual_image_line_counts else 0

        # Execute final prediction steps (Stitching, Lipi, Char Count)
        try:
            # 5. Select Largest Cluster, Stitching
            # 6. Lipi/Char Count
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._sample_lines_and_predict, self.all_line_coordinates)
                final_lipi_class, final_charac_count = future.result()

        except Exception as e:
            print(f"[ERROR] Final analysis failed in parallel execution: {e}")
            final_lipi_class = "ERROR"
            final_charac_count = 0

        # --- FINAL CONSOLIDATION ---
        print("\n--- FINAL PIPELINE RESULTS ---")
        print(f"1. Final Line Count: **{final_line_count}**")
        print(f"2. Final Lipi Class: **{final_lipi_class}**")
        print(f"3. Final Character Count: **{final_charac_count}**")

        return {
            "final_line_count": final_line_count,
            "final_lipi_class": final_lipi_class,
            "final_charac_count": final_charac_count,
            "processed_images": list(set(text_region_to_original_name))
        }


# Example Usage (Dummy setup for flow testing)
if __name__ == "__main__":
    test_folder = r"C:\Users\AnjaliC\OneDrive - Dharohar\Desktop\test\test"

    pipeline = Line_count()
    print("\nAttempting to run pipeline...")
    results = pipeline.execution(test_folder)
    print(results)

