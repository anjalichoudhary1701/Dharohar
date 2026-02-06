import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
from PIL import Image

MODEL_PATH_Lipi = r"C:\Users\AnjaliC\OneDrive - Dharohar\Desktop\test\Devanagri_lipi.pth" # MUST exist
MODEL_PATH_Cover_info = r""
# Standard normalization values
GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



class SimpleCNN_Lipi(nn.Module):


    CLASS_NAMES = ["Devanagari lipi", "Different lipi"]

    def __init__(self, num_classes, img_w, img_h):
        self.img_w = 240
        self.img_h = 60
        self.num_classes = 2
        super(SimpleCNN_Lipi, self).__init__()
        # ... (Same convolutional layers as before)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate Flattened Size:
        final_h = img_h // 8
        final_w = img_w // 8
        self.flatten_size = 128 * final_h * final_w

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes) # Uses NUM_CLASSES

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, self.flatten_size)
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleCNN_CoverInfo(nn.Module):


    CLASS_NAMES = ["Cover Page", "Information Sheet" , "Normal Page"]

    def __init__(self, num_classes, img_w, img_h):
        self.img_w = 240
        self.img_h = 240
        self.num_classes = 3
        super(SimpleCNN_CoverInfo, self).__init__()
        # ... (Same convolutional layers as before)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate Flattened Size:
        final_h = img_h // 8
        final_w = img_w // 8
        self.flatten_size = 128 * final_h * final_w

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes) # Uses NUM_CLASSES

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, self.flatten_size)
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def predict_single_image_by_path(model, image_path, img_h, img_w, device, class_names):

    single_image_transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD),
    ])

    # 1. Load Image from Path
    # This is the original logic that loads the image from disk.
    image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_np is None:
        return f"Error: Could not load image from {image_path}"

    # 2. Apply Transforms
    image = Image.fromarray(image_np)
    image_tensor = single_image_transform(image)

    # 3. Prepare for Inference
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 4. Run Inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    # 5. Get Prediction
    _, predicted_index = torch.max(output, 1)
    predicted_class = class_names[predicted_index.item()]

    return predicted_class


# -----------------------------------------------------------------------
# ADD: New function to handle the NumPy array input (used by demo.py)
# -----------------------------------------------------------------------
# NOTE: This is the function you need to import and use in demo.py now.
def predict_single_image_from_np(model, image_np, img_h, img_w, device, class_names):

    single_image_transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD),
    ])

    # 1. Image is already loaded as a NumPy array (image_np)
    if image_np is None or image_np.size == 0:
        return f"Error: Empty image array received"

    # Convert BGR/RGB to Grayscale (as the model expects 1 channel)
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # 2. Apply Transforms
    # IMPORTANT: Ensure the NumPy array dtype is correct before converting to PIL
    image = Image.fromarray(image_np.astype(np.uint8))
    image_tensor = single_image_transform(image)

    # 3. Prepare for Inference
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 4. Run Inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    # 5. Get Prediction
    _, predicted_index = torch.max(output, 1)
    predicted_class = class_names[predicted_index.item()]

    return predicted_class


# --- 3. Execution ---

if __name__ == '__main__':
    lipi_w = 240
    lipi_h =60
    num_classes = 2
      # --- Load Model Weights ---
    model = SimpleCNN_Lipi(num_classes, lipi_w, lipi_h).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH_Lipi, map_location=device))
        print(f"✅ Successfully loaded model weights from {MODEL_PATH_Lipi}")

        # V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V
        # -----------------------------------------------------------------------
        # STEP 1: SPECIFY YOUR NEW IMAGE PATH HERE
        # -----------------------------------------------------------------------

        # You still need to specify the path for the single image you want to test.
        MY_NEW_TEST_IMAGE_PATH = r"\\10.10.90.146\Dharohar\Corpus for ML training\lipi\lines_data\Devanagari\line_cluster_0_0fa58362.png"

        # -----------------------------------------------------------------------
        # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        if os.path.exists(MY_NEW_TEST_IMAGE_PATH):

            classes_name_to_use = SimpleCNN_Lipi.CLASS_NAMES

            predicted_class = predict_single_image(
                model,
                MY_NEW_TEST_IMAGE_PATH,
                classes_name_to_use,
                device,
                img_h = lipi_h,
                img_w = lipi_w
            )
            print(f"Predicted Class: {predicted_class}")


        else:
            print(f"\n⚠️ Warning: Custom image path not found: {MY_NEW_TEST_IMAGE_PATH}")
            print("To test a custom image, please update the MY_NEW_TEST_IMAGE_PATH variable.")


    except FileNotFoundError:
        print(f"\n❌ Error: Model file '{MODEL_PATH_Lipi}' not found.")
        print("Please ensure the model weights file is in the same directory or the path is correct.")
    except Exception as e:
        print(f"\nAn error occurred during model loading or testing: {e}")