import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # Optional, for further determinism

import random
import base64
from io import BytesIO
import numpy as np
import tensorflow as tf
import matplotlib
# Use the non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

# Set seeds for determinism
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
# If you previously disabled oneDNN, comment out the following line
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# -------------------------------------------------------------------------------
# Define custom loss and metric functions (matching your inference code)
# -------------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    # Example implementation using binary crossentropy.
    # Replace with the actual combined loss logic if needed.
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    # Example implementation of dice coefficient.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1)

@tf.keras.utils.register_keras_serializable()
def focal_tversky_loss(y_true, y_pred):
    smooth = 1.0
    alpha = 0.7
    beta = 0.3
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_pos = tf.reduce_sum(y_true_f * y_pred_f)
    false_neg = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    false_pos = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return 1 - tversky_index

# -------------------------------------------------------------------------------
# INFERENCE CODE (unchanged except top-1 result and additional mapping)
# -------------------------------------------------------------------------------

# Define the 7 classes and mapping from class name to integer label
class_names = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

# Mapping for additional details for each class
class_info = {
    'bkl': {'full_name': 'Benign Keratosis-like Lesions', 'type': 'Benign (Non-Cancerous) Lesion', 'recommendation': 'Monitor Changes Regularly.'},
    'nv': {'full_name': 'Melanocytic Nevi', 'type': 'Benign (Non-Cancerous) Lesion', 'recommendation': 'No Immediate Concern, Routine Check-Up Recommended.'},
    'df': {'full_name': 'Dermatofibroma', 'type': 'Benign (Non-Cancerous) Lesion', 'recommendation': 'No Immediate Concern, but Consult if Changes Occur.'},
    'mel': {'full_name': 'Melanoma', 'type': 'Malignant (Cancerous) Lesion', 'recommendation': 'Visit a Doctor Immediately.'},
    'vasc': {'full_name': 'Vascular Lesions', 'type': 'Benign (Non-Cancerous) Lesion', 'recommendation': 'Monitor Changes Regularly.'},
    'bcc': {'full_name': 'Basal Cell Carcinoma', 'type': 'Malignant (Cancerous) Lesion', 'recommendation': 'Visit a Doctor Immediately.'},
    'akiec': {'full_name': 'Actinic Keratoses', 'type': 'Malignant (Cancerous) Lesion', 'recommendation': 'Consult a Dermatologist for Evaluation.'}
}

# Update these paths to point to your correctly saved models.
model_path = "C:\\Users\\abhis\\OneDrive\\Desktop\\6CNN_Model.keras"
unet_model_path = "C:\\Users\\abhis\\OneDrive\\Desktop\\2UNET.keras"

print("Loading classification model from:", model_path)
model = tf.keras.models.load_model(model_path)
print("Loading segmentation model from:", unet_model_path)
unet_model = tf.keras.models.load_model(unet_model_path,
                                        custom_objects={'combined_loss': combined_loss,
                                                        'dice_coefficient': dice_coefficient,
                                                        'focal_tversky_loss': focal_tversky_loss})

###############################################################################
# 1) STABLE PREPROCESSING (NO RANDOM AUGMENTATION)
###############################################################################
def load_image_no_augmentation(image_path):
    image_data = tf.io.read_file(image_path)
    image_data = tf.image.decode_jpeg(image_data, channels=3)
    image_data = tf.image.resize(image_data, [256, 256])
    image_data = tf.image.convert_image_dtype(image_data, tf.float32)
    return image_data

###############################################################################
# 2) CLASSIFICATION INFERENCE (Top-1 Only, with Temperature Scaling)
###############################################################################
def temperature_scaled_softmax(probs, temperature=1.0):
    if temperature == 1.0:
        return probs
    logits = np.log(probs + 1e-20)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def classify_image(image_path, temperature=1.0):
    img_tensor = load_image_no_augmentation(image_path)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    predictions = model.predict(img_tensor)
    scaled_predictions = temperature_scaled_softmax(predictions, temperature=temperature)
    top_idx = np.argmax(scaled_predictions[0])
    top_class = class_names[top_idx]
    top_conf = scaled_predictions[0][top_idx]
    return top_class, top_conf

###############################################################################
# 3) SEGMENTATION INFERENCE (NO RANDOM AUGMENTATION)
###############################################################################
def segment_image(image_path):
    # Use the same preprocessing as training
    img_tensor = load_image_no_augmentation(image_path)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    mask_pred = unet_model.predict(img_tensor)
    # Apply thresholding to obtain a binary mask (flawless segmentation)
    mask = (mask_pred[0, :, :, 0] > 0.5).astype(np.uint8)
    return mask

###############################################################################
# (Optional) Function to overlay the segmentation mask on the original image.
# Uncomment if you prefer a blended visualization.
###############################################################################
# def overlay_mask_on_image(orig_img, mask):
#     orig = np.array(orig_img)
#     # Create a red color mask where the lesion is (assumed 0 in mask indicates lesion, so invert if needed)
#     color_mask = np.zeros_like(orig)
#     # For example, set the red channel to the mask (you can adjust alpha as needed)
#     color_mask[:, :, 0] = mask
#     overlay = (0.7 * orig + 0.3 * color_mask).astype(np.uint8)
#     return overlay

###############################################################################
# 4) HIGH-LEVEL FUNCTION: CLASSIFY & SEGMENT + RETURN INFO
###############################################################################
def classify_and_segment_image(image_path, temperature=1.0):
    top_class, top_conf = classify_image(image_path, temperature=temperature)
    mask = segment_image(image_path)
    # Get additional details for the predicted class
    info = class_info.get(top_class, {})
    full_name = info.get('full_name', 'N/A')
    lesion_type = info.get('type', 'N/A')
    recommendation = info.get('recommendation', 'N/A')
    # Load the original image for display using the same preprocessing (resized to 256x256)
    orig_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    # (Optional) Overlay the mask on the original image:
    # overlay = overlay_mask_on_image(orig_img, mask)
    return top_class, top_conf, full_name, lesion_type, recommendation, mask, orig_img
    
# -------------------------------------------------------------------------------
# FLASK APP SETUP
# -------------------------------------------------------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    # On GET, display the full homepage with project details and the upload section.
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            top_class, top_conf, full_name, lesion_type, recommendation, mask, orig_img = classify_and_segment_image(file_path, temperature=1.0)

            # Convert images (segmentation mask and original image) to base64 strings for HTML embedding.
            # Segmentation mask
            fig, ax = plt.subplots()
            ax.imshow(mask, cmap='gray')
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            segmentation_img = base64.b64encode(buf.getvalue()).decode("utf-8")
            plt.close(fig)
            # Original image
            fig2, ax2 = plt.subplots()
            ax2.imshow(orig_img)
            ax2.axis('off')
            buf2 = BytesIO()
            plt.savefig(buf2, format="png", bbox_inches='tight', pad_inches=0)
            buf2.seek(0)
            orig_img_str = base64.b64encode(buf2.getvalue()).decode("utf-8")
            plt.close(fig2)

            return render_template('results.html',
                                   top_class=top_class,
                                   top_conf=f"{top_conf*100:.2f}",
                                   full_name=full_name,
                                   lesion_type=lesion_type,
                                   recommendation=recommendation,
                                   segmentation_img=segmentation_img,
                                   orig_img=orig_img_str)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
