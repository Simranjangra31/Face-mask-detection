import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMG_SIZE = 224
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations")
OUTPUT_FILE = os.path.join(DATA_DIR, "dataset.npz")

def parse_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        
        # Get image dimensions to normalize boxes
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in CLASSES:
                continue
            label_id = CLASSES.index(label)

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            
            # Normalize coordinates (0-1)
            boxes.append([xmin/width, ymin/height, xmax/width, ymax/height])
            labels.append(label_id)

        return boxes, labels
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return [], []

def load_and_preprocess():
    images = []
    bboxes = []
    class_labels = []
    
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images. processing...")

    for file in tqdm(image_files):
        img_path = os.path.join(IMAGE_DIR, file)
        xml_file = os.path.splitext(file)[0] + ".xml"
        xml_path = os.path.join(ANNOTATION_DIR, xml_file)

        if not os.path.exists(xml_path):
            continue

        # Parse Annotations
        boxes, lbls = parse_xml(xml_path)
        if not boxes:
            continue
            
        # Load Image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized / 255.0

        # Finding largest box
        best_box_idx = 0
        max_area = 0
        for i, box in enumerate(boxes):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > max_area:
                max_area = area
                best_box_idx = i
        
        target_box = boxes[best_box_idx]
        target_label = lbls[best_box_idx]

        images.append(img_normalized)
        bboxes.append(target_box)
        class_labels.append(target_label)

    return np.array(images), np.array(bboxes), np.array(class_labels)

if __name__ == "__main__":
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(ANNOTATION_DIR):
        print("Data directories not found!")
        exit()

    X, y_box, y_class = load_and_preprocess() # X: Images, y_box: Bounding Boxes, y_class: Labels
    
    # One-hot encode labels (Manual implementation to avoid TF dep)
    y_class_one_hot = np.zeros((y_class.size, y_class.max() + 1))
    y_class_one_hot[np.arange(y_class.size), y_class] = 1
    y_class = y_class_one_hot

    print(f"Processed {len(X)} samples.")

    # Split: Train (70%), Val (15%), Test (15%)
    X_train, X_temp, yb_train, yb_temp, yl_train, yl_temp = train_test_split(
        X, y_box, y_class, test_size=0.3, random_state=42
    )

    X_val, X_test, yb_val, yb_test, yl_val, yl_test = train_test_split(
        X_temp, yb_temp, yl_temp, test_size=0.5, random_state=42
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    np.savez(OUTPUT_FILE,
             X_train=X_train, X_val=X_val, X_test=X_test,
             yb_train=yb_train, yb_val=yb_val, yb_test=yb_test,
             yl_train=yl_train, yl_val=yl_val, yl_test=yl_test)
    
    print(f"Dataset saved to {OUTPUT_FILE}")

