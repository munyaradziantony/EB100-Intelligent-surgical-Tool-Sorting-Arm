from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageDraw

def load_model(model_path: str):
    """
    Load a pre-trained YOLOv8 model.
    """
    model = YOLO(model_path)
    return model

def detect_objects(image_path: str, model, tray_width_cm: float, tray_height_cm: float, distance_from_tray: float):
    """
    Detect objects in the provided image and return details like class, bounding box, and centroid.
    
    Args:
    - image_path (str): Path to the image file.
    - model: The YOLOv8 model to use for detection.
    - tray_width_cm (float): The width of the tray in cm.
    - tray_height_cm (float): The height of the tray in cm.
    - distance_from_tray (float): The distance of the objects from the tray in cm.
    
    Returns:
    - List of dictionaries containing object details.
    """
    # Perform object detection
    results = model(image_path)

    # Get the first result (assuming there's only one image)
    result = results[0]

    # Get class names
    names = result.names

    # Initialize a list to store object information
    objects = []

    # Open the image for size information
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Calculate the center of the tray (assuming it's at the image center)
    tray_center_x = image_width // 2
    tray_center_y = image_height // 2

    # Iterate through each detection
    for box in result.boxes:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Calculate the centroid
        centroid_x = (x1 + x2) // 2
        centroid_y = (y1 + y2) // 2

        # Get the class ID and name
        class_id = int(box.cls[0])
        class_name = names[class_id]

        # Calculate centroid with respect to the tray center
        centroid_wrt_tray_x = centroid_x - tray_center_x
        centroid_wrt_tray_y = centroid_y - tray_center_y

        # Calculate bounding box dimensions in cm
        bbox_width_cm = (x2 - x1) * tray_width_cm / image_width
        bbox_height_cm = (y2 - y1) * tray_height_cm / image_height

        # Store object information in a dictionary
        object_info = {
            'name': class_name,
            'bounding_box_pixels': [(x1, y1), (x2, y2)],
            'bounding_box_cm': (bbox_width_cm, bbox_height_cm),
            'centroid': (centroid_x, centroid_y),
            'centroid_wrt_tray': (centroid_wrt_tray_x, centroid_wrt_tray_y),
            'height': distance_from_tray
        }
        objects.append(object_info)

    return objects

def draw_detections(image_path: str, objects):
    """
    Draw bounding boxes and centroids on the image.
    
    Args:
    - image_path (str): Path to the image file.
    - objects (list): List of object dictionaries containing bounding box and centroid info.
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    for obj in objects:
        x1, y1 = obj['bounding_box_pixels'][0]
        x2, y2 = obj['bounding_box_pixels'][1]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        centroid_x, centroid_y = obj['centroid']
        draw.ellipse((centroid_x - 5, centroid_y - 5, centroid_x + 5, centroid_y + 5), fill="red")

    plt.imshow(image)
    plt.show()
