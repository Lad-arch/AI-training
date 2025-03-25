import os
from ultralytics import YOLO

def train_yolov8_nano(config_yaml: str, output_dir: str, epochs: int = 50, batch: int = 16):
    """
    Trains a YOLOv8 Nano model using a specified YAML configuration file.

    Args:
        config_yaml (str): Path to the dataset configuration file in YAML format.
        output_dir (str): Directory where the trained model and logs will be saved.
        epochs (int): Number of training epochs. Default is 50.
        batch (int): Batch size for training. Default is 16.

    Returns:
        str: Path to the saved model file (.pt).
    """
    # Check if the YAML configuration file exists
    if not os.path.exists(config_yaml):
        raise FileNotFoundError(f"Configuration file '{config_yaml}' not found.")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the YOLOv8 Nano model
    model = YOLO('yolov8n.pt')  # Pretrained YOLOv8 Nano model

    # Train the model
    model.train(
        data=config_yaml,
        epochs=epochs,
        batch=batch,
        project=output_dir,
        name='yolov8_nano_training',
        exist_ok=True  # Allow overwriting existing runs
    )

    # Get the path to the trained model file
    trained_model_path = os.path.join(output_dir, 'yolov8_nano_training', 'weights', 'best.pt')

    if not os.path.exists(trained_model_path):
        raise FileNotFoundError(f"Trained model file not found at '{trained_model_path}'.")

    print(f"Training complete. Model saved at: {trained_model_path}")
    return trained_model_path

if __name__ == "__main__":
    # Example usage
    CONFIG_YAML = "C:\\Users\\htcam\\PycharmProjects\\AItraning\\tvnn.yaml"  # Replace with the path to your YAML file
    OUTPUT_DIR = "C:\\Users\\htcam\\PycharmProjects\\AItraning\\"  # Replace with the directory to save trained model
    EPOCHS = 50
    BATCH = 16

    trained_model = train_yolov8_nano(
        config_yaml=CONFIG_YAML,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        batch=BATCH
    )
    print(f"Model successfully trained and saved at: {trained_model}")