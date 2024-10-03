from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLOv8n model architecture without pre-trained weights
    model = YOLO('yolov8n.yaml')  # Load the model architecture

    # Train the model from scratch using the provided YAML file
    model.train(
        data='G:/projects/code/data.yaml',  # Path to your dataset YAML file
        device='cuda',                      # Use 'cuda' for GPU or 'cpu' for CPU
        epochs=100,                         # Number of training epochs
        imgsz=640,                          # Image size (640x640 pixels)
        batch=16,                           # Batch size (adjust based on your system's memory)
        name='yolov8n_person_scratch',      # Experiment name
        workers=16,                         # Set the number of workers to 16
        verbose=True,                       # Display detailed training logs
        patience=100,                       # Early stopping after 100 epochs with no improvement
        save=True,                          # Enable saving model checkpoints after each epoch
        exist_ok=True                       # Prevent overwriting existing models by saving to a unique name
    )
