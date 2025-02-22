
from src.classes import ImagePLATO
from src.config import PREFIXES, OBSID_LISTS

def preprocess_images(model, image_type):
    print(f"Starting preprocessing for {model} - {image_type}")
    processor = ImagePLATO(model, image_type)
    processor.process_obsids()
    print(f"Preprocessing completed for {model} - {image_type}")

def train_cnn():
    # Add code to train the CNN here
    print("Training the CNN...")
    # Example: model.fit(X_train, y_train, epochs=10, batch_size=32)
    print("CNN training completed.")

if __name__ == "__main__":
    # Preprocess images
    model = "FM16"
    image_type = "Plateaux"
    preprocess_images(model, image_type)

    # Continue with CNN training
    train_cnn()