import pickle
from api.config import Config

class ModelLoader:
    def __init__(self):
        self.model = None

    def load_model(self):
        """Load the trained machine learning model from a file."""
        try:
            with open(Config.MODEL_PATH, 'rb') as file:
                self.model = pickle.load(file)
        except FileNotFoundError:
            raise Exception(f"Model file not found at {Config.MODEL_PATH}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def get_model(self):
        if self.model is None:
            self.load_model()
        return self.model
