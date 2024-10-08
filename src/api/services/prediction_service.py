from api.models.model_loader import ModelLoader
from api.utils.preprocess import Preprocess

class PredictionService:
    def __init__(self):
        self.model_loader = ModelLoader()

    def make_prediction(self, input_data):
        """Make prediction using the trained model."""
        model = self.model_loader.get_model()

        # Preprocess the input data
        processed_data = Preprocess.preprocess_input(input_data)

        # Make prediction
        prediction = model.predict(processed_data)
        
        # Post-process the result if needed
        result = {"prediction": prediction[0]}
        return result
