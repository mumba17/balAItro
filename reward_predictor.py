from cnn_classifier import CNN
import torch
from torchvision import transforms
from PIL import Image

class RewardPredictor:
    def __init__(self, model_path, num_classes=7, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = CNN(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((160, 90)),
            transforms.ToTensor(),
        ])

    def predict(self, image):
        """
        Predict the class of a single image.
        
        :param image: PIL Image or numpy array
        :return: Predicted class (1-6)
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            pred = output.argmax(dim=1, keepdim=True).item()
        
        return pred