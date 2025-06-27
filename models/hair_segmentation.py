import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2

class UNet(nn.Module):
    """U-Net architecture for hair segmentation"""
    def __init__(self):
        super(UNet, self).__init__()
        # Simplified U-Net implementation
        # In a real application, this would be a full U-Net with skip connections
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder (simplified)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)

class HairSegmentationModel:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().to(self.device)
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Warning: No pre-trained hair segmentation model found.")
            print("Using untrained model, which won't produce good results.")
            
        self.model.eval()
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def segment(self, image):
        """
        Segment hair in the given image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Hair mask (binary image where 1 represents hair pixels)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        h, w = image.shape[:2]
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Convert output to mask
        mask = output[0, 0].cpu().numpy()
        mask = cv2.resize(mask, (w, h))
        mask = (mask > 0.5).astype(np.uint8)
        
        return mask