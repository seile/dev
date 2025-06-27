import os
import numpy as np
import torch
import cv2
from models.hair_segmentation import HairSegmentationModel
from models.face_reconstruction import FaceReconstructor
from models.hair_modeling import HairModeler
from utils.renderer import Renderer
from gui.viewer import ThreeDViewer

class HairstyleTransferApp:
    def __init__(self):
        # Initialize models
        self.hair_segmenter = HairSegmentationModel()
        self.face_reconstructor = FaceReconstructor()
        self.hair_modeler = HairModeler()
        self.renderer = Renderer()
        self.viewer = ThreeDViewer()
        
        print("Hairstyle Transfer Application initialized successfully")
        
    def process_images(self, selfie_path, target_hair_path):
        """
        Process the selfie and target hair images to create a 3D model with transferred hairstyle
        
        Args:
            selfie_path: Path to the user's selfie
            target_hair_path: Path to the image with the desired hairstyle
        
        Returns:
            3D model path that can be viewed in the viewer
        """
        print(f"Processing selfie image: {selfie_path}")
        print(f"Processing target hair image: {target_hair_path}")
        
        # Load images
        selfie_img = cv2.imread(selfie_path)
        target_img = cv2.imread(target_hair_path)
        
        if selfie_img is None or target_img is None:
            raise ValueError("Failed to load one or both images")
        
        # Step 1: Segment hair in both images
        selfie_hair_mask = self.hair_segmenter.segment(selfie_img)
        target_hair_mask = self.hair_segmenter.segment(target_img)
        
        # Step 2: Reconstruct 3D face models
        selfie_3d_model = self.face_reconstructor.reconstruct(selfie_img)
        target_3d_model = self.face_reconstructor.reconstruct(target_img)
        
        # Step 3: Extract and model the target hairstyle in 3D
        target_hair_3d = self.hair_modeler.model_hair(target_img, target_hair_mask, target_3d_model)
        
        # Step 4: Transfer the target hairstyle to the selfie 3D model
        combined_3d_model = self.hair_modeler.transfer_hair(
            selfie_3d_model, 
            target_hair_3d
        )
        
        # Step 5: Render the result
        output_path = os.path.join("output", "combined_model.obj")
        self.renderer.save_model(combined_3d_model, output_path)
        
        print(f"Processing complete. 3D model saved to {output_path}")
        return output_path
    
    def display_result(self, model_path):
        """Launch the 3D viewer to display and interact with the result"""
        self.viewer.load_and_display(model_path)

def main():
    app = HairstyleTransferApp()
    
    # In a real application, these would come from user input
    selfie_path = input("Enter path to your selfie: ")
    target_hair_path = input("Enter path to target hairstyle image: ")
    
    try:
        model_path = app.process_images(selfie_path, target_hair_path)
        app.display_result(model_path)
    except Exception as e:
        print(f"Error processing images: {e}")

if __name__ == "__main__":
    main()