import numpy as np
import torch
from pytorch3d.structures import Meshes

class HairModeler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def model_hair(self, image, hair_mask, face_model):
        """
        Create a 3D model of hair based on the segmented hair mask and face model
        
        Args:
            image: Input image
            hair_mask: Binary mask indicating hair pixels
            face_model: 3D face model
            
        Returns:
            3D hair model
        """
        print("Modeling 3D hair from segmentation mask...")
        
        # In a real application, this would use sophisticated hair modeling techniques
        # For simplicity, we'll create a basic representation
        
        # Extract face model vertices
        face_verts = face_model.verts_list()[0].cpu().numpy()
        
        # Create hair vertices by extending from the top of the head
        # This is a simplified approach; real hair modeling is much more complex
        top_verts = face_verts[face_verts[:, 1] > np.percentile(face_verts[:, 1], 90)]
        
        # Create hair strands from the top vertices
        hair_verts = []
        hair_faces = []
        
        for i, v in enumerate(top_verts):
            # Create a strand with 5 vertices
            strand_verts = [v.copy() + np.array([0, j*0.02, 0]) for j in range(5)]
            
            # Add random variation to make it look more natural
            for j in range(1, 5):
                strand_verts[j] += np.random.normal(0, 0.01, 3)
                
            # Add vertices to the list
            start_idx = len(hair_verts)
            hair_verts.extend(strand_verts)
            
            # Create faces connecting vertices
            for j in range(4):
                hair_faces.append([start_idx + j, start_idx + j + 1, start_idx + j])
        
        # Convert to PyTorch tensors
        hair_verts = torch.tensor(hair_verts, dtype=torch.float32).to(self.device)
        hair_faces = torch.tensor(hair_faces, dtype=torch.int64).to(self.device)
        
        # Create a PyTorch3D Meshes object
        hair_mesh = Meshes(
            verts=[hair_verts],
            faces=[hair_faces]
        )
        
        print(f"Created hair model with {len(hair_verts)} vertices and {len(hair_faces)} faces")
        
        return hair_mesh
        
    def transfer_hair(self, selfie_face_model, target_hair_model):
        """
        Transfer the target hairstyle to the selfie face model
        
        Args:
            selfie_face_model: 3D face model from the selfie
            target_hair_model: 3D hair model from the target image
            
        Returns:
            Combined 3D model with the selfie face and target hairstyle
        """
        print("Transferring hairstyle to selfie face model...")
        
        # Extract components
        face_verts = selfie_face_model.verts_list()[0]
        face_faces = selfie_face_model.faces_list()[0]
        
        hair_verts = target_hair_model.verts_list()[0]
        hair_faces = target_hair_model.faces_list()[0]
        
        # Align hair model to the face model
        # In a real implementation, this would involve sophisticated alignment techniques
        # For simplicity, we'll just adjust the position based on the face model's bounds
        
        face_center = face_verts.mean(dim=0)
        hair_center = hair_verts.mean(dim=0)
        
        # Translate hair to match face position
        hair_verts = hair_verts - hair_center + face_center
        
        # Adjust vertical position to sit on top of the head
        face_top = face_verts[:, 1].max()
        hair_bottom = hair_verts[:, 1].min()
        hair_verts[:, 1] = hair_verts[:, 1] - hair_bottom + face_top - 0.02  # Small offset
        
        # Combine face and hair vertices and faces
        # Adjust hair_faces indices to account for combined vertices
        num_face_verts = face_verts.shape[0]
        adjusted_hair_faces = hair_faces + num_face_verts
        
        combined_verts = torch.cat([face_verts, hair_verts], dim=0)
        combined_faces = torch.cat([face_faces, adjusted_hair_faces], dim=0)
        
        # Create combined mesh
        combined_mesh = Meshes(
            verts=[combined_verts],
            faces=[combined_faces]
        )
        
        print(f"Created combined model with {len(combined_verts)} vertices and {len(combined_faces)} faces")
        
        return combined_mesh