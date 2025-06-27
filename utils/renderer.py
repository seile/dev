import torch
import numpy as np
from pytorch3d.io import save_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)

class Renderer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.renderer = self._setup_renderer()
        
    def _setup_renderer(self):
        """Set up the PyTorch3D renderer"""
        # Initialize camera parameters
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        
        # Define rasterization settings
        raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=0.0, 
            faces_per_pixel=1,
        )
        
        # Define lights
        lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, 3.0]],
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.7, 0.7, 0.7),),
            specular_color=((0.3, 0.3, 0.3),)
        )
        
        # Create renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )
        
        return renderer
    
    def render_image(self, mesh, azimuth=0):
        """
        Render an image of the mesh from the specified angle
        
        Args:
            mesh: PyTorch3D Meshes object
            azimuth: Viewing angle in degrees
            
        Returns:
            Rendered image as a numpy array
        """
        # Set up camera for the specified angle
        R, T = look_at_view_transform(2.7, 15, azimuth)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        
        # Add simple vertex colors if not present
        if not mesh.textures:
            verts = mesh.verts_list()[0]
            vertex_colors = torch.ones_like(verts) * torch.tensor([0.7, 0.7, 0.7])
            mesh.textures = TexturesVertex(verts_features=[vertex_colors])
        
        # Render
        images = self.renderer(mesh, cameras=cameras)
        
        # Convert to numpy
        image = images[0, ..., :3].cpu().numpy()
        
        return image
    
    def save_model(self, mesh, output_path):
        """Save the 3D model to a file"""
        verts = mesh.verts_list()[0].cpu()
        faces = mesh.faces_list()[0].cpu()
        
        save_obj(output_path, verts, faces)
        print(f"Saved 3D model to {output_path}")
        
        return output_path