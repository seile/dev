import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import trimesh
from pytorch3d.io import load_obj

class ThreeDViewer:
    def __init__(self):
        print("Initializing 3D viewer...")
        
    def load_and_display(self, model_path):
        """
        Load and display the 3D model with interactive controls
        
        Args:
            model_path: Path to the OBJ file
        """
        print(f"Loading 3D model from {model_path}")
        
        # Load the model
        try:
            mesh = trimesh.load(model_path)
            print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return
        
        # Create figure and 3D axes
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the mesh
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Create the initial plot
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                        triangles=faces, color='lightgray', alpha=0.8)
        
        # Set initial view
        ax.view_init(elev=0, azim=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Hairstyle Preview (use sliders to rotate)')
        
        # Add sliders for interactive rotation
        ax_elev = plt.axes([0.25, 0.05, 0.65, 0.03])
        ax_azim = plt.axes([0.25, 0.01, 0.65, 0.03])
        
        slider_elev = Slider(ax_elev, 'Elevation', -90, 90, valinit=0)
        slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=0)
        
        def update(val):
            elev = slider_elev.val
            azim = slider_azim.val
            ax.view_init(elev=elev, azim=azim)
            fig.canvas.draw_idle()
            
        slider_elev.on_changed(update)
        slider_azim.on_changed(update)
        
        plt.tight_layout()
        plt.show()