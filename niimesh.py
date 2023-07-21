from dataclasses import dataclass
import nibabel

@dataclass
class NiiMesh():
    nii_path: int
    
    def __post_init__(self):
        self.nii = nibabel.load(self.nii_path)
        self.affine_matrix = self.nii.affine

    def meshify(self):
        """
        Convert the voxels to mesh"""
        self.mesh = None

    def sample_points(self, n):
        """Sample n pionts from the mesh"""
        if not self.mesh:
            self.meshify()

    def apply_deformation(self, grid):
        """Apply a flow matrix"""
        pass

    def get_verts_offset(self, grid):
        """Calculates the verts difference, which can be used in
        Meshes.offset_verts()
        """
        pass
    
    def save_obj(self):
        pass

    
nmesh = NiiMesh('objs/seg-test.nii.gz')
print('end')