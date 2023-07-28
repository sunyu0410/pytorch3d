from dataclasses import dataclass
import nibabel
from voxelfy import to_mesh, taubin_smoothing
from pytorch3d.io import save_obj
import pytorch3d.ops
import torch

@dataclass
class NiiMesh():
    nii_path: int
    value: int
    smooth: bool = True
    
    def __post_init__(self):
        self.nii = nibabel.load(self.nii_path)
        self.affine = torch.from_numpy(self.nii.affine)
        self.voxels = torch.from_numpy(self.nii.get_fdata()).unsqueeze(0)

    def to_mesh(self):
        """
        Convert the voxels to mesh"""
        self.mesh = to_mesh(self.voxels, self.value, affine=self.affine)

        if self.smooth:
            self.mesh = taubin_smoothing(self.mesh)

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
    
    def save_obj(self, path):
        save_obj(path, self.mesh.verts_list()[0], self.mesh.faces_list()[0])

    
nmesh = NiiMesh('../objs/seg-test.nii.gz', 3)
nmesh.to_mesh()
nmesh.save_obj('test_s.obj')
print('end')