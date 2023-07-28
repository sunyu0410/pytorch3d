from dataclasses import dataclass
import nibabel
from voxelfy import to_mesh, taubin_smoothing
from pytorch3d.io import save_obj
import pytorch3d.ops
import torch
import torch.nn.functional as F
import warp_mesh

@dataclass
class NiiMesh():
    nii_path: int
    value: int
    smooth: bool = True
    
    def __post_init__(self):
        self.nii = nibabel.load(self.nii_path)
        self.affine = torch.from_numpy(self.nii.affine)
        self.numpy = self.nii.get_fdata()
        self.voxels = torch.from_numpy(self.numpy).unsqueeze(0)
        self.shape = self.numpy.shape

    def to_mesh(self):
        """
        Convert the voxels to mesh"""
        self.mesh = to_mesh(self.voxels, self.value, affine=self.affine)

        if self.smooth:
            self.mesh = taubin_smoothing(self.mesh)

    def sample_points(self, n):
        """Sample n pionts from the mesh"""
        # TODO - there's a dtype mismatch: float vs double (new version doesn't have this issue)
        self.points = pytorch3d.ops.sample_points_from_meshes(self.mesh, n)
    
    @staticmethod
    def warp_point(grid, coord, shape):
        pt_img = warp_mesh.put_3d_com(coord, shape).view((1,1)+shape)
        pt_img_warp = F.grid_sample(pt_img.permute((0,1,4,3,2)), grid, align_corners=True).permute((0,1,4,3,2))
        return warp_mesh.get_3d_com(pt_img_warp.view((1,)+shape))
        
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


if __name__ == "__main__":
    affine_matrix = torch.tensor(
        [[
          [1, 0, 0, 0.2],
          [0, 1, 0, 0],
          [0, 0, 1, 0]
        ]],
        dtype=torch.float,
        requires_grad=True
    )

    grid = F.affine_grid(affine_matrix, (1,1,100,100,3), align_corners=True)
    coord = (2,2,2)
    shape = (3, 100, 100)

    nmesh = NiiMesh('../objs/seg-test.nii.gz', 3)
    com = nmesh.warp_point(grid, coord, shape)

    print(f'COM: {com}')

    # nmesh.to_mesh()
    # nmesh.save_obj('test_s.obj')