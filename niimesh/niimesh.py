from dataclasses import dataclass
import nibabel
from voxelfy import to_mesh, taubin_smoothing, column_stack
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

        self.affine = torch.from_numpy(self.nii.affine).float()
        self.affine = (self.affine.T*torch.tensor([-1,-1,1,1])).T # RAS to LPS
        
        self.numpy = self.nii.get_fdata()
        self.voxels = torch.from_numpy(self.numpy).unsqueeze(0).float()
        self.shape = self.numpy.shape

        # Two sets of meshes
        self.mesh_affine = taubin_smoothing(to_mesh(self.voxels, self.value, affine=self.affine)) # For validation
        self.mesh = taubin_smoothing(to_mesh(self.voxels, self.value, affine=None)) # For use


    def sample_points(self, n):
        """Sample n pionts from the mesh"""
        # TODO - there's a dtype mismatch: float vs double (new version doesn't have this issue)
        self.points = pytorch3d.ops.sample_points_from_meshes(self.mesh, n)
    
    @staticmethod
    def warp_point(grid, coord, shape):
        # TODO - the order in shape should be checked at the end
        pt_img = warp_mesh.put_3d_com(coord, shape).view((1,1)+shape)
        pt_img_warp = F.grid_sample(pt_img.permute((0,1,4,3,2)), grid, align_corners=True).permute((0,1,4,3,2))
        return warp_mesh.get_3d_com(pt_img_warp.view((1,)+shape))
        
    def apply_grid_to_points(self, grid):
        """Apply a flow matrix"""
        # self.warp_point(grid, )
        self.points = torch.stack([nmesh.warp_point(grid, p, nmesh.shape) for p in nmesh.points[0]]).unsqueeze(0)

    def apply_affine(self):
        self.points = [(nmesh.affine @ column_stack([self.points[0], torch.ones(self.points[0].size(0),1)]).T).T[:,:3]]
    
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
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0]
        ]],
        dtype=torch.float,
        requires_grad=True
    )

    nmesh = NiiMesh('../objs/seg-test.nii.gz', 3)
    
    shape = nmesh.shape
    grid = F.affine_grid(affine_matrix, (1,1,*reversed(shape)), align_corners=True)
    coord = (2,2,2)

    com = nmesh.warp_point(grid, coord, shape)

    print(f'COM: {com}')

    nmesh.sample_points(10)
    nmesh.apply_grid_to_points(grid)
    nmesh.apply_affine()

    print(nmesh.points)

    nmesh.save_obj('test_s.obj')
    save_obj('test_s_affine.obj', nmesh.mesh_affine.verts_list()[0], nmesh.mesh_affine.faces_list()[0])