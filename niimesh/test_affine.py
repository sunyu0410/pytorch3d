# To confirm how the affine grid works

import torch
import torch.nn.functional as F

z = 6
img = torch.zeros((1,1,3,3,z)).float()
img[0,0,2,2,2] = 1

affine_matrix = torch.tensor([
    [1.0, 0, 0, 0.4],
    [0, 1, 0, 0.1],
    [0, 0, 1, 0.2]
]).float().unsqueeze(0)

grid = F.affine_grid(affine_matrix, (1,1,z,3,3), align_corners=True)

warped = F.grid_sample(img.permute(0,1,4,3,2), grid, align_corners=True).permute(0,1,4,3,2)

print(warped)

from warp_mesh import get_3d_com, cvt_aff_matrix
com = get_3d_com(warped[0])-1/2
print(com)


matrix = cvt_aff_matrix(affine_matrix, (3,3,z))
matmul_result = torch.matmul(matrix[0], torch.tensor([2.0, 2, 2, 1]))
print(f'matmul: {matmul_result}')

torch.testing.assert_close(com, matmul_result[:-1])
