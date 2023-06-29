import torch

from pytorch3d import ops
from pytorch3d.io import save_obj

voxels = torch.zeros(1,3,5,5).float()
voxels[0,-1,0,0] = 1
voxels[0,-1,1:3,1:3] = 1
voxels[0,-1,3:,3:] = 1


# mesh = ops.cubify(voxels, 0.9)

# save_obj('objs/cubes2.obj', mesh.verts_list()[0], mesh.faces_list()[0])

# NIfTI example

import nibabel

cont = nibabel.load('objs/seg-test.nii.gz').get_fdata()
cont_mesh = ops.cubify(torch.from_numpy(cont==3).unsqueeze(0), 0.9)
save_obj('objs/seg-id-3.obj', cont_mesh.verts_list()[0], cont_mesh.faces_list()[0])

# Smoothing
cont_mesh_m = ops.taubin_smoothing(cont_mesh)
save_obj('objs/seg-id-3-s.obj', cont_mesh_m.verts_list()[0], cont_mesh_m.faces_list()[0])

# Localiser
# cont = nibabel.load('objs/seg-localiser.nii.gz').get_fdata()
# cont_mesh = ops.cubify(torch.from_numpy(cont!=0).unsqueeze(0), 0.9)
# save_obj('objs/seg-localiser.obj', cont_mesh.verts_list()[0], cont_mesh.faces_list()[0])
