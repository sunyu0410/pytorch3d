import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes, utils as struct_utils
from typing import Union, Tuple, Sequence
from pytorch3d.ops.cubify import ravel_index, unravel_index

# This is not in the eralier version, copied from version 0.7.4
def meshgrid_ij(
    *A: Union[torch.Tensor, Sequence[torch.Tensor]]
) -> Tuple[torch.Tensor, ...]:  # pragma: no cover
    """
    Like torch.meshgrid was before PyTorch 1.10.0, i.e. with indexing set to ij
    """
    if (
        # pyre-fixme[16]: Callable `meshgrid` has no attribute `__kwdefaults__`.
        torch.meshgrid.__kwdefaults__ is not None
        and "indexing" in torch.meshgrid.__kwdefaults__
    ):
        # PyTorch >= 1.10.0
        # pyre-fixme[6]: For 1st param expected `Union[List[Tensor], Tensor]` but
        #  got `Union[Sequence[Tensor], Tensor]`.
        return torch.meshgrid(*A, indexing="ij")
    # pyre-fixme[6]: For 1st param expected `Union[List[Tensor], Tensor]` but got
    #  `Union[Sequence[Tensor], Tensor]`.
    return torch.meshgrid(*A)

def norm_laplacian(
    verts: torch.Tensor, edges: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Norm laplacian computes a variant of the laplacian matrix which weights each
    affinity with the normalized distance of the neighboring nodes.
    More concretely,
    L[i, j] = 1. / wij where wij = ||vi - vj|| if (vi, vj) are neighboring nodes

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    edge_verts = verts[edges]  # (E, 2, 3)
    v0, v1 = edge_verts[:, 0], edge_verts[:, 1]

    # Side lengths of each edge, of shape (E,)
    w01 = 1.0 / ((v0 - v1).norm(dim=1) + eps)

    # Construct a sparse matrix by basically doing:
    # L[v0, v1] = w01
    # L[v1, v0] = w01
    e01 = edges.t()  # (2, E)

    V = verts.shape[0]
    L = torch.sparse.FloatTensor(e01, w01, (V, V))
    L = L + L.t()

    return L

def taubin_smoothing(
    meshes: Meshes, lambd: float = 0.53, mu: float = -0.53, num_iter: int = 10
) -> Meshes:
    """
    Taubin smoothing [1] is an iterative smoothing operator for meshes.
    At each iteration
        verts := (1 - λ) * verts + λ * L * verts
        verts := (1 - μ) * verts + μ * L * verts

    This function returns a new mesh with smoothed vertices.
    Args:
        meshes: Meshes input to be smoothed
        lambd, mu: float parameters for Taubin smoothing,
            lambd > 0, mu < 0
        num_iter: number of iterations to execute smoothing
    Returns:
        mesh: Smoothed input Meshes

    [1] Curve and Surface Smoothing without Shrinkage,
        Gabriel Taubin, ICCV 1997
    """
    verts = meshes.verts_packed()  # V x 3
    edges = meshes.edges_packed()  # E x 3

    for _ in range(num_iter):
        L = norm_laplacian(verts, edges)
        total_weight = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
        verts = (1 - lambd) * verts + lambd * torch.mm(L, verts) / total_weight

        L = norm_laplacian(verts, edges)
        total_weight = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
        verts = (1 - mu) * verts + mu * torch.mm(L, verts) / total_weight

    verts_list = struct_utils.packed_to_list(
        verts, meshes.num_verts_per_mesh().tolist()
    )
    mesh = Meshes(verts=list(verts_list), faces=meshes.faces_list())
    return mesh

def column_stack(tensors):
    hs = [t.size(0) for t in tensors]
    ws = [t.size(1) for t in tensors]
    assert len(set(hs)) == 1
    result = torch.zeros((hs[0], sum(ws)))
    col = 0
    for t in tensors:
        result[:, col:(col+t.size(1))] = t
        col += t.size(1)
    return result

@torch.no_grad()
def to_mesh(voxels, value, device=None, align: str = "topleft", norm:bool=False, affine=None) -> Meshes:
    r"""
    #* Create a mesh from a binary mask

    #* Modified from cubify()
    #* The main difference is that cubify will automatically normalise the dimensions
    #* This function can keep the dimensions absolute
    #* value: which discreate value in the contour to convert
    #* affine: the affine matrix (in RAS)


    The alignment between the vertices of the cubified mesh and the voxel locations (or pixels)
    is defined by the choice of `align`. We support three modes, as shown below for a 2x2 grid:

                X---X----         X-------X        ---------
                |   |   |         |   |   |        | X | X |
                X---X----         ---------        ---------
                |   |   |         |   |   |        | X | X |
                ---------         X-------X        ---------

                 topleft           corner            center

    In the figure, X denote the grid locations and the squares represent the added cuboids.
    When `align="topleft"`, then the top left corner of each cuboid corresponds to the
    pixel coordinate of the input grid.
    When `align="corner"`, then the corners of the output mesh span the whole grid.
    When `align="center"`, then the grid locations form the center of the cuboids.
    """

    if device is None:
        device = voxels.device

    if align not in ["topleft", "corner", "center"]:
        raise ValueError("Align mode must be one of (topleft, corner, center).")

    if len(voxels) == 0:
        return Meshes(verts=[], faces=[])
    


    N, D, H, W = voxels.size()

    assert N == 1 # takes one contour at a time

    # vertices corresponding to a unit cube: 8x3
    cube_verts = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.int64,
        device=device,
    )

    # faces corresponding to a unit cube: 12x3
    cube_faces = torch.tensor(
        [
            [0, 1, 2],
            [1, 3, 2],  # left face: 0, 1
            [2, 3, 6],
            [3, 7, 6],  # bottom face: 2, 3
            [0, 2, 6],
            [0, 6, 4],  # front face: 4, 5
            [0, 5, 1],
            [0, 4, 5],  # up face: 6, 7
            [6, 7, 5],
            [6, 5, 4],  # right face: 8, 9
            [1, 7, 3],
            [1, 5, 7],  # back face: 10, 11
        ],
        dtype=torch.int64,
        device=device,
    )

    wx = torch.tensor([0.5, 0.5], device=device).view(1, 1, 1, 1, 2)
    wy = torch.tensor([0.5, 0.5], device=device).view(1, 1, 1, 2, 1)
    wz = torch.tensor([0.5, 0.5], device=device).view(1, 1, 2, 1, 1)

    voxelv = voxels.eq(value).float()

    # N x 1 x D x H x W
    voxelv = voxelv.view(N, 1, D, H, W)

    # N x 1 x (D-1) x (H-1) x (W-1)
    voxelv_x = F.conv3d(voxelv, wx).gt(0.5).float()
    voxelv_y = F.conv3d(voxelv, wy).gt(0.5).float()
    voxelv_z = F.conv3d(voxelv, wz).gt(0.5).float()

    # 12 x N x 1 x D x H x W
    faces_idx = torch.ones((cube_faces.size(0), N, 1, D, H, W), device=device)

    # add left face
    faces_idx[0, :, :, :, :, 1:] = 1 - voxelv_x
    faces_idx[1, :, :, :, :, 1:] = 1 - voxelv_x
    # add bottom face
    faces_idx[2, :, :, :, :-1, :] = 1 - voxelv_y
    faces_idx[3, :, :, :, :-1, :] = 1 - voxelv_y
    # add front face
    faces_idx[4, :, :, 1:, :, :] = 1 - voxelv_z
    faces_idx[5, :, :, 1:, :, :] = 1 - voxelv_z
    # add up face
    faces_idx[6, :, :, :, 1:, :] = 1 - voxelv_y
    faces_idx[7, :, :, :, 1:, :] = 1 - voxelv_y
    # add right face
    faces_idx[8, :, :, :, :, :-1] = 1 - voxelv_x
    faces_idx[9, :, :, :, :, :-1] = 1 - voxelv_x
    # add back face
    faces_idx[10, :, :, :-1, :, :] = 1 - voxelv_z
    faces_idx[11, :, :, :-1, :, :] = 1 - voxelv_z

    faces_idx *= voxelv

    # N x H x W x D x 12
    faces_idx = faces_idx.permute(1, 2, 4, 5, 3, 0).squeeze(1)
    # (NHWD) x 12
    faces_idx = faces_idx.contiguous()
    faces_idx = faces_idx.view(-1, cube_faces.size(0))

    # boolean to linear index
    # NF x 2
    linind = torch.nonzero(faces_idx, as_tuple=False)
    # NF x 4
    nyxz = unravel_index(linind[:, 0], (N, H, W, D))

    # NF x 3: faces
    faces = torch.index_select(cube_faces, 0, linind[:, 1])

    grid_faces = []
    for d in range(cube_faces.size(1)):
        # NF x 3
        xyz = torch.index_select(cube_verts, 0, faces[:, d])
        permute_idx = torch.tensor([1, 0, 2], device=device)
        yxz = torch.index_select(xyz, 1, permute_idx)
        yxz += nyxz[:, 1:]
        # NF x 1
        temp = ravel_index(yxz, (H + 1, W + 1, D + 1))
        grid_faces.append(temp)
    # NF x 3
    grid_faces = torch.stack(grid_faces, dim=1)

    y, x, z = meshgrid_ij(torch.arange(H + 1), torch.arange(W + 1), torch.arange(D + 1))
    y = y.to(device=device, dtype=torch.float32)
    x = x.to(device=device, dtype=torch.float32)
    z = z.to(device=device, dtype=torch.float32)

    if align == "center":
        x = x - 0.5
        y = y - 0.5
        z = z - 0.5

    margin = 0.0 if align == "corner" else 1.0

    if norm:
        y = y * 2.0 / (H - margin) - 1.0
        x = x * 2.0 / (W - margin) - 1.0
        z = z * 2.0 / (D - margin) - 1.0

    # ((H+1)(W+1)(D+1)) x 3
    grid_verts = torch.stack((x, y, z), dim=3).view(-1, 3)

    if len(nyxz) == 0:
        verts_list = [torch.tensor([], dtype=torch.float32, device=device)] * N
        faces_list = [torch.tensor([], dtype=torch.int64, device=device)] * N
        return Meshes(verts=verts_list, faces=faces_list)

    num_verts = grid_verts.size(0)
    grid_faces += nyxz[:, 0].view(-1, 1) * num_verts
    idleverts = torch.ones(num_verts * N, dtype=torch.uint8, device=device)

    indices = grid_faces.flatten()
    if device.type == "cpu":
        indices = torch.unique(indices)
    idleverts.scatter_(0, indices, 0)
    grid_faces -= nyxz[:, 0].view(-1, 1) * num_verts
    split_size = torch.bincount(nyxz[:, 0], minlength=N)
    faces_list = list(torch.split(grid_faces, split_size.tolist(), 0))

    idleverts = idleverts.view(N, num_verts)
    idlenum = idleverts.cumsum(1)

    n = 0
    b = torch.index_select(
            grid_verts.index_select(
                0, (idleverts[n] == 0
            ).nonzero(as_tuple=False)[:, 0]), 
            1, 
            torch.tensor([2, 1, 0])
        )

    if affine is not None:
        verts_list = [(affine @ column_stack([b, torch.ones(b.size(0),1)]).float().T).T[:,:3]]
    else:
        verts_list = [b]
    
    faces_list = [nface - idlenum[n][nface] for n, nface in enumerate(faces_list)]

    return Meshes(verts=verts_list, faces=faces_list)
