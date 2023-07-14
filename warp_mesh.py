# This is an exploratory example of the question of:
#  Given a flow matrix (grid), how can we warp a point while being differentiable
#   - The flow matrix is defined by the grid
#   - the point is defined by (x, y, z) as floats

# Key idea
# Embed the float into an image as a 2x2x2 block. It's centre of mass end up being the (x, y, z)
# Warp the image using the flow matrix and retrieve back the centre of mass. This will be the point after warping

import torch
import torch.nn.functional as F
import math
import sys

def get_1d_com(arr):
    # get_1d_com(torch.tensor([0,1,0,1,0])) -> 2.5

    # -|x|-|x|-
    #   m   m
    # (m*1.5 + m*3.5)/(2m) = 2.5

    return (arr * (torch.arange(arr.size(0))+0.5)).sum() / arr.sum()

def get_3d_com(tensor):
    # tensor.shape: (N, C, H, W), N=1
    assert tensor.ndim==4 and tensor.size(0)==1
    return torch.stack(
        [get_1d_com(tensor.sum(idx)) for idx in ((0,2,3), (0,1,3), (0,1,2))]
    )

def put_1d_com(value):
    '''
    -|-|-|-|1|x|-----

    Returns the index of x and x

    put_1d_com(3.14, 5) -> returns 3, 1.7777

    1*(d-1+0.5) + x*(d+0.5) = v*(1+x)
    d-0.5 + x(d+0.5) = v + v*x
    d-0.5-v = (v-d-0.5)*x
    x = (d-0.5-v)/(v-d-0.5)

    v: value, d: floor index
    E.g. v=3.14, d=3, x=(3-0.5-3.14)/(3.14-3-0.5)=16/9=1.7777
    '''
    d = math.floor(value)

    return d, (d-0.5-value)/(value-d-0.5)

def put_3d_com(coord, shape):
    '''
    coord: three float values, (x,y,z)
    shape: three dims for the output

    Returns a tensor specify by (1, shape)

    Embed a float coordinate in an image so that the COM is the float.
    '''
    (dx, vx), (dy, vy), (dz, vz) = [put_1d_com(i) for i in coord]

    img = torch.zeros(shape)

    # Set two values
    img[dx-1, dy-1, dz-1] = 1
    img[dx, dy-1, dz-1] = vx

    # Copy the entire y plane (2 -> 4)
    img[:,dy,:] = img[:,dy-1,:] * vy
    
    # Copy the entire z plan (4 -> 8)
    img[:,:,dz] = img[:,:,dz-1] * vz

    # normalise
    img /= img.sum()

    return img.unsqueeze(0)

def cvt_aff_matrix(m, dims):
    '''
    Convert the affine matrix from the torch version to generic version
    m: the torch affine matrix;
    dims: the output dimension, (x, y, z)
    '''
    # Append the last row
    matrix = torch.cat((m.clone().detach()[0], torch.tensor([[0,0,0,1]])))
    
    # Unpack the dimension
    x, y, z = dims

    # Change the offsets
    matrix[0, -1] = x/2 * (1 - matrix[0,0]) + m[0,0,-1] * (x-1)/2
    matrix[1, -1] = y/2 * (1 - matrix[1,1]) + m[0,1,-1] * (y-1)/2
    matrix[2, -1] = z/2 * (1 - matrix[2,2]) + m[0,2,-1] * (z-1)/2

    matrix = torch.inverse(matrix).unsqueeze(0)
    return matrix

if __name__ == "__main__":
    affine_matrix = torch.tensor(
        [[1, 0, 0, 0.1],
         [0, 1, 0, 0],
         [0, 0, 1, 0.3]],
        dtype=torch.float,
        requires_grad=True
    ).unsqueeze(0)

    img = put_3d_com((2.1, 32, 43), (3,100,100)).unsqueeze(0)

    print(f'original: {get_3d_com(img.squeeze(0))}')

    grid = F.affine_grid(affine_matrix, (1,1,100,100,3), align_corners=True)
    warped = F.grid_sample(img.permute((0,1,4,3,2)), grid, align_corners=True).permute((0,1,4,3,2))

    print(f'get_3d_com: {get_3d_com(warped.squeeze(0))}')

    matrix = cvt_aff_matrix(affine_matrix, (3, 100, 100))
    torch.matmul(matrix[0], torch.tensor([2.1, 32, 43, 1]))
    print(f'matmul: {torch.matmul(matrix[0], torch.tensor([2.1, 32, 43, 1]))[:-1]}')

    print(f'warped sum: {warped.sum()}')
    print('end')