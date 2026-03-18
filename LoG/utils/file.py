import os
import numpy as np
import torch

def read_ply(filename):
    from plyfile import PlyData
    plydata = PlyData.read(filename)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return positions, colors

def write_ply(outname, xyz, colors):
    """
        xyz: (N, 3)
        colors: (N, 3)
        filename: str
    """
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    from plyfile import PlyData, PlyElement
    assert xyz.shape == colors.shape
    colors = np.clip(colors, 0, 1)
    structured_array = np.zeros(xyz.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # color => uint8
    colors = (colors * 255).astype(np.uint8)
    for i in range(xyz.shape[0]):
        structured_array[i] = tuple(xyz[i]) + tuple(colors[i])

    el = PlyElement.describe(structured_array, 'vertex')
    PlyData([el]).write(outname)

def write_mesh(outname, vertices, faces, vertex_colors):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.io.write_triangle_mesh(outname, mesh)

def read_ply_and_log(filename, scale3d=1., **kwargs):
    assert os.path.exists(filename), f'file not found: {filename}'
    if filename.endswith('.ply'):
        from plyfile import PlyData
        plydata = PlyData.read(filename)
        vertices = plydata['vertex']
        positions = scale3d * np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    elif filename.endswith('.npz'):
        plydata = dict(np.load(filename))
        positions = scale3d * plydata['xyz']
        colors = plydata['rgb'] / 255.
    elif filename.endswith('.obj'):
        import trimesh
        mesh = trimesh.load(filename)
        positions = np.asarray(mesh.vertices)
        colors = np.random.rand(*positions.shape)
    if 'offset' in kwargs:
        positions = positions - np.array(kwargs['offset']).reshape(1, 3)
    return positions, colors
    
def create_from_point(filename, scale3d, ret_scale=True, **kwargs):
    if isinstance(filename, dict):
        # load from dict
        xyz = filename['xyz']
        colors = filename['colors']
    elif isinstance(filename, str) and (filename.endswith('.ply') or filename.endswith('.npz') or filename.endswith('.obj')):
        xyz, colors = read_ply_and_log(filename, scale3d, **kwargs)
    else:
        raise NotImplementedError
    xyz = torch.FloatTensor(xyz)
    colors = torch.FloatTensor(colors)
    if ret_scale:
        from simple_knn._C import distCUDA2
        dist2 = torch.clamp_min(distCUDA2(xyz.cuda()), 1e-7) #3e-4^2
        # scales = torch.clamp(torch.sqrt(dist2), self.scale_min * 2, self.scale_max / 2).to(xyz.device)
        scales = torch.sqrt(dist2).cpu()
    else:
        scales = None
    return xyz, colors, scales
