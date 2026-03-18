import os
from os.path import join
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from .image_base import ImageBase
from .base import prepare_camera, rescale_camera
from .camera_utils import get_center_and_diag

def read_undistort_rescale_write(info):
    flag_read_img = False
    for scale in info['scales']:
        cachename = join(info['cache'], str(scale), info['imgname'])
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        if not os.path.exists(cachename):
            flag_read_img = True
            break
    else:
        return 0
    imgname = join(info['root'], info['imgname'])
    assert os.path.exists(imgname), imgname
    camera = info['camera']
    if flag_read_img:
        # read the image with pillow, because opencv ignore the orientation of image
        # img = cv2.imread(imgname)
        img = Image.open(imgname)
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_h, img_w = img.shape[:2]
        cam_h, cam_w = camera['H'], camera['W']
        if img_h != cam_h or img_w != cam_w:
            scale_w = img_w / cam_w if cam_w != 0 else 1.
            scale_h = img_h / cam_h if cam_h != 0 else 1.
            if not np.isclose(scale_w, 1.0, atol=1e-4) or not np.isclose(scale_h, 1.0, atol=1e-4):
                print(f'[ImageDataset] warn: {imgname} resolution {img_w}x{img_h} '
                      f'does not match COLMAP camera {cam_w}x{cam_h}. '
                      f'Rescaling intrinsics by ({scale_w:.6f}, {scale_h:.6f}).')
            camera['K'] = camera['K'].copy()
            camera['K'][0, 0] *= scale_w
            camera['K'][0, 2] *= scale_w
            camera['K'][1, 1] *= scale_h
            camera['K'][1, 2] *= scale_h
            camera['W'] = img_w
            camera['H'] = img_h
            camera.pop('mapx', None)
            camera.pop('mapy', None)
        else:
            img_h, img_w = cam_h, cam_w
        if 'mapx' in info['camera'].keys() and 'mapy' in info['camera'].keys():
            mapx, mapy = info['camera']['mapx'], info['camera']['mapy']
        else:
            width, height = camera['W'], camera['H']
            newK, roi = cv2.getOptimalNewCameraMatrix(camera['K'], camera['dist'], 
                        (width, height), 0, (width,height), centerPrincipalPoint=True)
            mapx, mapy = cv2.initUndistortRectifyMap(camera['K'], camera['dist'], None, newK, (width, height), 5)
            camera['K'] = newK
        if mapx is not None and mapy is not None:
            img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    else:
        if 'mapx' not in info['camera'].keys():
            width, height = camera['W'], camera['H']
            newK, roi = cv2.getOptimalNewCameraMatrix(camera['K'], camera['dist'], 
                        (width, height), 0, (width,height), centerPrincipalPoint=True)
            mapx, mapy = cv2.initUndistortRectifyMap(camera['K'], camera['dist'], None, newK, (width, height), 5)
            camera['K'] = newK

    for scale in info['scales']:
        cachename = join(info['cache'], str(scale), info['imgname'])
        if os.path.exists(cachename):
            continue
        camera_scale = camera.copy()
        camera_scale['K'] = camera['K'].copy()
        W = int(camera['W'] / scale)
        H = int(camera['H'] / scale)
        dst = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        cv2.imwrite(cachename, dst)
    return 0

class ImageDataset(ImageBase):
    @staticmethod
    def init_camera(camera):
        width, height = camera['W'], camera['H']
        assert width != 0 and height != 0, f'width or height is 0: {width}, {height}'
        dist = camera['dist']
        if np.linalg.norm(dist) < 1e-5:
            mapx, mapy = None, None
            newK = camera['K'].copy()
        else:
            newK, roi = cv2.getOptimalNewCameraMatrix(camera['K'], camera['dist'], 
                        (width, height), 0, (width,height), centerPrincipalPoint=True)
            mapx, mapy = cv2.initUndistortRectifyMap(camera['K'], camera['dist'], None, newK, (width, height), 5)
        return mapx, mapy, newK

    def check_undis_camera(self, camname, cameras_cache, camera_undis, share_camera=False):
        if share_camera:
            cache_camname = 'cache'
        else:
            if '/' in camname:
                cache_camname = camname.split('/')[0]
            else:
                cache_camname = camname

        if cache_camname not in cameras_cache:
            print(f'[{self.__class__.__name__}] init camera {cache_camname}')
            cameras_cache[cache_camname] = self.init_camera(camera_undis)
        mapx, mapy, newK = cameras_cache[cache_camname]
        camera = {
            'K': newK,
            'mapx': mapx,
            'mapy': mapy
        }
        for key in ['R', 'T', 'W', 'H', 'center', 'dist']:
            camera[key] = camera_undis[key]
        return camera

    def __init__(self, root, cameras='sparse/0', scales=[1,2,4], 
                scale3d=1., ext='.JPG', images='images', scale_camera_K=1., 
                mask_ignore=None,
                 pre_undis=True, share_camera=False, crop_size=[-1, -1],
                 crop_ltrb=None, namelist=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.root = os.path.abspath(root)
        self.cameras = cameras
        self.image_dir = images
        self.ext = ext
        self.namelist = namelist
        self.mask_ignore = mask_ignore
        self.scales = scales
        self.downsample_scale = 1
        self.scale3d = scale3d
        self.crop_size = crop_size
        self.crop_ltrb = crop_ltrb
        
        if self.cache is None:
            self.cache = join(self.root, 'cache')
            cachedir = self.cache
        else:
            if self.cache.endswith('.pkl'):
                cachedir = join(self.root, self.cache.replace('.pkl', ''))
            else:
                cachedir = join(self.root, self.cache)
        self.cachedir = cachedir
        flag, infos = self.read_cache(name=cachedir+'.pkl')
        if not flag:
            cameras = self.check_cameras(scale3d=scale3d, scale_camera_K=scale_camera_K)
            print(f'[{self.__class__.__name__}] 从相机文件加载了 {len(cameras)} 个相机')
            
            # 如果指定了namelist，只处理列表中的相机
            if self.namelist is not None:
                print(f'[{self.__class__.__name__}] 使用namelist过滤，指定相机: {self.namelist}')
                print(f'[{self.__class__.__name__}] 相机文件中的相机数量: {len(cameras)}')
                print(f'[{self.__class__.__name__}] 相机文件中的前10个相机: {list(cameras.keys())[:10]}')
                
                filtered_cameras = {}
                for name in self.namelist:
                    if name in cameras:
                        filtered_cameras[name] = cameras[name]
                        print(f'[{self.__class__.__name__}] 找到相机: {name}')
                    else:
                        print(f'[{self.__class__.__name__}] 警告: namelist中的相机 {name} 在相机文件中不存在')
                cameras = filtered_cameras
                print(f'[{self.__class__.__name__}] 过滤后剩余 {len(cameras)} 个相机')
            
            # undistort and scale
            cameras_cache = {}
            infos = []
            processed_count = 0
            for camname, camera_dis in cameras.items():
                processed_count += 1
                print(f'[{self.__class__.__name__}] 处理相机 {processed_count}/{len(cameras)}: {camname}')
                if pre_undis:
                    camera = self.check_undis_camera(camname, cameras_cache, camera_dis, share_camera)
                else:
                    camera = camera_dis
                camera_ = camera.copy()
                # camera_.pop('mapx')
                # camera_.pop('mapy')
                
                # 尝试不同的图像格式
                imgname = None
                possible_extensions = [ext, ext.lower(), ext.upper()]
                # 添加常见的图像格式
                common_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP', '.tiff', '.TIFF']
                for common_ext in common_extensions:
                    if common_ext not in possible_extensions:
                        possible_extensions.append(common_ext)
                
                for test_ext in possible_extensions:
                    test_imgname = join(self.root, images, camname + test_ext)
                    if os.path.exists(test_imgname):
                        imgname = test_imgname
                        break
                
                if imgname is None:
                    # 尝试模糊匹配
                    images_dir = join(self.root, images)
                    if os.path.exists(images_dir):
                        files_in_dir = os.listdir(images_dir)
                        
                        # 尝试模糊匹配文件名
                        for file in files_in_dir:
                            file_base = os.path.splitext(file)[0]
                            if file_base == camname or file_base.startswith(camname) or camname in file_base:
                                imgname = join(images_dir, file)
                                break
                    
                    if imgname is None:
                        print(f'[{self.__class__.__name__}] 警告: 无法找到相机 {camname} 的图像文件')
                        print(f'  尝试的路径: {join(self.root, images, camname + ext)}')
                        print(f'  图像目录: {join(self.root, images)}')
                        if os.path.exists(images_dir):
                            files_in_dir = os.listdir(images_dir)
                            print(f'  目录中的文件: {files_in_dir[:10]}...' if len(files_in_dir) > 10 else f'  目录中的文件: {files_in_dir}')
                        continue
                
                # 获取实际的文件扩展名和相对路径
                actual_ext = os.path.splitext(imgname)[1]
                actual_filename = os.path.basename(imgname)
                infos.append({
                    'root': self.root,
                    'cache': cachedir,
                    'imgname': join(images, actual_filename),
                    'camera': camera_,
                    'scales': scales
                })
            print(f'[{self.__class__.__name__}] undistort and scale {len(infos)} images ')
            for info in tqdm(infos):
                read_undistort_rescale_write(info)
                info['camera'].pop('mapx', None)
                info['camera'].pop('mapy', None)
            self.write_cache(infos, name=cachedir+'.pkl')
        
        if len(infos) == 0:
            # 提供更详细的调试信息
            debug_info = f'[{self.__class__.__name__}] 没有找到任何有效的图像文件！\n'
            debug_info += f'配置信息:\n'
            debug_info += f'  - 根目录: {self.root}\n'
            debug_info += f'  - 图像目录: {join(self.root, images)}\n'
            debug_info += f'  - 扩展名: {ext}\n'
            debug_info += f'  - 相机文件: {join(self.root, cameras)}\n'
            
            if self.namelist is not None:
                debug_info += f'  - namelist: {self.namelist}\n'
            
            # 检查图像目录
            images_dir = join(self.root, images)
            if os.path.exists(images_dir):
                files = os.listdir(images_dir)
                debug_info += f'图像目录中的文件数量: {len(files)}\n'
                debug_info += f'前20个文件: {files[:20]}\n'
                
                # 检查namelist中的文件
                if self.namelist is not None:
                    debug_info += f'namelist文件匹配情况:\n'
                    for name in self.namelist:
                        matching_files = [f for f in files if name in f]
                        debug_info += f'  {name}: {matching_files}\n'
            else:
                debug_info += f'图像目录不存在: {images_dir}\n'
            
            raise ValueError(debug_info)
        
        centers = np.stack([-info['camera']['R'].T @ info['camera']['T'] for info in infos], axis=0)
        offset, radius = get_center_and_diag(centers)
        
        # Check if point cloud exists, if not generate random points
        self.check_and_generate_pointcloud(centers, radius)
        
        self.current_scale = scales[-1]
        self.infos = infos

    def check_and_generate_pointcloud(self, centers, radius):
        """Check if point cloud exists, if not generate random points like 3DGS"""
        import os
        from os.path import join
        
        # Check if sparse.npz exists
        sparse_path = join(self.root, self.cameras, 'sparse.npz')
        if os.path.exists(sparse_path):
            return
        
        print(f'[{self.__class__.__name__}] No point cloud found, generating random points...')
        
        # Calculate scene bounds based on camera centers and radius
        scene_center = np.mean(centers, axis=0).flatten()  # Ensure it's 1D
        scene_radius = radius * 1.1  # Similar to 3DGS
        
        # Generate random points inside the scene bounds
        num_pts = 100_000  # Same as 3DGS
        print(f'[{self.__class__.__name__}] Generating {num_pts} random points...')
        
        # Generate points in a cube around scene center
        xyz = np.random.random((num_pts, 3)) * 2 * scene_radius - scene_radius
        xyz += scene_center
        
        # Generate random colors (similar to 3DGS)
        rgb = np.random.random((num_pts, 3)) * 255.0
        
        # Create sparse directory if it doesn't exist
        sparse_dir = join(self.root, self.cameras)
        os.makedirs(sparse_dir, exist_ok=True)
        
        # Save as sparse.npz
        np.savez(sparse_path, xyz=xyz.astype(np.float32), rgb=rgb.astype(np.float32))
        print(f'[{self.__class__.__name__}] Generated random point cloud saved to: {sparse_path}')
        print(f'[{self.__class__.__name__}] Scene center: {scene_center}, radius: {scene_radius}')

    def set_state(self, scale=None, crop_size=None, downsample_scale=1, namelist=None):
        if scale is not None:
            assert scale in self.scales, f'scale {scale} not in {self.scales}'
            self.current_scale = scale
        self.downsample_scale = downsample_scale
        if crop_size is not None:
            print(f'[{self.__class__.__name__}] set crop size {crop_size}')
            self.crop_size = crop_size
        print(f'[{self.__class__.__name__}] set scale {scale}, crop_size: {self.crop_size}, downsample_scale: {downsample_scale}')

    def __len__(self):
        if self.partial_indices is None:
            return len(self.infos)
        else:
            return len(self.partial_indices)

    def crop_image(self, img, crop_size):
        if isinstance(img, str):
            pass
        xranges = np.arange(0, img.shape[1] - crop_size[1] + 1)
        yranges = np.arange(0, img.shape[0] - crop_size[0] + 1)
        sample_x = int(np.random.choice(xranges))
        sample_y = int(np.random.choice(yranges))
        l, t, r, b = sample_x, sample_y, sample_x + crop_size[1], sample_y + crop_size[0]
        return l, t, r, b

    def update_crop(self, img, camera, l, t, r, b):
        camera['K'] = camera['K'].copy()
        img = img[t:b, l:r]
        camera['K'][0, 2] -= l
        camera['K'][1, 2] -= t
        camera['W'] = r - l
        camera['H'] = b - t
        return img, camera

    def __getitem__(self, index):
        if self.partial_indices is None:
            true_index = index
        else:
            true_index = self.partial_indices[index]
        data = self.infos[true_index]
        imgname = data['imgname']
        imgname = join(self.cachedir, str(self.current_scale), imgname)
        if self.read_img and os.path.exists(imgname):
            img = self.read_image_with_cache(imgname)
        else:
            img = imgname
        if self.downsample_scale != 1:
            scale = self.downsample_scale * self.current_scale
            camera = rescale_camera(data['camera'], scale)
            if self.read_image:
                # cv2.INTER_AREA for anti-alias resize
                img = cv2.resize(img, (camera['W'], camera['H']), interpolation=cv2.INTER_AREA)
        else:
            camera = rescale_camera(data['camera'], self.current_scale)
        # check mask
        msk = None
        if self.mask_ignore is not None:
            mask_cfg = self.mask_ignore

            def _mask_cfg_get(cfg, key, default=None):
                if isinstance(cfg, dict):
                    return cfg.get(key, default)
                return getattr(cfg, key, default)

            mask_path = _mask_cfg_get(mask_cfg, 'path')
            mask_type = _mask_cfg_get(mask_cfg, 'type', 'background')
            mask_scale = _mask_cfg_get(mask_cfg, 'scale', None)
            invert_default = str(mask_type).lower() == 'background'
            invert_mask = bool(_mask_cfg_get(mask_cfg, 'invert', invert_default))
            dilate_cfg = _mask_cfg_get(mask_cfg, 'dilate', 'auto' if invert_default else None)

            if self.read_img:
                msk_relpath = data['imgname'].replace(self.ext, '.png')
                mask_dir = join(self.root, mask_path)
                candidate_paths = [
                    join(mask_dir, msk_relpath),
                    join(mask_dir, os.path.basename(msk_relpath)),
                ]
                mskname = None
                for cand in candidate_paths:
                    if os.path.exists(cand):
                        mskname = cand
                        break
                if mskname is not None:
                    msk = self.read_mask(mskname)
                    if mask_scale is not None and self.current_scale != mask_scale:
                        import ipdb;ipdb.set_trace()

                    # Apply optional dilation
                    border = None
                    if dilate_cfg is not None:
                        if isinstance(dilate_cfg, str) and dilate_cfg.lower() == 'auto':
                            border = int(msk.shape[0] // 50) * 2 + 1
                        elif isinstance(dilate_cfg, (int, float)):
                            border = int(round(dilate_cfg))
                            if border % 2 == 0:
                                border += 1
                    if border is not None and border > 1:
                        kernel = np.ones((border, border), np.float32)
                        msk = cv2.dilate(msk, kernel)

                    if invert_mask:
                        msk = 1 - msk
                    msk = np.clip(msk, 0.0, 1.0)
        
        if self.crop_ltrb is not None and not isinstance(img, str):
            l, t, r, b = self.crop_ltrb
            img, camera = self.update_crop(img, camera, l, t, r, b)
        elif self.crop_size[0] > 0 and self.crop_size[1] > 0 and not isinstance(img, str):
            l, t, r, b = self.crop_image(img, self.crop_size)
            img, camera = self.update_crop(img, camera, l, t, r, b)
        camera = prepare_camera(camera, scale=1, znear=self.znear, zfar=self.zfar)
        ret = {
            'image': img,
            'imgname': imgname,
            'index': index,
            'true_index': true_index,
            'camera': camera,
        }
        if msk is not None:
            ret['mask_ignore'] = msk
        ret.update(data.get('extra', {}))
        return ret

class DepthDataset(ImageDataset):
    def __init__(self, depth_scale, depth_dir='depth', **kwargs):
        super().__init__(**kwargs)
        self.depth_scale = depth_scale
        self.depth_dir = depth_dir
    
    def __getitem__(self, index):
        ret = super().__getitem__(index)
        depthname = ret['imgname'].replace(
            self.image_dir, self.depth_dir)\
            .replace(f'{os.sep}{self.current_scale}{os.sep}{self.depth_dir}', f'{os.sep}{self.depth_scale}{os.sep}{self.depth_dir}')\
            + '.png'
        if self.read_image:
            # depth: (0.0 -> 1.0)
            depth = self.read_depth(depthname)
        else:
            depth = depthname
        ret['depth'] = depth
        return ret
