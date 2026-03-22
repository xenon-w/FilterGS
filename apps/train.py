import os
from os.path import join
import numpy as np
from tqdm import tqdm
from LoG.utils.config import load_object, Config
from LoG.utils.command import update_global_variable, load_statedict, copy_git_tracked_files
from LoG.utils.metric import psnr as compute_psnr
import cv2
import torch,setproctitle


def _resolve_demo_split(cfg, split):
    """
    Some configs (including the provided block_A files) only define train/val.
    When users pass --cfg ... split demo we fall back gracefully instead of erroring.
    Preference order: exact split -> 'demo' -> 'val' -> 'train'.
    """
    candidate_keys = [split]
    if split not in candidate_keys:
        candidate_keys.append(split)
    if split.startswith('demo'):
        candidate_keys.extend(['demo', 'val'])
    candidate_keys.extend(['val', 'train'])
    seen = set()
    for key in candidate_keys:
        if key in seen:
            continue
        seen.add(key)
        if key in cfg:
            return key, cfg[key]
    raise KeyError(f"Split '{split}' not found in config (checked: {candidate_keys})")


def demo(cfg, model, device):
    split_key, split_cfg = _resolve_demo_split(cfg, cfg.split)
    dataset = load_object(split_cfg.dataset.module, split_cfg.dataset.args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # prepare the renderer
    if 'render' in split_cfg:
        renderer = load_object(split_cfg.render.module, split_cfg.render.args)
    else:
        renderer = load_object(cfg.train.render.module, cfg.train.render.args)
        renderer.split = 'demo'
    renderer.to(device)
    model.to(device)
    model.eval()
    if 'model_state' in split_cfg:
        model.set_state(**split_cfg['model_state'])
    if 'render_state' in split_cfg:
        renderer.set_state(**split_cfg['render_state'])
    from LoG.utils.trainer import prepare_batch
    from tqdm import tqdm
    total_time = 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
    render_type = cfg.get('render_type', 'rgb')
    if render_type == 'depth':
        renderer.render_depth = True
        depth_min = cfg.get('depth_min', 0.01)
        depth_max = cfg.get('depth_max', 10.)
    elif render_type == 'height':
        renderer.render_depth = True
        height_min = cfg.get('height_min', 0.01)
        height_max = cfg.get('height_max', 10.)

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = prepare_batch(batch, device)
        with torch.no_grad():
            output = renderer.vis(batch, model)
        if batch_idx > 10:
            break

    for batch in tqdm(dataloader):
        batch = prepare_batch(batch, device)
        if 'model_state' in batch:
            model.set_state(**batch['model_state'])
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = renderer.vis(batch, model)
            end.record()
            end.synchronize()
            total_time += start.elapsed_time(end)
        render = output['render'][0]
        if render_type == 'depth':
            depth = output['depth'][0]
            depth = (depth - depth_min)/(depth_max - depth_min)
            vis = renderer.marigold_depth_vis(depth)
        elif render_type == 'height':
            depth = output['height'][0]
            print(depth.min(), depth.max(), depth.mean())
            depth = (depth - height_min)/(height_max - height_min)
            vis = renderer.marigold_depth_vis(depth)        
        else:
            vis = renderer.tensor_to_bgr(render)
        outname = os.path.join(cfg.exp, cfg.split, render_type, f'{batch["index"].item():06d}.jpg')
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, vis)
        if 'mask' in output:
            mask = output['mask'][0].detach().cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            vis = np.dstack([vis, mask[:, :, None]])
            rgbaname = os.path.join(cfg.exp, cfg.split, 'rgba', f'{batch["index"].item():06d}.png')
            os.makedirs(os.path.dirname(rgbaname), exist_ok=True)
            cv2.imwrite(rgbaname, vis)
    print('Average time: {:.2f} ms, fps: {:.1f}'.format(total_time / len(dataloader), 1000 / (total_time / len(dataloader))))
    renderer.make_video(os.path.dirname(outname), fps=split_cfg.get('fps', 30))

def validate_for_metric(exp, dataset, model, renderer, device):
    renderer.to(device)
    model.to(device)
    model.eval()
    from LoG.utils.trainer import prepare_batch
    for scale in [8, 4, 2, 1]:
        if scale not in dataset.scales: continue
        dataset.set_state(scale=scale)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        outdir = join(exp, 'test', 'scale_{}'.format(scale))
        os.makedirs(join(outdir, 'gt'), exist_ok=True)
        os.makedirs(join(outdir, 'renders'), exist_ok=True)
        total_time = 0
        psnr_values = []
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch = prepare_batch(batch, device)
            with torch.no_grad():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                output = renderer.vis(batch, model)
                torch.cuda.synchronize()
                end.record()
            total_time += start.elapsed_time(end)

            with torch.no_grad():
                gt_tensor = batch['image'][0].permute(2, 0, 1)
                render_tensor = torch.clamp(output['render'][0], 0., 1.)
                frame_psnr = compute_psnr(render_tensor, gt_tensor)
                psnr_values.append(frame_psnr)
                tqdm.write(f'[PSNR][scale {scale}] frame {batch_idx:04d}: {frame_psnr:.2f} dB')
            append_mask = False
            # use foreground mask
            if 'mask' in batch.keys():
                mask = batch['mask'][0].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                append_mask = True
            if torch.is_tensor(batch['image'][0]):
                gt = batch['image'][0].cpu().numpy()
                gt = (gt[:,:,::-1] * 255).astype(np.uint8)
                if append_mask:
                    gt = np.dstack([gt, mask[:, :, None]])
                gt_name = join(outdir, 'gt', '%04d.png'%(batch_idx))
                cv2.imwrite(gt_name, gt)
            renders = output['render'][0].permute(1, 2, 0).cpu().numpy()
            renders = (np.clip(renders[:, :,::-1], 0., 1.) * 255).astype(np.uint8)
            if append_mask:
                renders = np.dstack([renders, mask[:, :, None]])
            render_name = join(outdir, 'renders', '%04d.png'%(batch_idx))
            cv2.imwrite(render_name, renders)
        print('scale: {}, Average time: {:.2f} ms, fps: {:.1f}'.format(scale, total_time / len(dataloader), 1000 / (total_time / len(dataloader))))
        if psnr_values:
            avg_psnr = sum(psnr_values) / len(psnr_values)
            max_psnr = max(psnr_values)
            min_psnr = min(psnr_values)
            print(f'[PSNR][scale {scale}] avg: {avg_psnr:.2f} dB, max: {max_psnr:.2f} dB, min: {min_psnr:.2f} dB')

def main():
    usage = 'run'
    args, cfg = Config.load_args(usage=usage)
    cfg = update_global_variable(cfg, cfg)

    exp = cfg.exp
    if isinstance(exp, str):
        norm = exp.replace("\\", "/")
        if norm.startswith("output/") and not norm.startswith("output/train/"):
            rel = norm[len("output/"):].strip("/")
            if rel.endswith("/train"):
                rel = rel[:-len("/train")]
            elif rel == "train":
                rel = ""
            exp = os.path.join("output", "train", rel) if rel else os.path.join("output", "train")
            cfg.exp = exp
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    print(f'Using GPUs: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    print('Write to {}'.format(exp))
    # write the parameter to the exp
    os.makedirs(exp, exist_ok=True)
    if cfg.split == 'train':
        print(cfg, file=open(os.path.join(exp, 'config.yaml'), 'w'))
    from LoG.utils.trainer import Trainer, seed_everything
    seed_everything(666)

    device = torch.device('cuda')
    
    # Check and generate point cloud if needed
    if hasattr(cfg.model.args, 'gaussian') and hasattr(cfg.model.args.gaussian, 'init_ply'):
        ply_filename = cfg.model.args.gaussian.init_ply.filename
        if not os.path.exists(ply_filename):
            print(f'[Auto Point Cloud] Point cloud not found: {ply_filename}')
            print(f'[Auto Point Cloud] Generating random point cloud...')
            
            # Get camera centers from dataset to determine scene bounds
            dataset = load_object(cfg.train.dataset.module, cfg.train.dataset.args)
            centers = []
            for i in range(min(len(dataset), 100)):  # Sample up to 100 cameras
                try:
                    camera = dataset.get_camera(i)
                    if hasattr(camera, 'c2w'):
                        center = camera.c2w[:3, 3]
                        centers.append(center)
                    elif hasattr(camera, 'T'):
                        center = -camera.R.T @ camera.T
                        centers.append(center)
                except:
                    continue
            
            if centers:
                centers = np.array(centers)
                scene_center = np.mean(centers, axis=0)
                scene_radius = np.max(np.linalg.norm(centers - scene_center, axis=1)) * 1.1
                
                # Generate random points
                num_pts = 100_000
                xyz = np.random.random((num_pts, 3)) * 2 * scene_radius - scene_radius
                xyz += scene_center
                rgb = np.random.random((num_pts, 3)) * 255.0
                
                # Save point cloud
                os.makedirs(os.path.dirname(ply_filename), exist_ok=True)
                np.savez(ply_filename, xyz=xyz.astype(np.float32), rgb=rgb.astype(np.float32))
                print(f'[Auto Point Cloud] Generated {num_pts} random points')
                print(f'[Auto Point Cloud] Scene center: {scene_center}, radius: {scene_radius}')
                print(f'[Auto Point Cloud] Saved to: {ply_filename}')
            else:
                print(f'[Auto Point Cloud] Warning: Could not determine scene bounds, using default')
                # Generate default random points
                num_pts = 100_000
                xyz = np.random.random((num_pts, 3)) * 20 - 10  # [-10, 10] cube
                rgb = np.random.random((num_pts, 3)) * 255.0
                
                os.makedirs(os.path.dirname(ply_filename), exist_ok=True)
                np.savez(ply_filename, xyz=xyz.astype(np.float32), rgb=rgb.astype(np.float32))
                print(f'[Auto Point Cloud] Generated {num_pts} default random points')
                print(f'[Auto Point Cloud] Saved to: {ply_filename}')
    
    model = load_object(cfg.model.module, cfg.model.args)
    if cfg.split == 'train':
        outdir = copy_git_tracked_files('./', exp)
        dataset = load_object(cfg.train.dataset.module, cfg.train.dataset.args)
        if 'base_iter' in cfg:
            base_iter = cfg.base_iter
        else:
            # round 100 iteration
            if len(dataset) < 1000:
                base_iter = (len(dataset) // 100 + 1) * 100
            else:
                base_iter = (len(dataset) // 1000 + 1) * 1000
        print('Base iteration: {}'.format(base_iter))
        model.base_iter = base_iter
        renderer = load_object(cfg.train.render.module, cfg.train.render.args)
        trainer = Trainer(cfg, model, renderer, logdir=outdir)
        trainer.to(device)
        trainer.init(dataset)
        trainer.fit(dataset)
    elif cfg.split.startswith('demo') or cfg.split == 'trainvis':
        if cfg.split == 'trainvis':
            cfg.split = 'train'
        if 'ckptname' in cfg.keys():
            model.load_state_dict(load_statedict(cfg.ckptname))
        demo(cfg, model, device)
    elif cfg.split == 'val':
        if 'ckptname' in cfg.val.keys():
            ckpt_path = cfg.val['ckptname']
            print(f'Loading checkpoint from: {ckpt_path}')
            model.load_state_dict(load_statedict(ckpt_path))
        else:
            print('[Warning] No checkpoint specified under cfg.val.ckptname; using current model weights.')
        if 'model_state' in cfg.val:
            model.set_state(**cfg.val['model_state'])
        dataset = load_object(cfg.val.dataset.module, cfg.val.dataset.args)
        renderer = load_object(cfg.train.render.module, cfg.train.render.args)
        renderer.split = 'val'
        validate_for_metric(exp, dataset, model, renderer, device)

if __name__ == '__main__':
    try:
        setproctitle.setproctitle("WangYiXian_City_Train")
    except:
        pass  # Ignore error if setproctitle is unavailable
    main()
