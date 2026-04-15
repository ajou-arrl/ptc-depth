"""PTC-Depth Rerun visualizer.

Usage:
    python examples/visualize_sample.py --dataset roadside
    python examples/visualize_sample.py --dataset roadside_thr
    python examples/visualize_sample.py --dataset forest
    python examples/visualize_sample.py --dataset roadside --segmentation
    python examples/visualize_sample.py --data-path /path/to/custom/data
"""

import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from glob import glob

PROJECT_ROOT = Path(__file__).parent.parent

import rerun as rr
from rerun import blueprint as rrb
from ptc_depth import PTCDepth


def compute_metrics(z, gt, max_depth=80.0):
    m = (z > 0) & (gt > 0) & (gt < max_depth) & np.isfinite(z) & np.isfinite(gt)
    if m.sum() < 100:
        return None
    ratio = np.maximum(z[m] / gt[m], gt[m] / z[m])
    return {
        'd125': (ratio < 1.25).mean() * 100,
    }


def depth_to_colormap(d, vmin=0, vmax=80):
    import matplotlib.pyplot as plt
    d = d.copy().astype(np.float32)
    d[~np.isfinite(d) | (d <= 0) | (d > 200)] = np.nan
    nan_mask = np.isnan(d)
    norm = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
    norm[nan_mask] = 0
    colored = (plt.cm.Spectral_r(norm)[:, :, :3] * 255).astype(np.uint8)
    colored[nan_mask] = 0
    return colored


def error_to_colormap(z, gt, max_depth=80.0):
    import matplotlib.pyplot as plt
    H, W = gt.shape[:2]
    err_img = np.zeros((H, W, 3), dtype=np.uint8)
    if z is None or np.size(z) == 0:
        return err_img
    m = (z > 0) & (z < max_depth) & (gt > 0) & (gt < max_depth) & np.isfinite(z) & np.isfinite(gt)
    if m.sum() < 10:
        return err_img
    with np.errstate(divide='ignore', invalid='ignore'):
        signed_err = np.where(gt > 0, (z - gt) / gt, 0)
    norm = np.clip(signed_err / 0.5 * 0.5 + 0.5, 0, 1)
    cmap = plt.colormaps['RdBu_r']
    rgba = cmap(norm[m])
    err_img[m, 0] = (rgba[:, 0] * 255).astype(np.uint8)
    err_img[m, 1] = (rgba[:, 1] * 255).astype(np.uint8)
    err_img[m, 2] = (rgba[:, 2] * 255).astype(np.uint8)
    return err_img


def remove_flying_pixels(depth, max_depth=80.0, grad_thresh=0.15):
    valid = (depth > 0) & (depth < max_depth) & np.isfinite(depth)
    if grad_thresh <= 0:
        return valid
    d = depth.copy()
    d[~valid] = 0
    dz_x = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
    dz_y = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
    max_grad = np.maximum(dz_x, dz_y)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_grad = max_grad / np.maximum(d, 1e-3)
    valid[rel_grad > grad_thresh] = False
    return valid


def depth_to_pointcloud(depth, image, fx, fy, cx, cy, max_depth=80.0, grad_thresh=0.15):
    H, W = depth.shape[:2]
    valid = remove_flying_pixels(depth, max_depth, grad_thresh)
    vs, us = np.where(valid)
    if len(vs) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    zs = depth[vs, us]
    xs = (us - cx) / fx * zs
    ys = (vs - cy) / fy * zs
    points = np.stack([xs, ys, zs], axis=-1).astype(np.float32)
    if image is not None and image.ndim == 3:
        colors = np.ascontiguousarray(image[vs, us, ::-1], dtype=np.uint8)
    else:
        norm = np.clip(zs / (max_depth + 1e-6), 0, 1)
        norm_u8 = (norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm_u8.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
        colors = np.ascontiguousarray(colored[:, ::-1], dtype=np.uint8)
    return points, colors


DATASET_MAP = {
    'roadside':     'wheel_roadside_rgb',
    'roadside_thr': 'wheel_roadside_thr',
    'forest':       'wheel_forest_rgb',
}


def main():
    parser = argparse.ArgumentParser(description='PTC-Depth Rerun visualizer')
    parser.add_argument('--dataset', type=str, default='roadside',
                        choices=DATASET_MAP.keys(),
                        help='Sample dataset name (roadside, roadside_thr, forest)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Custom data path (overrides --dataset)')
    parser.add_argument('--segmentation', action='store_true',
                        help='Enable edge-aware segmentation for per-segment scale')
    args = parser.parse_args()

    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = PROJECT_ROOT / 'data' / DATASET_MAP[args.dataset]

    if not data_path.exists():
        print(f"Data not found: {data_path}")
        print("Run: python examples/download_sample.py")
        sys.exit(1)

    # Load intrinsics
    with open(data_path / 'intrinsics.json') as f:
        intr = json.load(f)
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    H, W = intr['H'], intr['W']

    # Images, baselines, GT
    image_paths = sorted(glob(str(data_path / 'images' / '*.png')))
    n_frames = len(image_paths)
    bl_file = data_path / 'baselines.npy'
    baselines = np.load(str(bl_file)) if bl_file.exists() else np.ones(n_frames)
    gt_dir = data_path / 'depth_gt'
    has_gt = gt_dir.exists()

    start = 0

    # Detect LiDAR valid row range from first GT frame for cropping
    crop_r0, crop_r1 = 0, H
    if has_gt:
        for gi in range(n_frames):
            gf = gt_dir / f'{gi:06d}.npy'
            if gf.exists():
                gt_sample = np.load(str(gf))
                valid_rows = np.where((gt_sample > 0).any(axis=1))[0]
                if len(valid_rows) > 10:
                    crop_r0 = max(0, int(valid_rows[0]) - 5)
                    crop_r1 = min(H, int(valid_rows[-1]) + 5)
                    break

    # Blueprint
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="z_obs (tri)", origin="depth/z_obs"),
                    rrb.Spatial2DView(name="err: z_obs", origin="error/z_obs"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="z_fused", origin="depth/z_fused"),
                    rrb.Spatial2DView(name="err: z_fused", origin="error/z_fused"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="z_refined", origin="depth/z_refined"),
                    rrb.Spatial2DView(name="err: z_refined", origin="error/z_refined"),
                ),
                rrb.TimeSeriesView(name="d<1.25 (%)", origin="d125"),
                rrb.TimeSeriesView(name="Variance median", origin="var"),
                row_shares=[2, 2, 2, 1, 1],
            ),
            rrb.Spatial3DView(
                name="Point Cloud",
                origin="pointcloud",
                background=[0, 0, 0, 255],
                line_grid=rrb.LineGrid3D(visible=False),
                eye_controls=rrb.EyeControls3D(
                    kind=rrb.Eye3DKind.Orbital,
                    position=[-1.19, -11.96, -12.96],
                    look_target=[1.15, 0.05, 25.40],
                    eye_up=[0.0, -1.0, 0.0],
                ),
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )

    rr.init("ptc_depth", spawn=True)
    rr.send_blueprint(blueprint)

    # Load config
    import yaml
    config_path = PROJECT_ROOT / 'configs' / 'default.yaml'
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    inv_dir = data_path / 'inv_depth'
    if not inv_dir.exists():
        print("Error: inv_depth/ not found in data.")
        sys.exit(1)

    # Pipeline kwargs: YAML config + overrides
    pipe_kwargs = {k: v for k, v in cfg.items() if isinstance(v, (int, float, bool, str))}
    pipe_kwargs['verbose'] = True

    pipeline = PTCDepth(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, **pipe_kwargs)

    segmenter = None
    if args.segmentation:
        from ptc_depth.segmentation import EdgeAwareSegmentation
        segmenter = EdgeAwareSegmentation()
        print("Segmentation: enabled")

    print(f"Data: {data_path}")
    print(f"Frames: {start}-{n_frames-1}, Size: {W}x{H}")

    max_depth = float(cfg.get('max_depth', 80.0))
    vmin, vmax = 0, max_depth

    for i in range(start, n_frames):
        rr.set_time("frame", sequence=i)

        img_bgr = cv2.imread(image_paths[i])
        d_rel = np.load(str(inv_dir / f'{i:06d}.npy')).astype(np.float32)
        bl = baselines[min(i, len(baselines) - 1)]

        seg_labels = None
        if segmenter is not None:
            rel_depth = np.where(d_rel > 1e-7, 1.0 / (d_rel + 1e-7), 0)
            sky_mask = d_rel < 1e-7
            seg_labels = segmenter.segment(img_bgr, rel_depth, sky_mask)

        result = pipeline(img_bgr, d_rel, bl, seg_labels=seg_labels)

        gt = None
        if has_gt:
            gt_file = gt_dir / f'{i:06d}.npy'
            if gt_file.exists():
                gt = np.load(str(gt_file))

        z_obs = result.get('z_obs')
        z_fused = result.get('z_fused')
        z_refined = result.get('depth')

        # Depth maps (cropped to LiDAR valid region)
        r0, r1 = crop_r0, crop_r1
        if z_obs is not None and np.size(z_obs) > 0:
            rr.log("depth/z_obs", rr.Image(depth_to_colormap(z_obs, vmin, vmax)[r0:r1]))
        if z_fused is not None and np.size(z_fused) > 0:
            rr.log("depth/z_fused", rr.Image(depth_to_colormap(z_fused, vmin, vmax)[r0:r1]))
        if z_refined is not None and np.size(z_refined) > 0:
            rr.log("depth/z_refined", rr.Image(depth_to_colormap(z_refined, vmin, vmax)[r0:r1]))

        # Error maps (cropped, only if GT available)
        if gt is not None:
            if z_obs is not None and np.size(z_obs) > 0:
                rr.log("error/z_obs", rr.Image(error_to_colormap(z_obs, gt)[r0:r1]))
            if z_fused is not None and np.size(z_fused) > 0:
                rr.log("error/z_fused", rr.Image(error_to_colormap(z_fused, gt)[r0:r1]))
            if z_refined is not None and np.size(z_refined) > 0:
                rr.log("error/z_refined", rr.Image(error_to_colormap(z_refined, gt)[r0:r1]))

            # d<1.25
            for key, arr in [('tri', z_obs), ('refined', z_refined)]:
                if arr is not None:
                    met = compute_metrics(arr, gt)
                    if met:
                        rr.log(f"d125/{key}", rr.Scalars(met['d125']))
            if z_fused is not None:
                met = compute_metrics(z_fused, gt)
                if met:
                    rr.log("d125/fused", rr.Scalars(met['d125']))

        # Variance
        var = result.get('variance')
        if var is not None and np.size(var) > 0:
            v = var[np.isfinite(var) & (var > 0)]
            if len(v) > 0:
                rr.log("var/V_median", rr.Scalars(np.median(v)))

        # 3D Point Cloud
        if z_refined is not None:
            pts_est, col_est = depth_to_pointcloud(z_refined, img_bgr, fx, fy, cx, cy)
            if len(pts_est) > 0:
                rr.log("pointcloud/estimated", rr.Points3D(
                    pts_est, colors=col_est, radii=rr.Radius.ui_points(1.0)))

        if gt is not None:
            gt_valid = gt.copy()
            gt_valid[(gt <= 0) | (gt > 80) | ~np.isfinite(gt)] = 0
            pts_gt, _ = depth_to_pointcloud(gt_valid, None, fx, fy, cx, cy, grad_thresh=0)
            if len(pts_gt) > 0:
                import matplotlib
                _rainbow = matplotlib.colormaps['rainbow']
                RAINBOW_LUT = (_rainbow(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
                y_vals = pts_gt[:, 1]
                d_norm = np.clip((y_vals - 1.0) / (-8.0 - 1.0), 0, 1)
                gt_colors = RAINBOW_LUT[(d_norm * 255).astype(np.uint8)]
                rr.log("pointcloud/gt", rr.Points3D(
                    pts_gt, colors=gt_colors, radii=rr.Radius.ui_points(1.0)))

        if i % 50 == 0 or i == n_frames - 1:
            print(f"[{i+1:4d}/{n_frames}]")

    print("Done.")


if __name__ == '__main__':
    main()
