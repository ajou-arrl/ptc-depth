import numpy as np
import cv2


class EdgeDetector:
    def __init__(self, ksize=7):
        self.ksize = ksize

    def __call__(self, depth, image=None):
        x = np.asarray(depth, np.float32)
        x[~np.isfinite(x)] = 0.0
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=self.ksize)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=self.ksize)
        mag = np.sqrt(gx * gx + gy * gy)

        # Intersect with color edge: min(depth_edge, color_edge)
        if image is not None:
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32)
            cgx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.ksize)
            cgy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=self.ksize)
            color_mag = np.sqrt(cgx * cgx + cgy * cgy)
            d_max = mag.max() + 1e-6
            c_max = color_mag.max() + 1e-6
            mag = np.minimum(mag / d_max, color_mag / c_max) * d_max

        return mag


class EdgeAwareSegmentation:
    """Segmentation using depth edges + RGB Lab for guide construction."""

    def __init__(self, seg_params=None,
                 wrgb=0.6, wx=0.15, wgrad=0.2, grad_power=0.22, seg_down=0.3):
        self.edge = EdgeDetector()
        self.seg_params = seg_params or {"scale": 150, "sigma": 0.0, "min_size": 1000}
        self.wrgb = wrgb
        self.wx = wx
        self.wgrad = wgrad
        self.grad_power = grad_power
        self.seg_down = seg_down

    def _build_guide(self, rgb_or_gray, depth, edge_map, sky_mask):
        img = rgb_or_gray
        is_gray = img.ndim == 2

        if is_gray:
            # Grayscale (thermal etc.): use normalized intensity instead of Lab A/B
            intensity = img.astype(np.float32) / 255.0
            chans = [
                self.wrgb * intensity,
                self.wx * depth.astype(np.float32),
                self.wgrad * (edge_map.astype(np.float32) ** self.grad_power),
            ]
        else:
            lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            A = (lab[..., 1] - 128.0) / 128.0
            B = (lab[..., 2] - 128.0) / 128.0
            chans = [
                self.wrgb * A,
                self.wrgb * B,
                self.wx * depth.astype(np.float32),
                self.wgrad * (edge_map.astype(np.float32) ** self.grad_power),
            ]
        guide = np.stack(chans, axis=-1).astype(np.float32)

        non_sky = ~sky_mask
        if np.any(non_sky):
            guide[sky_mask] = np.median(guide[non_sky], axis=0)
        else:
            guide[sky_mask] = 0

        return guide

    def segment(self, image, depth, sky_mask=None):
        import warnings
        from skimage.segmentation import felzenszwalb
        warnings.filterwarnings('ignore', message='.*multichannel 2d image.*')

        H, W = depth.shape[:2]
        if sky_mask is None:
            sky_mask = np.zeros((H, W), dtype=bool)

        edge_map = self.edge(depth, image)
        guide = self._build_guide(image, depth, edge_map, sky_mask)

        if 0.0 < self.seg_down < 1.0:
            new_w = max(1, int(W * self.seg_down))
            new_h = max(1, int(H * self.seg_down))
            guide_small = cv2.resize(guide, (new_w, new_h), interpolation=cv2.INTER_AREA)
            labels_small = felzenszwalb(guide_small, **self.seg_params).astype(np.int32)
            labels = cv2.resize(labels_small, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        else:
            labels = felzenszwalb(guide, **self.seg_params).astype(np.int32)

        return labels
