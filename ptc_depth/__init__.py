import numpy as np

try:
    import ptc_depth_cpp as _cpp
except ImportError:
    raise ImportError(
        "ptc_depth C++ module not found. Install with: pip install ."
    )


class PTCDepth:
    """PTC-Depth pipeline.

    Args:
        H, W: Image dimensions
        fx, fy, cx, cy: Camera intrinsics
        **kwargs: Override config parameters (see configs/default.yaml)
    """

    def __init__(self, H: int, W: int, fx: float, fy: float, cx: float, cy: float, **kwargs):
        config = _cpp.PTCDepthConfig()
        config.H, config.W = H, W
        config.fx, config.fy, config.cx, config.cy = fx, fy, cx, cy

        for key, val in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, val)

        self._config = config
        self._pipeline = _cpp.PTCDepth(config)

    def __call__(self, image: np.ndarray, inv_depth: np.ndarray, baseline: float,
                 seg_labels: np.ndarray = None,
                 external_R: np.ndarray = None,
                 external_t: np.ndarray = None,
                 flow: np.ndarray = None) -> dict:
        """Process one frame.

        Args:
            image: BGR or grayscale image (H, W, 3) or (H, W)
            inv_depth: Inverse depth from foundation model (H, W) float32
            baseline: Translation magnitude between frames (meters)
            seg_labels: Optional segmentation labels (H, W) int32
            external_R: Optional rotation matrix (3, 3) float64
            external_t: Optional translation vector (3,) float64
            flow: Optional optical flow (H, W, 2) float32. If not provided, computed internally via DIS.

        Returns:
            dict with 'depth' and 'variance'.
            If verbose=True: also 'pose', 'z_obs', 'z_fused'.
        """
        # Normalize inv_depth to [0, 1] if needed
        d = inv_depth.astype(np.float32)
        d_max = d.max()
        if d_max > 1.0:
            d = d / d_max

        labels = seg_labels.astype(np.int32) if seg_labels is not None else np.array([], dtype=np.int32)
        ext_R = external_R.astype(np.float64) if external_R is not None else np.array([], dtype=np.float64)
        ext_t = external_t.astype(np.float64) if external_t is not None else np.array([], dtype=np.float64)
        flow_arr = flow.astype(np.float32) if flow is not None else np.array([], dtype=np.float32)

        return self._pipeline.refine(
            image, d, float(baseline),
            labels, ext_R, ext_t, flow_arr,
        )

    def reset(self):
        """Reset pipeline state."""
        self._pipeline.reset()
