"""Pixel-wise check of a method's render against the dense ground-truth.

Usage:
  check_render_against_reference.py <method.npz>

The method's npz must contain at least an `image` array of shape (H, W, 3)
matching the reference shape. Optionally a `hit` mask and `t_hit` map for
deeper diagnostics.

Exit code: 0 if max-pixel error ≤ 0.05 (≈ 13/255) AND hit-mask agreement
≥ 99.5%, else 1. Tunneling typically shows up as flipped hit-mask pixels
on silhouettes — these get reported separately.
"""
import sys
import numpy as np

REF = '/system/user/volkmann/implicit_surfaces_project/assets/reference_renders/dense_gt/image.npz'

def main():
    if len(sys.argv) != 2:
        print("usage: check_render_against_reference.py <method.npz>")
        sys.exit(2)
    method_path = sys.argv[1]

    ref = np.load(REF)
    m = np.load(method_path)

    img_ref = ref['image']
    img_m = m['image']
    if img_ref.shape != img_m.shape:
        print(f"FAIL  shape mismatch: ref {img_ref.shape} vs method {img_m.shape}")
        sys.exit(1)

    diff = np.abs(img_ref - img_m)
    l_inf = float(diff.max())
    rmse = float(np.sqrt((diff ** 2).mean()))
    pct_off_1 = 100.0 * (diff.max(axis=-1) > 1/255).mean()    # > 1 LSB
    pct_off_5 = 100.0 * (diff.max(axis=-1) > 5/255).mean()    # > 5 LSB

    print(f"image:     L∞={l_inf:.4f}  RMSE={rmse:.4f}  "
          f"px>1lsb={pct_off_1:.3f}%  px>5lsb={pct_off_5:.3f}%")

    hit_match = None
    if 'hit' in ref.files and 'hit' in m.files:
        hit_diff = ref['hit'] != m['hit']
        n_diff = int(hit_diff.sum())
        hit_match = 100.0 * (1 - hit_diff.mean())
        print(f"hit mask:  agree {hit_match:.3f}%  ({n_diff:,} differing pixels)")

    if 't_hit' in ref.files and 't_hit' in m.files:
        # only meaningful where both hit
        if 'hit' in ref.files and 'hit' in m.files:
            both = ref['hit'] & m['hit']
            if both.any():
                td = np.abs(ref['t_hit'] - m['t_hit'])[both]
                print(f"t_hit:     where both hit — L∞={td.max():.4f}  "
                      f"mean={td.mean():.4f}  >1e-2: {100*(td>1e-2).mean():.3f}%")

    ok = (l_inf <= 0.05) and (hit_match is None or hit_match >= 99.5)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
