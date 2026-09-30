"""Tests for the NumPy propagation functions in waveprop.rs."""
import warnings

import numpy as np

from waveprop.rs import angular_spectrum_np, fft_di


def test_angular_spectrum_np_no_pad_runs_without_error():
    """`pad=False` must run successfully and preserve the input shape."""
    u = np.ones((64, 64), dtype=complex)
    u_out, x2, y2 = angular_spectrum_np(u, wv=532e-9, d1=2e-6, dz=1e-3, pad=False)
    assert u_out.shape == u.shape
    assert x2.shape == (1, u.shape[1])
    assert y2.shape == (u.shape[0], 1)


def test_angular_spectrum_np_no_pad_preserves_plane_wave():
    """For a constant (plane-wave) input with bandlimit disabled, the unpadded
    branch should reproduce a constant field — proving it captured the correct
    Ny/Nx from the original (unpadded) shape."""
    u = np.ones((32, 32), dtype=complex)
    out_no_pad, x2, y2 = angular_spectrum_np(
        u, wv=632e-9, d1=4e-6, dz=5e-4, pad=False, bandlimit=False
    )
    assert out_no_pad.shape == u.shape
    # Plane wave stays plane wave: magnitude uniform, ~1.
    assert np.allclose(np.abs(out_no_pad), 1.0, atol=1e-3)
    # Output coordinates must reflect the original (Ny, Nx), not the padded size.
    assert x2.shape == (1, 32)
    assert y2.shape == (32, 1)


def test_fft_di_preserves_complex_phase():
    """A complex input field must propagate without losing its imaginary part."""
    u = np.full((32, 32), 1.0 + 2.0j, dtype=complex)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        S, _, _ = fft_di(u, wv=532e-9, d1=2e-6, dz=1e-3)

    complex_msgs = [
        str(w.message) for w in caught if "complex" in str(w.message).lower()
    ]
    assert complex_msgs == [], f"Unexpected complex warning(s): {complex_msgs}"

    assert S.shape == (32, 32)
    assert np.iscomplexobj(S)
    assert np.max(np.abs(S.imag)) > 1e-6


if __name__ == "__main__":
    test_angular_spectrum_np_no_pad_runs_without_error()
    test_angular_spectrum_np_no_pad_preserves_plane_wave()
    test_fft_di_preserves_complex_phase()
    print("all rs propagation tests passed")
