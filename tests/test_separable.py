import numpy as np
import time


N = [768, 1024]
n_trials = 10

u = np.random.randn(N[0])[:, np.newaxis]
v = np.random.randn(N[1])[np.newaxis, :]

x = u @ v

print("\n-- 2D FFT")

# separable FFT
start_time = time.perf_counter()
for _ in range(n_trials):
    U_fft = np.fft.fft(u, axis=0)
    V_fft = np.fft.fft(v, axis=1)
    X_sep = U_fft @ V_fft
proc_time_sep = (time.perf_counter() - start_time) / n_trials
print("Separable", proc_time_sep)

# Full 2D FFT
start_time = time.perf_counter()
for _ in range(n_trials):
    X_fft = np.fft.fft2(x)
proc_time_full = (time.perf_counter() - start_time) / n_trials
print("Full 2D", proc_time_full)

print("Factor reduction", proc_time_full / proc_time_sep)
assert np.allclose(X_sep, X_fft)


"""
Using separable in ASM, just when computing FT of u_in
"""
print("\n-- ASM")

from waveprop.rs import angular_spectrum

N = 512  # number of grid points per size
L = 1e-2  # total size of grid
wv = 1e-6  # wavelength
dz = 1  # distance [m]
d1 = L / N  # source-plane grid spacing

start_time = time.perf_counter()
for _ in range(n_trials):
    u_out_asm_sep = angular_spectrum(u_in_x=v, u_in_y=u, wv=wv, d1=d1, dz=dz)[0]
proc_time_sep = (time.perf_counter() - start_time) / n_trials
print("Separable", proc_time_sep)

start_time = time.perf_counter()
for _ in range(n_trials):
    u_out_asm = angular_spectrum(u_in=x, wv=wv, d1=d1, dz=dz)[0]
proc_time_full = (time.perf_counter() - start_time) / n_trials
print("ASM full", proc_time_full)

print("Factor reduction", proc_time_full / proc_time_sep)
assert np.allclose(u_out_asm_sep, u_out_asm)

"""
Using separable for Fresnel
"""

print("\n-- Fresnel")

from waveprop.fresnel import fresnel_conv

N = 512  # number of grid points per size
L = 1e-2  # total size of grid
wv = 1e-6  # wavelength
dz = 1  # distance [m]
d1 = L / N  # source-plane grid spacing

start_time = time.perf_counter()
for _ in range(n_trials):
    u_out_asm_sep = fresnel_conv(u_in_x=v, u_in_y=u, wv=wv, d1=d1, dz=dz)[0]
proc_time_sep = (time.perf_counter() - start_time) / n_trials
print("Separable", proc_time_sep)

start_time = time.perf_counter()
for _ in range(n_trials):
    u_out_asm = fresnel_conv(u_in=x, wv=wv, d1=d1, dz=dz)[0]
proc_time_full = (time.perf_counter() - start_time) / n_trials
print("Fresnel full", proc_time_full)

print("Factor reduction", proc_time_full / proc_time_sep)
assert np.allclose(u_out_asm_sep, u_out_asm)


"""
Using separable for Fraunhofer
"""
