import numpy as np
import torch

def create_fourier_weights(signal_length, inverse=False, symmetry=False, real=False):  
    """
    symmetry: use that DFT of real signal is symmteric and use only half of transformed signal for inverse trafo
    """
    k_vals, n_vals = np.mgrid[0:signal_length, 0:signal_length]
    theta_vals = 2 * np.pi * k_vals * n_vals / signal_length
    sign = 1.0 if inverse else -1.0
    norm = 1/np.sqrt(signal_length)
    if symmetry:
        nyquist_k = signal_length//2
        if inverse:
            w_0 = np.ones(signal_length)[np.newaxis,:]
            w_nyquist = np.ones(signal_length)[np.newaxis,:]
            if real:
                return norm*np.vstack([w_0, 2*np.cos(theta_vals[1:nyquist_k]), w_nyquist, -2*np.sin(theta_vals[1:nyquist_k])])
            else:
                return norm*np.vstack([w_0, 2*np.exp(sign * 1j * theta_vals[1:nyquist_k]), w_nyquist])
        else:
            if real:
                return norm*np.hstack([np.cos(theta_vals[:,:nyquist_k+1]), -np.sin(theta_vals[:,1:nyquist_k])])
            else:
                return norm*np.exp(sign * 1j * theta_vals[:,:nyquist_k+1]) 
    else:
        if real:
            if inverse: 
                return norm*np.vstack([np.cos(theta_vals), -np.sin(theta_vals)])
            else:
                return norm*np.hstack([np.cos(theta_vals), -np.sin(theta_vals)])
        else:
            # inverse handled by sign
            return norm*np.exp(sign * 1j * theta_vals) 
def rectangle_window(m, width, signal_length, shift):
    """
    m: int, window shift (in units of time points)
    width: int, window width
    shift: int, fraction of window width by which window is shifted

    Caution: Perfect reconstruction condition fullfilled only for shift=(1,2)
    """
    w_nm = np.zeros(signal_length)
    w_nm[m:m+width] = 1/np.sqrt(shift)
    return w_nm


def halfsine_window(m, width, signal_length, shift=None):
    """
    m: int, window shift (in units of time points)
    width: int, window width
    shift: dummy parameter
    """
    w_nm = np.zeros(signal_length)
    w_nm[m:m+width] = np.sin(np.pi/width*(np.arange(width) + 0.5))
    return w_nm
WINDOWS = {"rectangle": rectangle_window, "halfsine": halfsine_window}
def create_window_mask(shift, width, signal_length, window_function):
    ms = np.arange(0, signal_length-width+1, width//shift)

    W_mn = [window_function(m, width, signal_length, shift)[np.newaxis] for m in ms]
    W_mn = np.concatenate(W_mn, axis=0)
    return W_mn.transpose((1,0))


def create_short_time_fourier_weights(signal_length, shift, window_width, window_shape, inverse=False, real=False, symmetry=False):
    assert window_shape in ("rectangle", "halfsine", "hann"), "Available window shapes: rectangle, halfsine"

    if window_shape=="rectangle":
        window_function = rectangle_window
    elif window_shape=="halfsine":
        window_function = halfsine_window
    #elif window_shape=="hann":
    #    window_function = hann_window

    W_mn = create_window_mask(shift, window_width, signal_length, window_function)
    
    DFT_kn = create_fourier_weights(signal_length, inverse=inverse, symmetry=symmetry, real=real)
    
    dtype = np.complex64 if not real else np.float16

    if inverse:
        W = W_mn.sum(axis=1)

        DFT_kn_m = np.zeros((W_mn.shape[1]*DFT_kn.shape[0], DFT_kn.shape[1]), dtype=dtype)
        for ki, i in enumerate(range(0, DFT_kn_m.shape[0],DFT_kn.shape[0])):
            DFT_kn_m[i:i+DFT_kn.shape[0]] = DFT_kn

        STDFT_mkn = DFT_kn_m / W.astype(dtype)
    else:
        STDFT_mkn = np.zeros((DFT_kn.shape[0], W_mn.shape[1]*DFT_kn.shape[1]), dtype=dtype)
        for m, k in enumerate(range(0, W_mn.shape[1]*DFT_kn.shape[1], DFT_kn.shape[1])):
            STDFT_mkn[:,k:k+DFT_kn.shape[1]] = DFT_kn * W_mn[:,m][:,np.newaxis]
    return STDFT_mkn
