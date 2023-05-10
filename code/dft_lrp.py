import numpy as np
import torch
import torch.nn as nn
import utils.dft_utils as dft_utils


class DFTLRP():
    def __init__(self, signal_length, precision=32, cuda=True, leverage_symmetry=False, window_shift=None, window_width=None, window_shape=None, create_inverse=True, create_transpose_inverse=True, create_forward=True, create_dft=True, create_stdft=True) -> None:
        """
        Class for Discrete Fourier transform in pytorch and relevance propagation through DFT layer.

        Args:
        signal_length: number of time steps in the signal
        leverage_symmetry: if True, levereage that for real signal the DF transformed signal is symmetric and compute only first half it
        cuda: use gpu
        precision: 32 or 16 for reduced precision with less memory usage

        window_width: width of the window for short time DFT
        window_shift: width/hopsize of window for short time DFT
        window_shape: shape of window for STDFT, options are 'rectangle' and 'halfsine'

        create_inverse: create weights for inverse DFT
        create_transpose: cretae weights for transpose inverse DFT (for DFT-LRP)
        create_forward: create weights for forward DFT 
        create_stdft: create weights for short time DFT
        create_stdft: create weights DFT
        """
        self.signal_length = signal_length
        self.nyquist_k = signal_length//2
        self.precision = precision
        self.cuda = cuda
        self.symmetry = leverage_symmetry
        self.stdft_kwargs = {"window_shift": window_shift, "window_width": window_width, "window_shape": window_shape}

        # create fourier layers
        # dft
        if create_dft:
            if create_forward:
                self.fourier_layer = self.create_fourier_layer(signal_length=self.signal_length, symmetry=self.symmetry, transpose=False, inverse=False, short_time=False, cuda=self.cuda, precision=self.precision)
            # inverse dft
            if create_inverse:
                self.inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length, symmetry=self.symmetry, transpose=False, inverse=True, short_time=False, cuda=self.cuda, precision=self.precision)
            # transpose inverse dft for dft-lrp
            if create_transpose_inverse:
                self.transpose_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length, symmetry=self.symmetry, transpose=True, inverse=True, short_time=False, cuda=self.cuda, precision=self.precision)

        if create_stdft:   
            # stdft
            if create_forward:
                self.st_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length, symmetry=self.symmetry, transpose=False, inverse=False, short_time=True, cuda=self.cuda, precision=self.precision, **self.stdft_kwargs)
            # inverse stdft
            if create_inverse:
                self.st_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length, symmetry=self.symmetry, transpose=False, inverse=True, short_time=True, cuda=self.cuda, precision=self.precision, **self.stdft_kwargs)
            # transpose inverse stdft for dft-lrp
            if create_transpose_inverse:
                self.st_transpose_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length, symmetry=self.symmetry, transpose=True, inverse=True, short_time=True, cuda=self.cuda, precision=self.precision, **self.stdft_kwargs)


    @staticmethod
    def _array_to_tensor(input: np.ndarray, precision: float, cuda: bool) -> torch.tensor:
        dtype = torch.float32 if precision==32 else torch.float16
        input = torch.tensor(input, dtype=dtype)
        if cuda:
            input = input.cuda()
        return input


    @staticmethod
    def create_fourier_layer(signal_length: int, inverse: bool, symmetry: bool, transpose: bool, short_time: bool, cuda: bool, precision:int, **stdft_kwargs):
        """
        Create linear layer with Discrete Fourier Transformation weights

        Args:
        inverse: if True, create weights for inverse DFT
        symmetry: if True, levereage that for real signal the DF transformed signal is symmetric and compute only first half it
        transpose: create layer with transposed DFT weights for explicit relevance propagation
        short_time: short time DFT
        cuda: use gpu
        precision: 32 or 16 for reduced precision with less memory usage
        """
        if short_time:
            weights_fourier = dft_utils.create_short_time_fourier_weights(signal_length, stdft_kwargs["window_shift"], stdft_kwargs["window_width"], stdft_kwargs["window_shape"], inverse=inverse, real=True, symmetry=symmetry)
        else:
            weights_fourier = dft_utils.create_fourier_weights(signal_length=signal_length, real=True, inverse=inverse, symmetry=symmetry)

        if transpose:
            weights_fourier = weights_fourier.T

        weights_fourier = DFTLRP._array_to_tensor(weights_fourier, precision, cuda).T

        n_in, n_out = weights_fourier.shape
        fourier_layer = torch.nn.Linear(n_in, n_out, bias=False)
        with torch.no_grad():
            fourier_layer.weight = nn.Parameter(weights_fourier)
        del weights_fourier
        
        if cuda:
            fourier_layer = fourier_layer.cuda()

        return fourier_layer        


    @staticmethod
    def reshape_signal(signal: np.ndarray, signal_length: int, relevance: bool, short_time: bool, symmetry: bool):
        """
        Restructure array from concatenation of real and imaginary parts to complex (if array contains signal) or sum of real and imaginary part (if array contains relevance). Additionallty, reshapes time-frequenc
        
        Args:
        relevance: True if array contains relevance, not signal itself
        symmetry: if True, levereage that for real signal the DF transformed signal is symmetric and compute only first half it
        short_time: short time DFT
        """
        bs = signal.shape[0]
        if symmetry:
            nyquist_k = signal_length//2
            if short_time:
                n_windows = signal.shape[-1] // signal_length
                signal = signal.reshape(bs, n_windows,signal_length)
            zeros = np.zeros_like(signal[...,:1])
            if relevance:
                signal = signal[...,:nyquist_k+1] + np.concatenate([zeros, signal[...,nyquist_k+1:], zeros], axis=-1)
            else:
                signal = signal [...,:nyquist_k+1] + 1j*np.concatenate([zeros, signal[...,nyquist_k+1:], zeros], axis=-1)            
        else:
            if short_time:
                n_windows = signal.shape[-1] // signal_length // 2
                signal = signal.reshape(bs, n_windows, signal_length*2)
            if relevance:
                signal = signal[..., :signal_length] + signal[..., signal_length:]
            else:
                signal = signal[...,:signal_length] + 1j*signal[...,signal_length:]
        return signal


    def fourier_transform(self, signal: np.ndarray, real:bool=True, inverse:bool=False, short_time:bool=False) -> np.ndarray:
        """
        Discrete Fourier transform (DFT) of signal in time (inverse=False) or inverse DFT of signal in frequency.
        
        Args:
        inverse: if True, perform inverse DFT
        short_time: if True, perform short time DFT
        real: if real, the output is split into real and imaginary parts of the signal in freq. domain y_k, i.e. (y_k^real, y_k^imag)
        """
        if inverse:
            if short_time:
                transform = self.st_inverse_fourier_layer
            else:
                transform = self.inverse_fourier_layer
        else:
            if short_time:
                transform = self.st_fourier_layer
            else:
                transform = self.fourier_layer

        signal = self._array_to_tensor(signal, self.precision, self.cuda)

        with torch.no_grad():
            signal_hat = transform(signal).cpu().numpy()
        
        # render y_k as complex number of shape (n_windows, signal_length) //2
        if not real and not inverse:
            signal_hat = self.reshape_signal(signal_hat, self.signal_length, relevance=False, short_time=short_time, symmetry=self.symmetry)
        return signal_hat

    
    def dft_lrp(self, relevance: np.ndarray, signal: np.ndarray, signal_hat=None, short_time=False, epsilon=1e-6, real=False) -> np.ndarray:
        """
        Relevance propagation thorugh DFT

        relevance: relevance in time domain
        signal: signal in time domain, same shape as relevance
        signal_hat: signal in frequency domain, if None it is computed using signal
        short_time: relevance propagation through short time DFT
        epsilon: small constant to stabilize denominantor in DFT-LRP
        real: if True, the signal_hat after DFT and correspondong relevance is split into real and imaginary parts of the signal in freq. domain y_k, i.e. (y_k^real, y_k^imag)
        """
        if short_time:
            transform = self.st_fourier_layer
            dft_transform = self.st_transpose_inverse_fourier_layer
        else:
            transform = self.fourier_layer
            dft_transform = self.transpose_inverse_fourier_layer
        
        signal = self._array_to_tensor(signal, self.precision, self.cuda)
        if signal_hat is None:
            signal_hat = transform(signal)
        
        relevance = self._array_to_tensor(relevance, self.precision, self.cuda)
        norm = signal + epsilon
        relevance_normed = relevance / norm

        relevance_normed = self._array_to_tensor(relevance_normed, self.precision, self.cuda)
        signal_hat = self._array_to_tensor(signal_hat, self.precision, self.cuda)
        with torch.no_grad():
            relevance_hat = dft_transform(relevance_normed)
            relevance_hat = signal_hat * relevance_hat

        relevance_hat = relevance_hat.cpu().numpy()
        signal_hat = signal_hat.cpu().numpy()

        # add real and imaginary part of relevance and signal
        if not real:
            signal_hat = self.reshape_signal(signal_hat, self.signal_length, relevance=False, short_time=short_time, symmetry=self.symmetry)
            relevance_hat = self.reshape_signal(relevance_hat, self.signal_length, relevance=True, short_time=short_time, symmetry=self.symmetry)
   
        return signal_hat, relevance_hat
