import numpy as np 
import math

import torch
import functional

def calc_snr(waveform, noise):
    """
    return signal noise ratio.
    """
    assert len(waveform.shape) == len(noise.shape)
    assert waveform.shape[-1] == noise.shape[-1]
    L = waveform.shape[-1]
    energy_signal = (waveform ** 2).sum()
    energy_noise = (noise ** 2).sum()
    original_snr_db = 10 * (math.log10(energy_signal) - math.log10(energy_noise))
    return original_snr_db

def add_noise(waveform, noise, snr):
    """
    scales and adds noise to waveform.
    """
    assert waveform.ndim - 1 == noise.ndim - 1 and waveform.ndim - 1 == snr.ndim

    L = waveform.shape[-1]
    energy_signal = (waveform ** 2).sum(-1)
    energy_noise = (noise ** 2).sum(-1)
    original_snr_db = 10 * (np.log10(energy_signal) - np.log10(energy_noise))
    scale = 10 ** ((original_snr_db - snr)/20.0)
   
    #scale noise
    scaled_noise = np.expand_dims(scale, -1) * noise

    return waveform + scaled_noise # (*, L)



def test_add_noise():

    waveform = np.random.rand(1,16000) * 100
    noise = np.random.rand(1,16000)
    snr = np.zeros((1))
    original_snr_db = calc_snr(waveform, noise)

    waveform_noised = add_noise(waveform, noise, snr)

    waveform_noised_th = functional.add_noise(torch.from_numpy(waveform), torch.from_numpy(noise), torch.from_numpy(snr))

    mae = np.abs(waveform_noised - waveform_noised_th.numpy()).mean()
    print("mae", mae)

if __name__ == "__main__":
    test_add_noise()
