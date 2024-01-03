import torch
import torch.nn as nn
import numpy as np
import math
import torchaudio

def hamming_window(window_length:int, alpha=0.54, beta=0.46):
    pass
    """
    https://runebook.dev/es/docs/pytorch/generated/torch.hamming_window
    w[n] = alpha - beta*cos(2*p*n/(N-1))
    """
    n = np.arange(window_length, dtype=np.float32)
    window = alpha - beta*np.cos(2*np.pi*n/(window_length-1))
    return window

def hann_window(window_length:int):
    """
    w[n] = 1/2*(1 - cos(2*pi*n/(N - 1))) = sin(pi*n/(N-1))^2
    """
    n = np.arange(window_length, dtype=np.float32)
    window = np.sin(np.pi*n/(window_length - 1))
    window = np.power(window, 2)
    return window   

def stft_torch(input, n_fft, hop_length=None, win_length=None, window=None, center=False):
    """
    input : (n_samples,)
    output : (F, T, 2)
    """
    if center:
        input = np.pad(input, (n_fft//2, n_fft//2))
    num_frames = (input.size - win_length) // hop_length + 1
    output = np.zeros((num_frames, n_fft//2+1, 2), np.float32)
    fft_input = np.zeros(n_fft, np.float32)
    for i in range(num_frames):
        fft_input = input[i*hop_length:i*hop_length+win_length].copy()
        #import pdb; pdb.set_trace()
        fft_input *= window
        outdata = torch.fft.fft(torch.from_numpy(fft_input)).numpy()
        output[i,:,0] = outdata[0:n_fft//2+1].real
        output[i,:,1] = outdata[0:n_fft//2+1].imag
    output = np.transpose(output, (1,0,2))
    return output

def stft(input, n_fft, hop_length=None, win_length=None, window=None):
    """
    input : (n_samples,)
    output : (F, T, 2)
    """
    input = np.pad(input, (win_length-hop_length, 0))
    num_frames = (input.size - win_length) // hop_length + 1
    output = np.zeros((num_frames, n_fft//2+1, 2), np.float32)
    fft_input = np.zeros(n_fft, np.float32)
    for i in range(num_frames):
        fft_input = input[i*hop_length:i*hop_length+win_length].copy()
        fft_input *= window
        outdata = np.fft.rfft(fft_input)
        #import pdb; pdb.set_trace()
        output[i,:,0] = outdata.real
        output[i,:,1] = outdata.imag
    output = np.transpose(output, (1,0,2))
    return output

def istft(input, n_fft, hop_length=None, win_length=None, window=None):
    pass 
    """
    input: (F, T, 2)
    output: (n_samples,)
    win_length = 8, hop_size = 1
    0   1   2   3
        0   1   2   3
            0   1   2   3
                0   1   2   3
    0   10  210 3210  
    """
    freq_size, num_frames, _ = input.shape 
    assert freq_size == n_fft//2+1
    overlap = np.zeros(win_length, dtype=np.float32)
    wsum = np.zeros(win_length, dtype=np.float32)
    output = np.zeros(num_frames*hop_length, dtype=np.float32)
    for i in range(num_frames):
        ifftdata = np.fft.irfft(input[:,i,0]+input[:,i,1]*1j)
        ifftdata *= window
        overlap[0:(win_length - hop_length)] = overlap[hop_length:]
        overlap[(win_length - hop_length):] = np.zeros(hop_length, dtype=np.float32)
        overlap += ifftdata
        wsum[0:(win_length - hop_length)] = wsum[hop_length:]
        wsum[(win_length - hop_length):] = np.zeros(hop_length, dtype=np.float32)
        wsum += window**2
        output[i*hop_length:i*hop_length+hop_length] = overlap[0:hop_length] / wsum[0:hop_length]
    return output

def test_window():
    # hamming_window
    window_length = 512
    w1 = torch.hamming_window(window_length, dtype=torch.float32)
    w2 = hamming_window(window_length)
    mae = np.mean(w1.numpy() - w2)
    print("hamming_window mae:", mae)
    assert mae < 1.0e-3
    # hann_window 
    w1 = torch.hann_window(window_length, dtype=torch.float32)
    w2 = hann_window(window_length)
    mae = np.mean(w1.numpy() - w2)
    print("hann_window mae:", mae)
    assert mae < 1.0e-3 

def test_stft():
    sample_rate = 16000
    n_fft = 512
    frame_length = 512
    hop_length = 128
    x = np.random.rand(1*sample_rate)
    w1 = hann_window(n_fft)
    #w1 = np.ones((n_fft), dtype=np.float32)
    y1 = torch.stft(torch.from_numpy(x), n_fft, hop_length, frame_length, torch.from_numpy(w1), 
                    center=False, return_complex=False)  
    #y1 = torch.permute(y1, (1,0,2))
    y2 = stft_torch(x, n_fft, hop_length, frame_length, w1)
    mae = np.mean(y1.numpy() - y2)
    print("stft mae:", mae)
    assert mae < 1.0e-3 

    # ###
    # yk = stft(x, n_fft,hop_length,frame_length, np.ones(frame_length))
    # x2 = istft(yk, n_fft,hop_length,frame_length, np.ones(frame_length))
    # import pdb; pdb.set_trace()


def test_stft_center():
    """"""
    # num_smaples = 16
    # window_length = 8
    # hop_length = 2
    # fft_size = 8
    num_smaples = 16000
    window_length = 512
    hop_length = 128
    fft_size = 512
    x = torch.arange(num_smaples)
    x = torch.arange(num_smaples).type(torch.float32)
    window = hamming_window(window_length) #np.ones(window_length)
    window_th = torch.from_numpy(window) #torch.ones(window_length)
    # torch stft/istft
    x_pad = torch.cat((torch.zeros(fft_size//2, dtype=torch.float32), x, torch.zeros(fft_size//2,dtype=torch.float32)))
    x_pad_l = torch.cat((torch.zeros(fft_size - hop_length, dtype=torch.float32), x))
    yk1 = torch.stft(x, fft_size,hop_length,window_length,window_th, center=True, pad_mode="constant")
    yk2 = torch.stft(x_pad, fft_size,hop_length,window_length,window_th,center=False)
    yk3 = torch.stft(x, fft_size,hop_length,window_length,window_th,center=False)
    yk4 = torch.stft(x_pad_l, fft_size,hop_length,window_length,window_th,center=False)
    mae = (yk1-yk2).abs().mean().item()
    print("torch stft pad mae:", mae)
    assert mae <= 1.0e-9
    # torch ifft
    x1 = torch.istft(yk1, fft_size,hop_length,window_length,window_th)
    print(x1)
    print(x)
    print(x - x1)
    mae = (x1-x).abs().mean().item()
    print("torch stft pad mae:", mae)
    assert mae <= 1.0e-3

    x2 = torch.istft(yk2, fft_size,hop_length,window_length,window_th)
    x4 = torch.istft(yk4, fft_size,hop_length,window_length,window_th) # pad left 4 points.
    # 
    yk5 = stft_torch(x.numpy(), fft_size,hop_length,window_length, window, center=True)
    yk6 = stft_torch(x_pad.numpy(), fft_size,hop_length,window_length, window)
    yk7 = stft_torch(x.numpy(), fft_size,hop_length,window_length, window)
    assert np.abs(yk5-yk1.numpy()).mean() <= 1.0e-6
    assert np.abs(yk6-yk2.numpy()).mean() <= 1.0e-6
    assert np.abs(yk7-yk3.numpy()).mean() <= 1.0e-6
    # stft/istft
    yk10 = stft(x, fft_size,hop_length,window_length, window)
    mae = np.abs(yk10 - yk4.numpy()).mean()
    max = np.abs(yk10 - yk4.numpy()).max()
    print("stft mae", mae, "max", max)
    #import pdb; pdb.set_trace()
    assert mae <= 1.0e-2  #TODO:
    x12 = istft(yk10, fft_size,hop_length,window_length, window)
    mae = np.abs(x12[window_length-hop_length:] - x[:num_smaples-(window_length-hop_length)].numpy()).mean() 
    print("istft mae", mae)
    assert mae <= 1.0e-3
    #import pdb; pdb.set_trace()


if __name__ == "__main__":
    test_window()
    test_stft()
    test_stft_center()