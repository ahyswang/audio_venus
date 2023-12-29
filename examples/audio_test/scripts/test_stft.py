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

def stft(input, n_fft, hop_length=None, win_length=None, window=None):
    """
    input : (n_samples,)
    output : (F, T, 2)
    """
    num_frames = (input.size - win_length) // hop_length + 1
    output = np.zeros((num_frames, n_fft//2+1, 2), np.float32)
    fft_input = np.zeros(n_fft, np.float32)
    for i in range(num_frames):
        fft_input = input[i*hop_length:i*hop_length+win_length].copy()
        fft_input *= window
        outdata = torch.fft.fft(torch.from_numpy(fft_input)).numpy()
        output[i,:,0] = outdata[0:n_fft//2+1].real
        output[i,:,1] = outdata[0:n_fft//2+1].imag
    output = np.transpose(output, (1,0,2))
    return output

def istft(input, n_fft, hop_length=None, win_length=None, window=None):
    pass

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
    y2 = stft(x, n_fft, hop_length, frame_length, w1)
    mae = np.mean(y1.numpy() - y2)
    print("stft mae:", mae)
    assert mae < 1.0e-3 
    import pdb; pdb.set_trace()

    source = torch.zeros((1,16000))
    print(source.shape)
    stft_data = torch.stft(source, 512, 128, 512, torch.hann_window(512))
    print(stft_data.shape)
    istft_data = torch.istft(stft_data, 512, 128, 512, torch.hann_window(512))
    print(istft_data.shape)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_window()
    test_stft()