import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import audio_py
import math

def test_fbank_diff():
    """"""
    num_smaples = 16000
    sample_rate = 16000
    num_mel_bins = 80
    frame_length = 25
    frame_shift = 10
    dither = 0.0
    # load wav
    # wav_file = "./bin/dca6dc35f6c3c45380b88015027fb4d5.wav"
    # waveform, sample_rate = torchaudio.load(wav_file)
    # waveform = waveform * (1 << 15)
    # rand 
    # waveform = torch.zeros((1, num_smaples),dtype=torch.float32)
    # for i in range(waveform.shape[1]):
    #     waveform[0][i] = i%255
    waveform = torch.randn((1, num_smaples),dtype=torch.float32)
    
    mat = kaldi.fbank(waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    energy_floor=0.0,
                    sample_frequency=sample_rate,
                    window_type="hamming")
    #print(waveform[0,8000:8000+10])
    #print(mat[0:2])
    waveform = waveform.numpy()
    mat = mat.numpy()

    # wekws fbank
    input_data = waveform.reshape(-1)  
    #print(input_data[8000:8000+10])
    frame_length = sample_rate*25//1000  # must be int
    frame_shift = sample_rate*10//1000
    mylib = audio_py.FBank(num_mel_bins, sample_rate, frame_length, frame_shift, "./lib/libaudio_so.so")
    output_data = mylib.process(input_data)
    output_data = output_data.reshape((-1, num_mel_bins))
    #print(output_data[0:2])
    # stat diff
    mat = mat.reshape(-1)
    output_data = output_data.reshape(-1)
    size = mat.size if mat.size < output_data.size else output_data.size
    diff = np.abs(mat[0:size] - output_data[0:size])
    mae = np.mean(diff)
    max = np.max(diff)
    print("mae:", mae, "max:", max)
    assert mae < 1.0e-5 and max < 1.0e-3

"""
Y(u) = C(u)*sqrt(2/N)*sigma(X(m)*cos(((2*m+1)*u*pi)/(2*N)), m=0, m=M-1), u=0,1,...,N-1
C(u) =  1/sqrt(2)   u = 0  正交化因子
        1           u = 0 
"""
def _get_dct_matrix(num_ceps: int, num_mel_bins: int) -> torch.Tensor:
    # returns a dct matrix of size (num_mel_bins, num_ceps)
    # size (num_mel_bins, num_mel_bins)
    dct_matrix = torchaudio.functional.create_dct(num_mel_bins, num_mel_bins, "ortho")
    # kaldi expects the first cepstral to be weighted sum of factor sqrt(1/num_mel_bins)
    # this would be the first column in the dct_matrix for torchaudio as it expects a
    # right multiply (which would be the first column of the kaldi's dct_matrix as kaldi
    # expects a left multiply e.g. dct_matrix * vector).
    dct_matrix[:, 0] = math.sqrt(1 / float(num_mel_bins))
    dct_matrix = dct_matrix[:, :num_ceps]
    return dct_matrix


def test_mfcc_2():
    """"""
    num_smaples = 16000
    sample_rate = 16000
    num_mel_bins = 80
    frame_length = 25
    frame_shift = 10
    dither = 0.0
    num_ceps = 64
    # load wav
    # wav_file = "./bin/dca6dc35f6c3c45380b88015027fb4d5.wav"
    # waveform, sample_rate = torchaudio.load(wav_file)
    # waveform = waveform * (1 << 15)
    # rand 
    # waveform = torch.zeros((1, num_smaples),dtype=torch.float32)
    # for i in range(waveform.shape[1]):
    #     waveform[0][i] = i%255
    waveform = torch.randn((1, num_smaples),dtype=torch.float32)
    #bank + dct
    feature = kaldi.fbank(waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    energy_floor=0.0,
                    sample_frequency=sample_rate,
                    window_type="hamming",
                    # subtract_mean=False,  
                    # use_log_fbank=True,
                    # use_power=True,
                    )
    # size (num_mel_bins, num_ceps)
    dct_matrix = _get_dct_matrix(num_ceps, num_mel_bins)
    # size (m, num_ceps)
    mat = feature.matmul(dct_matrix)
    print("waveform", waveform[0:10])
    print("feature", feature[0])
    print("mat", mat[0]) 
    #mfcc
    mat_mfcc = kaldi.mfcc(waveform,
                    num_mel_bins=num_mel_bins,
                    num_ceps=num_ceps,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    energy_floor=0.0,
                    sample_frequency=sample_rate,
                    window_type="hamming")

    mat = mat_mfcc.numpy()
    mat_mfcc = mat_mfcc.numpy()

    diff = np.abs(mat - mat_mfcc)
    mae = np.mean(diff)
    max = np.max(diff)
    print("mae:", mae, "max:", max)
    assert mae < 1.0e-5 and max < 1.0e-3

if __name__ == "__main__":
    #test_fbank_diff()
    test_mfcc_2()