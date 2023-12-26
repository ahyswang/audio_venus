import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import audio_py

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

if __name__ == "__main__":
    test_fbank_diff()