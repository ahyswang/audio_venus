import numpy as np 
import math 
import torch
import torchaudio
import functional

def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    assert mel_scale in [ "htk"]
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq/700.0))
    
def _mel_to_hz(mels: float, mel_scale: str = "htk") -> float:
    assert mel_scale in [ "htk"]
    if mel_scale == "htk":
        return 700.0*(10.0**(mels * 2595) - 1.0)

if __name__ == "__main__":

    # mels = np.arange(256, dtype=np.float32)
    # hzs = np.zeros(256, dtype=np.float32)
    # print(mels)
    # for i,m in  enumerate(mels):
    #     hzs[i] = _mel_to_hz(m)
    # print(hzs)

    n_freqs = 640
    f_min = 0
    f_max = 8000
    n_mels = 256
    sample_rate = 16000
    mat = functional.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate)

    freq, hop_length = 1025, 512
    # (channel, freq, time)
    complex_specgrams = torch.randn(2, freq, 300, dtype=torch.cfloat)
    rate = 1.3 # Speed up by 30%
    phase_advance = torch.linspace(0, math.pi * hop_length, freq)[..., None]
    import pdb; pdb.set_trace()
    x = functional.phase_vocoder(complex_specgrams, rate, phase_advance)
    x.shape # with 231 == ceil(300 / 1.3)
    torch.Size([2, 1025, 231])



"""
int anker_erb_amp2bank_new(anker_erb_t *p_this, void* input, void* out, unsigned int channel_num)
{
	int* p_band_segment_idx = p_this->p_band_segment_idx_;
	int  n_bank = p_this->n_filters_;
	int  n_freq = p_this->n_freq_;

// 	int* p_band_segment_idx = (int *)MALLOCA32(p_share, p_this->n_filters_*sizeof(int));
// 	memcpy(p_band_segment_idx, p_this->p_band_segment_idx_, p_this->n_filters_*sizeof(int)); 
	
#if DUMPER_ENABLE && DUMPER_ERB
	DUMPER_WRITE_FLOAT32(erb_amp2bank_input, input, channel_num*n_freq);
#endif

	for (int k = 0; k < channel_num; k++)
	{
		float *p_freq = (float *)input + k*n_freq;
		float *p_bank = (float *)out + k*n_bank;
	
#if 1
		memcpy(p_bank, p_freq, p_this->band_start_ * sizeof(float));
		memset(p_bank + p_this->band_start_, 0, (n_bank - p_this->band_start_) * sizeof(float));
		for (int i = p_this->band_start_; i < n_bank - 1; i++)
		{
			int band_size = p_band_segment_idx[i + 1] - p_band_segment_idx[i];
			for (int j = 0; j < band_size; j++)
			{
				float frac = (float)(j) / band_size;
				float tmp = p_freq[p_band_segment_idx[i] + j];
				p_bank[i] += (1.0f - frac) * tmp;
				p_bank[i + 1] += frac * tmp;
			}
		}
#else 
		memset(p_bank, 0, n_bank * sizeof(float));

		for (int i = 0; i < n_bank-1; i++)
		{
			int band_size = p_band_segment_idx[i+1] - p_band_segment_idx[i];
#if ENABLE_ERB_OPT
			if (band_size == 1)
			{
				p_bank[i] = p_freq[p_band_segment_idx[i]];
			}
			else
			{
				for (int j = 0; j < band_size; j++)
				{
					float frac = (float)(j) / band_size;
					float tmp = p_freq[p_band_segment_idx[i] + j];
					p_bank[i] += (1.0f - frac) * tmp;
					p_bank[i + 1] += frac * tmp;
				}
			}
#else 
			for (int j = 0; j < band_size; j++)
			{
				float frac = (float)(j)/band_size;
				float tmp = p_freq[p_band_segment_idx[i] + j];
				p_bank[i] += (1.0f - frac) * tmp;
				p_bank[i+1] += frac * tmp;
			}
#endif 
		}
#endif 
		// p_bank[0] *= 2;
		// p_bank[n_bank-1] *= 2;
	}
#if DUMPER_ENABLE && DUMPER_ERB
	DUMPER_WRITE_FLOAT32(erb_amp2bank_output, out, channel_num*n_bank);
#endif 

	return 0;
}

"""

"""
int anker_erb_bank2amp_new(anker_erb_t *p_this, void* input, void* out, unsigned int channel_num)
{
	int* p_band_segment_idx = p_this->p_band_segment_idx_;
	int  n_bank = p_this->n_filters_;
	int  n_freq = p_this->n_freq_;

// 	int* p_band_segment_idx = (int *)MALLOCA32(p_share, p_this->n_filters_*sizeof(int));
// 	memcpy(p_band_segment_idx, p_this->p_band_segment_idx_, p_this->n_filters_*sizeof(int));
#if DUMPER_ENABLE && DUMPER_ERB
	DUMPER_WRITE_FLOAT32(erb_bank2amp_input, input, channel_num*n_bank);
#endif 

	for (int k = 0; k < channel_num; k++)
	{
		float *p_bank = (float *)input + k*n_bank;
		float *p_freq = (float *)out + k*n_freq;

#if 1
		memcpy(p_freq, p_bank, p_this->band_start_ * sizeof(float));
		memset(p_freq + p_this->band_start_, 0, (n_freq - p_this->band_start_) * sizeof(float));

		for (int i = p_this->band_start_; i < n_bank - 1; i++)
		{
			int band_size = p_band_segment_idx[i + 1] - p_band_segment_idx[i];
			for (int j = 0; j < band_size; j++)
			{
				float frac = (float)(j) / band_size;
				p_freq[p_band_segment_idx[i] + j] = (1 - frac)*p_bank[i] + frac*p_bank[i + 1];
			}
		}
#else 
		memset(p_freq, 0, n_freq * sizeof(float));
		
		for (int i = 0; i < n_bank-1; i++)
		{
			int band_size = p_band_segment_idx[i+1] - p_band_segment_idx[i];
#if ENABLE_ERB_OPT
			if (band_size == 1)
			{
				p_freq[p_band_segment_idx[i]] = p_bank[i];
			}
			else
			{
				for (int j = 0; j < band_size; j++)
				{
					float frac = (float)(j) / band_size;
					p_freq[p_band_segment_idx[i] + j] = (1 - frac)*p_bank[i] + frac*p_bank[i + 1];
				}
			}
#else 
			for (int j = 0; j < band_size; j++)
			{
				float frac = (float)(j)/band_size;
				p_freq[p_band_segment_idx[i]+j] = (1-frac)*p_bank[i] + frac*p_bank[i + 1];
			}
#endif
		}
#endif 
	}
#if DUMPER_ENABLE && DUMPER_ERB
	DUMPER_WRITE_FLOAT32(erb_bank2amp_output, out, channel_num*n_freq);
#endif 

	return 0;
}
"""