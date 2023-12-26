#include "fbank_c.h"
#include "define.h"
#include "fft.h"

typedef struct tag_fbank_t
{
    int num_bins_;
    int sample_rate_;
    int frame_length_, frame_shift_;
    int fft_points_;
    int use_log_;
    int remove_dc_offset_;
    float* center_freqs_;
    int* bins_starts_; // bank => weight starts
    int* bins_indexs_; // bank => freq bin index
    int* bins_sizes_; // bank => freq bin size
    float* bins_weights_;
    float* hamming_window_;
    // std::default_random_engine generator_;
    // std::normal_distribution<float> distribution_;
    float dither_;

    // bit reversal table
    int* bitrev_;
    // trigonometric function table
    float* sintbl_;

    // tmp memory
    float* fft_real_tmp_;
    float* fft_img_tmp_;
    float* power_tmp_;
    float* data_tmp_;
}fbank_t;

static inline float InverseMelScale(float mel_freq) {
    return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
  }

  static inline float MelScale(float freq) {
    return 1127.0f * logf(1.0f + freq / 700.0f);
  }

  static int UpperPowerOfTwo(int n) {
    return (int)(pow(2, ceil(log(n) / log(2))));
  }

  // preemphasis
  static void PreEmphasis(float coeff, float* data, int size) {
    if (coeff == 0.0) return;
    for (int i = size - 1; i > 0; i--)
      (data)[i] -= coeff * (data)[i - 1];
    (data)[0] -= coeff * (data)[0];
  }

  // add hamming window
  static void Hamming(fbank_t* svr, float* data, int size) {
    //CHECK(data->size() >= hamming_window_.size());
    for (int i = 0; i < svr->frame_length_; ++i) {
      (data)[i] *= svr->hamming_window_[i];
    }
  }

int fbank_query_mem(int num_bins, int frame_length)
{
  unsigned ret = 0;
  unsigned fft_points = UpperPowerOfTwo(frame_length);
    // generate bit reversal table and trigonometric function table
  unsigned fft_points_4 = fft_points / 4;
  unsigned int num_fft_bins = fft_points / 2;
  ret = ALIGNMENT_SIZE;
  ret += ALIGNED_SIZE(sizeof(fbank_t), ALIGNMENT_SIZE);
  ret += ALIGNED_SIZE(fft_points*sizeof(int), ALIGNMENT_SIZE);
  ret += ALIGNED_SIZE((fft_points + fft_points_4)*sizeof(float), ALIGNMENT_SIZE);
  ret += ALIGNED_SIZE(num_bins*sizeof(int), ALIGNMENT_SIZE);
  ret += ALIGNED_SIZE(num_bins*sizeof(int), ALIGNMENT_SIZE);
  ret += ALIGNED_SIZE(num_bins*sizeof(int), ALIGNMENT_SIZE);
  //ret += ALIGNED_SIZE(num_bins*num_fft_bins*sizeof(float), ALIGNMENT_SIZE);
  ret += ALIGNED_SIZE(2*num_fft_bins*sizeof(float), ALIGNMENT_SIZE);  // freq2bank weight is sparse matrix.
  ret += ALIGNED_SIZE(num_bins*sizeof(int), ALIGNMENT_SIZE);
  ret += ALIGNED_SIZE(frame_length*sizeof(float), ALIGNMENT_SIZE);
  return ret;
}

int fbank_query_shm(int num_bins, int frame_length)
{
  unsigned ret = 0;
  unsigned fft_points = UpperPowerOfTwo(frame_length);
  ret = ALIGNMENT_SIZE;
  ret += ALIGNED_SIZE(fft_points*sizeof(float), ALIGNMENT_SIZE);
  ret += ALIGNED_SIZE(fft_points*sizeof(float), ALIGNMENT_SIZE); 
  ret += ALIGNED_SIZE((fft_points / 2)*sizeof(float), ALIGNMENT_SIZE);
  ret += ALIGNED_SIZE(frame_length*sizeof(float), ALIGNMENT_SIZE);
  return ret;
}

fbank_t* fbank_init(void* mem_addr, void* shm_addr, int num_bins, int sample_rate, int frame_length, int frame_shift)
{
    fbank_t *svr;
    void *p_mem, *p_shm;

    ASSERT(mem_addr != 0 && shm_addr != 0);
    
    p_mem               = mem_addr;
    p_shm               = shm_addr; 
    svr                 = (fbank_t *)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE); 
    p_mem               = (char*)p_mem + ALIGNED_SIZE(sizeof(fbank_t), ALIGNMENT_SIZE);
    svr->num_bins_      = num_bins;
    svr->sample_rate_   = sample_rate;
    svr->frame_length_  = frame_length;
    svr->frame_shift_   = frame_shift;
    svr->use_log_       = true;
    svr->remove_dc_offset_ = true;
    // svr->generator_ = 0,
    // svr->distribution_ = 0, 1.0,
    svr->dither_        = 0.0; 
    svr->fft_points_    = UpperPowerOfTwo(svr->frame_length_);
    // generate bit reversal table and trigonometric function table
    const int fft_points_4  = svr->fft_points_ / 4;
    svr->bitrev_            = (int*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem                   = (char*)p_mem + ALIGNED_SIZE(svr->fft_points_*sizeof(int), ALIGNMENT_SIZE);
    svr->sintbl_            = (float*)(int*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem                   = (char*)p_mem + ALIGNED_SIZE((svr->fft_points_ + fft_points_4)*sizeof(float), ALIGNMENT_SIZE);
    int num_fft_bins = svr->fft_points_ / 2;
    svr->bins_starts_       = (int*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem                   = (char*)p_mem + ALIGNED_SIZE(svr->num_bins_*sizeof(int), ALIGNMENT_SIZE);
    svr->bins_indexs_       = (int*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem                   = (char*)p_mem + ALIGNED_SIZE(svr->num_bins_*sizeof(int), ALIGNMENT_SIZE);
    svr->bins_sizes_        = (int*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem                   = (char*)p_mem + ALIGNED_SIZE(svr->num_bins_*sizeof(int), ALIGNMENT_SIZE);
    svr->bins_weights_      = (float*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem                   = (char*)p_mem + ALIGNED_SIZE(2*num_fft_bins*sizeof(float), ALIGNMENT_SIZE);
    svr->center_freqs_      = (float*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem                   = (char*)p_mem + ALIGNED_SIZE(svr->num_bins_*sizeof(int), ALIGNMENT_SIZE);
    svr->hamming_window_    = (float*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem                   = (char*)p_mem + ALIGNED_SIZE(svr->frame_length_*sizeof(float), ALIGNMENT_SIZE);
    // malloc shm
    svr->fft_real_tmp_      = (float*)ALIGNED_ADDR(p_shm, ALIGNMENT_SIZE);
    p_shm                   = (char*)p_shm + ALIGNED_SIZE(svr->fft_points_*sizeof(float), ALIGNMENT_SIZE); 
    svr->fft_img_tmp_       = (float*)ALIGNED_ADDR(p_shm, ALIGNMENT_SIZE);
    p_shm                   = (char*)p_shm + ALIGNED_SIZE(svr->fft_points_*sizeof(float), ALIGNMENT_SIZE); 
    svr->power_tmp_         = (float*)ALIGNED_ADDR(p_shm, ALIGNMENT_SIZE);
    p_shm                   = (char*)p_shm + ALIGNED_SIZE((svr->fft_points_ / 2)*sizeof(float), ALIGNMENT_SIZE);
    svr->data_tmp_          = (float*)ALIGNED_ADDR(p_shm, ALIGNMENT_SIZE);
    p_shm                   = (char*)p_shm + ALIGNED_SIZE(svr->frame_length_*sizeof(float), ALIGNMENT_SIZE);
    
    ASSERT((uintptr_t)(p_mem) - (uintptr_t)(mem_addr) <= fbank_query_mem(num_bins, frame_length));
    ASSERT((uintptr_t)(p_shm) - (uintptr_t)(shm_addr) <= fbank_query_shm(num_bins, frame_length));

    make_sintbl(svr->fft_points_, svr->sintbl_);
    make_bitrev(svr->fft_points_, svr->bitrev_);

    float fft_bin_width = (float)(svr->sample_rate_) / svr->fft_points_;
    int low_freq = 20, high_freq = svr->sample_rate_ / 2;
    float mel_low_freq = MelScale(low_freq);
    float mel_high_freq = MelScale(high_freq);
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1);
    int bins_start = 0;
    for (int bin = 0; bin < num_bins; ++bin) {
      float left_mel = mel_low_freq + bin * mel_freq_delta,
            center_mel = mel_low_freq + (bin + 1) * mel_freq_delta,
            right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;
      svr->center_freqs_[bin] = InverseMelScale(center_mel);
      //std::vector<float> this_bin(num_fft_bins);
      float* this_bin = svr->bins_weights_ + bin*num_fft_bins;
      int first_index = -1, last_index = -1;
      svr->bins_starts_[bin] = bins_start;
      for (int i = 0; i < num_fft_bins; ++i) {
        float freq = (fft_bin_width * i);  // Center frequency of this fft
        // bin.
        float mel = MelScale(freq);
        if (mel > left_mel && mel < right_mel) {
          float weight;
          if (mel <= center_mel)
            weight = (mel - left_mel) / (center_mel - left_mel);
          else
            weight = (right_mel - mel) / (right_mel - center_mel);
          svr->bins_weights_[bins_start++] = weight;
          if (first_index == -1) first_index = i;
          last_index = i;
        }
      }
      svr->bins_indexs_[bin] = first_index;
      svr->bins_sizes_[bin] = last_index + 1 - svr->bins_indexs_[bin];
    }

    // NOTE(cdliang): add hamming window
    double a = M_2PI / (frame_length - 1);
    for (int i = 0; i < frame_length; i++) {
      double i_fl = (double)(i);
      svr->hamming_window_[i] = 0.54 - 0.46 * cos(a * i_fl);
    }

    return svr;
  }

void fbank_uninit(fbank_t *svr)
{
}


  // Compute fbank feat, return num frames
  // wave num_samples*1
  // feat num_frames * num_bins
  int fbank_compute(fbank_t* svr, float* wave, float* feat, int num_samples) 
  {
    if (num_samples < svr->frame_length_) return 0;
    int num_frames = 1 + ((num_samples - svr->frame_length_) / svr->frame_shift_);
    float* fft_real = svr->fft_real_tmp_; memset(fft_real, 0, svr->fft_points_*sizeof(float));
    float* fft_img = svr->fft_img_tmp_; memset(fft_img, 0, svr->fft_points_*sizeof(float));
    float* power = svr->power_tmp_;
    float* data = svr->data_tmp_;
    for (int i = 0; i < num_frames; ++i) {
      memcpy(data, wave + i * svr->frame_shift_, svr->frame_length_*sizeof(float));
      // optional add noise
      // if (svr->dither_ != 0.0) {
      //   for (size_t j = 0; j < data.size(); ++j)
      //     data[j] += svr->dither_ * distribution_(svr->generator_);
      // }
      // optinal remove dc offset
      if (svr->remove_dc_offset_) {
        float mean = 0.0;
        for (int j = 0; j < svr->frame_length_; ++j) mean += data[j];
        mean /= svr->frame_length_;
        for (int j = 0; j < svr->frame_length_; ++j) data[j] -= mean;
      }
      
      PreEmphasis(0.97, data, svr->frame_length_);
      // Povey(&data);
      Hamming(svr, data, svr->frame_length_);
      // copy data to fft_real
      memset(fft_img, 0, sizeof(float) * svr->fft_points_);
      memset(fft_real + svr->frame_length_, 0,
             sizeof(float) * (svr->fft_points_ - svr->frame_length_));
      memcpy(fft_real, data, sizeof(float) * svr->frame_length_); 
      fft(svr->bitrev_, svr->sintbl_, fft_real, fft_img,
          svr->fft_points_);
      // power
      for (int j = 0; j < svr->fft_points_ / 2; ++j) {
        power[j] = fft_real[j] * fft_real[j] + fft_img[j] * fft_img[j];
      }
      
      // (*feat)[i].resize(svr->num_bins_);
      // cepstral coefficients, triangle filter array
      const float epsilon = 1.0e-10f;
      for (int j = 0; j < svr->num_bins_; ++j) {
        float mel_energy = 0.0;
        int s = svr->bins_indexs_[j];
        int weight_start = svr->bins_starts_[j];
        for (int k = 0; k < svr->bins_sizes_[j]; ++k) {
          mel_energy += svr->bins_weights_[weight_start + k] * power[s + k];
        }
        // optional use log
        if (svr->use_log_) {
          if (mel_energy < epsilon)
            mel_energy = epsilon;
          mel_energy = logf(mel_energy);
        }
        feat[i*svr->num_bins_+j] = mel_energy;
        // printf("%f ", mel_energy);
      }
      // printf("\n");
    }
    return num_frames;
  }
