import ctypes
import math 
import numpy as np
import argparse

class FBank(object):
    """ FBank wrap """
    def __init__(self, num_bins:int, sample_rate:int, frame_length:int, frame_shift:int, so_path:str = "libwekws.so") -> None:
        """ init object """
        self.num_bins = num_bins
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        #print(self.num_bins, self.sample_rate, self.frame_length, self.frame_shift)
        # so (very important !!!!!)
        self.mylib = ctypes.cdll.LoadLibrary(so_path)
        self.mylib.fbank_query_mem.argtypes = (ctypes.c_int32, ctypes.c_int32)
        self.mylib.fbank_query_mem.restype = ctypes.c_uint32
        self.mylib.fbank_query_shm.argtypes = (ctypes.c_int32, ctypes.c_int32)
        self.mylib.fbank_query_shm.restype = ctypes.c_uint32
        self.mylib.fbank_init.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
        self.mylib.fbank_init.restype = ctypes.c_void_p
        self.mylib.fbank_compute.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)
        self.mylib.fbank_compute.restype = ctypes.c_uint32
        self.mylib.fbank_uninit.argtypes = (ctypes.c_void_p, )
        # init 
        mem_size = self.mylib.fbank_query_mem(num_bins, frame_length)
        shm_size = self.mylib.fbank_query_shm(num_bins, frame_length)
        self.mem_data = np.zeros(mem_size + 1024, np.int8)
        self.shm_data = np.zeros(shm_size + 1024, np.int8)
        self.handle = self.mylib.fbank_init(self.mem_data.ctypes.data_as(ctypes.c_void_p), 
                                            self.shm_data.ctypes.data_as(ctypes.c_void_p), 
                                            num_bins, sample_rate, frame_length, frame_shift)
         
    def process(self, input_data):
        """ process input data """
        output_data = np.zeros((input_data.size//self.frame_shift + 1)*self.num_bins, np.float32)
        #import pdb; pdb.set_trace()
        num_frames = self.mylib.fbank_compute(self.handle,
                                input_data.ctypes.data_as(ctypes.c_void_p),
                                output_data.ctypes.data_as(ctypes.c_void_p),
                                input_data.size)
        #print("ret:", ret, input_data.size)
        return output_data[0:num_frames*self.num_bins]
    
if __name__ == "__main__":
    
    num_samples = 1600
    num_bins = 80
    sample_rate = 16000
    frame_length = sample_rate*25//1000  # must be int
    frame_shift = sample_rate*10//1000

    input_data = np.zeros((num_samples,), dtype=np.float32)
    for i in range(num_samples):
        input_data[i] = i%256
        
    mylib = FBank(num_bins, sample_rate, frame_length, frame_shift, "./lib/libwekws_so.so")
    output_data = mylib.process(input_data)

    print(input_data)
    print(output_data)
    