import ctypes
import numpy as np
import argparse
import librosa

class SnoringDetect(object):
    """ mylib wrap """
    def __init__(self, 
                 res_data, 
                 num_bins = 64,
                 sample_rate = 16000,
                 frame_length = 64,
                 frame_shift = 32,
                 qvalue = 4, 
                 so_path = "libsnoring_detect_so.so") -> None:
        """ init object """
        class SnoringDetectConf(ctypes.Structure):
            _fields_ = [
                ("classes_num", ctypes.c_int),
                ("num_bins", ctypes.c_int),
                ("sample_rate", ctypes.c_int),
                ("max_frame_num", ctypes.c_int),
                ("frame_length", ctypes.c_int),
                ("frame_shift", ctypes.c_int),
                ("qvalue", ctypes.c_int),
                ("res_size", ctypes.c_int),
                ("res_addr", ctypes.c_void_p),
                ("extra1", ctypes.c_void_p),
                ("extra2", ctypes.c_void_p),
            ]
        # so (very important !!!!!)
        self.mylib = ctypes.cdll.LoadLibrary(so_path)
        self.mylib.snoring_detect_query_mem.argtypes = (ctypes.c_void_p, )
        self.mylib.snoring_detect_query_mem.restype = ctypes.c_uint32
        self.mylib.snoring_detect_query_shm.argtypes = (ctypes.c_void_p, )
        self.mylib.snoring_detect_query_shm.restype = ctypes.c_uint32
        self.mylib.snoring_detect_init.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self.mylib.snoring_detect_init.restype = ctypes.c_void_p
        self.mylib.snoring_detect_process.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)
        self.mylib.snoring_detect_process.restype = ctypes.c_uint32
        self.mylib.snoring_detect_uninit.argtypes = (ctypes.c_void_p, )
        assert res_data.dtype == np.int8
        # conf 
        self.res_data = res_data
        self.extra1 = np.zeros((64*num_bins,), dtype=np.float32)
        self.extra2 = np.zeros((64*num_bins,), dtype=np.float32)
        self.conf = SnoringDetectConf()
        self.conf.classes_num = 2
        self.conf.num_bins = num_bins
        self.conf.sample_rate = sample_rate
        self.conf.max_frame_num = 64
        self.conf.frame_length = frame_length*sample_rate//1000
        self.conf.frame_shift = frame_shift*sample_rate//1000
        self.conf.qvalue = qvalue
        self.conf.res_size = len(res_data)
        self.conf.res_addr = self.res_data.ctypes.data_as(ctypes.c_void_p)
        self.conf.extra1 = self.extra1.ctypes.data_as(ctypes.c_void_p)   # export fea
        self.conf.extra2 = 0 #self.extra2.ctypes.data_as(ctypes.c_void_p) # import fea
        # init 
        mem_size = self.mylib.snoring_detect_query_mem(ctypes.pointer(self.conf))
        shm_size = self.mylib.snoring_detect_query_shm(ctypes.pointer(self.conf))
        self.mem_data = np.zeros(mem_size, np.int8)
        self.shm_data = np.zeros(shm_size, np.int8)
        print(f"men_size:{mem_size}, shm_size:{shm_size}")
        self.handle = self.mylib.snoring_detect_init(self.mem_data.ctypes.data_as(ctypes.c_void_p), 
                                                     self.shm_data.ctypes.data_as(ctypes.c_void_p), 
                                                     ctypes.pointer(self.conf))
        #print(f"handle:{self.handle}")
        
    def process(self, input_data):
        """ process input data """
        assert input_data.dtype == np.float32
        assert input_data.size > self.conf.frame_shift and input_data.size <= (self.conf.max_frame_num * self.conf.frame_shift)
        output_data = np.zeros(self.conf.classes_num, np.float32)
        self.mylib.snoring_detect_process(self.handle,
                                        input_data.ctypes.data_as(ctypes.c_void_p),
                                        output_data.ctypes.data_as(ctypes.c_void_p),
                                        input_data.size)
        return output_data
    
def get_args():
    parser = argparse.ArgumentParser(description='snoring detect engine.')
    parser.add_argument('--res', required=True, help='res file.')
    parser.add_argument('--wav', required=True, help='wav file.')
    parser.add_argument('--so', default="libsnoring_detect_so.so", help='so file.')
    parser.add_argument('--qvalue', default=4, help='onnx input qvalue.')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    print(args) 
    res_data = np.fromfile(args.res, np.int8)
    input_data, sr = librosa.load(args.wav, sr=None, mono=True)

    if sr != 16000:
        input_data = librosa.resample(input_data, orig_sr=sr, target_sr=16000)
        sr = 16000
    if input_data.shape[0] < 2*16000:
        input_data = np.pad(input_data, int(np.ceil((2.048*sr-input_data.shape[0])/2)), mode='reflect')
    else:
        input_data = input_data[:2.048*sr]

    input_data.tofile("./data.ignore/input_data_float32.bin")
   
    mylib = SnoringDetect(res_data, qvalue=args.qvalue, so_path=args.so)
    output_data = mylib.process(input_data)

    print("pred:", output_data)  


if __name__ == "__main__":
    main()
    
    