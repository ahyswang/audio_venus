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
    

class Thinker(object):
    class Tensor(ctypes.Structure):
        _fields_ = [
            ("dim", ctypes.c_int),
            ("shape", ctypes.c_int * 4),
            ("scale", ctypes.c_int),
            ("dtype", ctypes.c_int),
            ("size", ctypes.c_int),
            ("data", ctypes.c_void_p),
        ]
    """ mylib wrap """
    def __init__(self, so_path, res_data) -> None:
        """ init object """
        # so (very important !!!!!)
        self.mylib = ctypes.cdll.LoadLibrary(so_path)
        self.mylib.thinker_query_mem.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        self.mylib.thinker_query_mem.restype = ctypes.c_uint32
        self.mylib.thinker_query_shm.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        self.mylib.thinker_query_shm.restype = ctypes.c_uint32
        self.mylib.thinker_init.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)
        self.mylib.thinker_init.restype = ctypes.c_void_p
        self.mylib.thinker_forward.argtypes = (ctypes.c_void_p, )
        self.mylib.thinker_forward.restype = ctypes.c_int32
        self.mylib.thinker_uninit.argtypes = (ctypes.c_void_p, )
        self.mylib.thinker_set_input.argtypes = (ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p)
        self.mylib.thinker_set_input.restype = ctypes.c_int32
        self.mylib.thinker_get_output.argtypes = (ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p)
        self.mylib.thinker_get_output.restype = ctypes.c_int32
        # conf 
        self.res_data = res_data
        res_addr = self.res_data.ctypes.data_as(ctypes.c_void_p)
        res_size = self.res_data.size 
        # init 
        mem_size = self.mylib.thinker_query_mem(res_addr, res_size)
        shm_size = self.mylib.thinker_query_shm(res_addr, res_size)
        self.mem_data = np.zeros(mem_size + 1024, np.int8)
        self.shm_data = np.zeros(shm_size + 1024, np.int8)
        self.handle = self.mylib.thinker_init(self.mem_data.ctypes.data_as(ctypes.c_void_p),
                                              self.shm_data.ctypes.data_as(ctypes.c_void_p), 
                                            res_addr, res_size)
        #print(f"mem_size:{mem_size}, shm_size:{shm_size}, handle:{self.handle}")
        
    def set_input(self, id, scale, input):
        assert input.dtype == np.int8
        tensor = Thinker.Tensor()
        tensor.dim = len(input.shape)
        tensor.shape = input.shape 
        tensor.scale = scale
        tensor.data = input.ctypes.data_as(ctypes.c_void_p)
        tensor.size = input.size
        ret = self.mylib.thinker_set_input(self.handle, id, ctypes.pointer(tensor))
        return ret 
    
    def get_output(self, id):
        tensor = Thinker.Tensor()
        tensor.size = 0
        ret = self.mylib.thinker_get_output(self.handle, id, ctypes.pointer(tensor))
        assert 0==ret 
        output = np.zeros((tensor.size), dtype=np.int8)
        tensor.data = output.ctypes.data_as(ctypes.c_void_p) 
        tensor.size = output.size
        ret = self.mylib.thinker_get_output(self.handle, id, ctypes.pointer(tensor))
        assert 0==ret
        return output, tensor.scale

    def forward(self):    
        """"""
        ret = self.mylib.thinker_forward(self.handle)
        assert 0==ret
        return None
    
def test_fbank():
    """"""
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

def test_thinker():
    so_path = "/data/user/yswang/task/wekws/wekws_venus/build_linux/lib/libwekws_so.so"
    res_path = "/data/user/yswang/task/wekws/thinker/model.pkg"
    input = np.fromfile("/data/user/yswang/task/wekws/thinker/demo/resnet50/input.bin", dtype=np.float32)
    output_ref = np.fromfile("/data/user/yswang/task/wekws/thinker/demo/resnet50/output.bin", dtype=np.float32)
    scale = 7
    input = np.floor(input * (1<<scale) + 0.5).astype(np.int8)
    res_data = np.fromfile(res_path, dtype=np.int8)
    input=input.reshape((1,1,32,32))
    print("input", input)
    print("output_ref", output_ref)

    thinker = Thinker(so_path, res_data)
    thinker.set_input(0, 7, input)
    thinker.forward()
    output, scale = thinker.get_output(0)
    print("output quant", output)
    output = (output.astype(np.float32) / (1<<scale))
    print(output.shape, scale)
    print("output float", output)

if __name__ == "__main__":
    test_fbank()
    #test_thinker()
    