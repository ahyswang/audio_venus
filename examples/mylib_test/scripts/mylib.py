import ctypes
import math 
import numpy as np
import argparse

class Mylib(object):
    """ mylib wrap """
    def __init__(self, so_path, res_data) -> None:
        """ init object """
        class MylibConf(ctypes.Structure):
            _fields_ = [
                ("p_res", ctypes.c_void_p * 4),
                ("p_size", ctypes.c_int * 4),
            ]
        # so (very important !!!!!)
        self.mylib = ctypes.cdll.LoadLibrary(so_path)
        self.mylib.mylib_query_mem.argtypes = (ctypes.c_void_p, )
        self.mylib.mylib_init.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self.mylib.mylib_init.restype = ctypes.c_void_p
        self.mylib.mylib_process.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)
        self.mylib.mylib_process.restype = ctypes.c_uint32
        self.mylib.mylib_uninit.argtypes = (ctypes.c_void_p, )
        # conf 
        self.res_data = res_data
        self.conf = MylibConf()
        self.conf.p_res[0] = self.res_data.ctypes.data_as(ctypes.c_void_p)
        self.conf.p_size[0] = self.res_data.size 
        # init 
        mem_size = self.mylib.mylib_query_mem(ctypes.pointer(self.conf))
        self.mem_data = np.zeros(mem_size + 1024, np.int8)
        self.handle = self.mylib.mylib_init(self.mem_data.ctypes.data_as(ctypes.c_void_p), 
                                            ctypes.pointer(self.conf))
        
    def process(self, input_data):
        """ process input data """
        output_data = np.zeros(input_data.size + self.res_data.size, np.int8)
        self.mylib.mylib_process(self.handle,
                                        input_data.ctypes.data_as(ctypes.c_void_p),
                                        output_data.ctypes.data_as(ctypes.c_void_p),
                                        input_data.size)
        return output_data
    
if __name__ == "__main__":

    so_path = "./lib/libmylib_so.so"
    res_data = np.array([1, 2, 3, 4], np.int8)
    input_data = np.array([5, 6, 7, 8, 9, 10], np.int8)
    #np.fromfile(res_path, np.int8)

    mylib = Mylib(so_path, res_data)
    output_data = mylib.process(input_data)

    print(res_data)
    print(input_data)
    print(output_data)
    
    #print(np.concatenate((res_data, input_data), axis=0) == output_data)