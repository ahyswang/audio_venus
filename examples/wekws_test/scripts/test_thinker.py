import numpy as np
import wekws

def test_thinker():
    so_path = "./lib/libwekws_so.so"
    res_path = "./data.ignore/conv_linear.pkg"
    input = np.fromfile("./data.ignore/conv_linear.input.bin", dtype=np.float32)
    output_ref = np.fromfile("./data.ignore/conv_linear.output.bin", dtype=np.float32)
    scale = 4
    input = np.floor(input * (1<<scale) + 0.5).astype(np.int8)
    res_data = np.fromfile(res_path, dtype=np.int8)
    input=input.reshape((1,10,10,10))
    # print("input", input)
    # print("output_ref", output_ref)

    thinker = wekws.Thinker(so_path, res_data)
    thinker.set_input(0, scale, input)
    thinker.forward()
    output, scale = thinker.get_output(0)
    #print("output quant", output)
    output = (output.astype(np.float32) / (1<<scale))
    print(output.shape, scale)
    #print("output float", output)
    print("mae:", np.mean(output-output_ref))
    assert np.mean(output-output_ref) == 0

def test_thinker_2():
    so_path = "./lib/libwekws_so.so"
    res_path = "./data/snoring_net.quant.pkg"
    input = np.fromfile("./data/snoring_net.quant.input_int8.bin", dtype=np.int8)
    output_ref = np.fromfile("./data/snoring_net.quant.output_int8.bin", dtype=np.int8)
    scale = 4
    res_data = np.fromfile(res_path, dtype=np.int8)
    input=input.reshape((1,1,-1,64))
    # print("input", input)
    # print("output_ref", output_ref)

    thinker = wekws.Thinker(so_path, res_data)
    thinker.set_input(0, scale, input)
    thinker.forward()
    output, scale = thinker.get_output(0)
    #print("output quant", output)
    print(output.shape, scale)
    #print("output float", output)
    print("mae:", np.mean(output-output_ref))
    assert np.mean(output-output_ref) == 0

if __name__ == "__main__":
    test_thinker()