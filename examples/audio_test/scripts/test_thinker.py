import numpy as np
import audio_py

def test_thinker():
    so_path = "./lib/libaudio_so.so"
    res_path = "./data.ignore/conv_linear.pkg"
    input = np.fromfile("./data.ignore/conv_linear.input.bin", dtype=np.float32)
    output_ref = np.fromfile("./data.ignore/conv_linear.output.bin", dtype=np.float32)
    scale = 4
    input = np.floor(input * (1<<scale) + 0.5).astype(np.int8)
    res_data = np.fromfile(res_path, dtype=np.int8)
    input=input.reshape((1,10,10,10))
    # print("input", input)
    # print("output_ref", output_ref)

    thinker = audio_py.Thinker(so_path, res_data)
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
    so_path = "./lib/libaudio_so.so"
    res_path = "../../examples/snoring_detect_test/data/snoring_net.quant.pkg"
    input = np.fromfile("../../examples/snoring_detect_test/data/input_float32.bin", dtype=np.float32)
    output_ref = np.fromfile("../../examples/snoring_detect_test/data/output_float32.bin", dtype=np.float32)
    scale = 4
    res_data = np.fromfile(res_path, dtype=np.int8)
    input=input.reshape((1,1,-1,64))
    # print("input", input)
    print("output_ref", output_ref)

    input = np.floor(input*(1<<scale) + 0.5).astype(np.int8)

    thinker = audio_py.Thinker(so_path, res_data)
    thinker.set_input(0, scale, input)
    thinker.forward()
    output, scale = thinker.get_output(0)
    #print("output quant", output)
    output = output / (1<<scale)
    print(output.shape, scale)
    print("output float", output)
    print("mae:", np.mean(output-output_ref))
    assert np.mean(output-output_ref) == 0

if __name__ == "__main__":
    test_thinker_2()