import numpy as np 
import math
import sys 
import os 
import argparse 

def bin2header(bin_path, header_path):
    data = np.fromfile(bin_path, dtype=np.uint8)
    data = data.reshape(-1)
    size = data.shape[0]
    varname = os.path.basename(bin_path)
    #assert varname.endswith(".bin")
    varname = varname.replace(".bin", "").replace(".", "_").replace("-", "_")

    header_file = open(header_path, "w")
    header_file.write(f"const unsigned int %s_reslen = %d;\n"%(varname, size))
    #header_file.write(f"const unsigned char %s_resource[%d] = {{\n"%(varname, size))
    # TODO: thinker init memcpy bug.
    header_file.write(f"unsigned char %s_resource[%d] = {{\n"%(varname, size))
    for i in range(size):
        header_file.write("0x%x,\t"%(data[i]))
        if i > 0 and (i+1)%16 == 0:
            header_file.write(f"\n")
    header_file.write(f"}};\n")

def main():
    parser = argparse.ArgumentParser(description="convert binary file to c header file.")
    parser.add_argument("-i", "--input", help="input file", required=True)
    parser.add_argument("-o", "--output", help="output file", required=True)
    args = parser.parse_args()
    print(args)
    assert os.path.exists(args.input)
    bin2header(args.input, args.output)

if __name__ == "__main__":
    main()