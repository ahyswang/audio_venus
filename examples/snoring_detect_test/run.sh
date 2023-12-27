script_dir=$(dirname $(readlink -f "$0"))
script_name=$(basename $(readlink -f "$0") .sh)
root_dir=${script_dir}/../../

pushd ${script_dir}/
mkdir ./data.ignore/
mkdir ./bin/
mkdir ./lib/
cp ${root_dir}/build_linux/bin/snoring_detect_test ./bin/
cp ${root_dir}/build_linux/lib/libsnoring_detect.a ./lib/
cp ${root_dir}/build_linux/lib/libsnoring_detect_so.so ./lib/
chmod +x ./bin/snoring_detect_test
popd

pushd ${script_dir}/bin/
./snoring_detect_test
#./snoring_detect_test ../data/snoring_net.quant.pkg ../data/audio_segment_20_pad_float32.bin 
popd

# test python
pushd ${script_dir}/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./lib:./build_linux/lib
python ./scripts/snoring_detect_py.py --res "./data/snoring_net.quant.pkg" \
--wav "./data/audio_segment_20.wav"
popd 

