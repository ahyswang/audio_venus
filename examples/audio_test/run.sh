script_dir=$(dirname $(readlink -f "$0"))
script_name=$(basename $(readlink -f "$0") .sh)
root_dir=${script_dir}/../../

pushd ${script_dir}/
mkdir ./bin/
mkdir ./lib/
cp ${root_dir}/build_linux/bin/audio_test ./bin/
cp ${root_dir}/build_linux/lib/libaudio.a ./lib/
cp ${root_dir}/build_linux/lib/libaudio_so.so ./lib/
chmod +x ./bin/audio_test
popd

pushd ${script_dir}/bin/
./audio_test
popd

# test python
pushd ${script_dir}/
mkdir ./data.ignore
/data/user/yswang/anaconda3/envs/wekws_py3.7.3_torch1.9.0_cu10.2/bin/python ./scripts/test_linger.py 
/data/user/yswang/anaconda3/envs/thinker_py3.8.5/bin/tpacker -g ./data.ignore/conv_linear.onnx -o ./data.ignore/conv_linear.pkg
python ./scripts/test_fbank.py
python ./scripts/test_thinker.py
popd 

