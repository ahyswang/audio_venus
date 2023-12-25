script_dir=$(dirname $(readlink -f "$0"))
script_name=$(basename $(readlink -f "$0") .sh)
root_dir=${script_dir}/../../

pushd ${script_dir}/
mkdir ./bin/
mkdir ./lib/
cp ${root_dir}/build_linux/bin/wekws_test ./bin/
cp ${root_dir}/build_linux/lib/libwekws.a ./lib/
cp ${root_dir}/build_linux/lib/libwekws_so.so ./lib/
chmod +x ./wekws_test
popd

pushd ${script_dir}/bin/
./wekws_test
popd

# test python
pushd ${script_dir}/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./lib/:../lib/:${root_dir}/deps/linux64/lib/
/data/user/yswang/anaconda3/envs/wekws_py3.7.3_torch1.9.0_cu10.2/bin/python ./scripts/test_linger.py 
/data/user/yswang/anaconda3/envs/thinker_py3.8.5/bin/tpacker -g ./data.ignore/conv_linear.onnx -o ./data.ignore/conv_linear.pkg
python ./scripts/wekws.py
python ./scripts/test_fbank.py
python ./scripts/test_thinker.py
popd 

