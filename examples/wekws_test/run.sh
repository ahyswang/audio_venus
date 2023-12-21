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

pushd ${script_dir}/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./lib/:../lib/
#python ./scripts/wekws.py
python ./scripts/test_fbank.py
popd 