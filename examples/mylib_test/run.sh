script_dir=$(dirname $(readlink -f "$0"))
script_name=$(basename $(readlink -f "$0") .sh)
root_dir=${script_dir}/../../

pushd ${script_dir}/
mkdir ./bin/
mkdir ./lib/
cp ${root_dir}/build_linux/bin/mylib_test ./bin/
cp ${root_dir}/build_linux/lib/libmylib.a ./lib/
cp ${root_dir}/build_linux/lib/libmylib_so.so ./lib/
chmod +x ./mylib_test
popd

pushd ${script_dir}/bin/
./mylib_test
popd

pushd ${script_dir}/
python ./scripts/mylib.py
popd 