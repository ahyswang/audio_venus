rm -rf build_linux
mkdir build_linux 

build_type=Release
if [ "$1" == "debug" ];then
    build_type=Debug
fi

cd build_linux
echo "BUILD TYPE=" $build_type
cmake -DCMAKE_BUILD_TYPE=$build_type -DPLATFORM_NAME=linux64 ..
make 
cd ..