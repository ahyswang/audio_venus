# AUDIO_VENUS

VENUS芯片上运行的音频示例引擎，包括离线鼾声检测、流式唤醒（待完成）。

## 环境部署

* 1.安装python运行环境。
```
conda create -n audio_venus python=3.7.3
conda activate audio_venus
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## 源码编译 

* 1.linux64编译。
```
 ./build_linux.sh
```

## 样例测试

* [喊声检测] (examples/snoring_detect_test/README.md) 完成
* [流式唤醒] (examples/wekws_test/README.md) 待完成

## 参考

* Thinker推理框架（commit dfc3a0fefbe49c35e134948b273fc1fb5cee068c） [https://github.com/LISTENAI/thinker]   
* Linger量化训练（commit 64edb4c7d4d9db7bf67a9ec419b4b219c164457c） [https://github.com/LISTENAI/linger]  
* Wekws流式唤醒 [https://github.com/wenet-e2e/wekws]  

