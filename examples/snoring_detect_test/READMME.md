# SNORING_DETECT_TEST

喊声检测引擎测试

## 步骤

* 1.打包资源文件(打包步骤参考Thinker项目)。

```
/data/user/yswang/anaconda3/envs/thinker_py3.8.5/bin/tpacker -g ./data/snoring_net.quant.onnx -o ./data/snoring_net.quant.pkg
```

* 2.运行测试脚本。

```
>bash run.sh
```

## 参考

* 喊声检测训练项目 [https://github.com/mywang44/snoring_net]。
* Thinker项目 [https://github.com/LISTENAI/thinker]。
