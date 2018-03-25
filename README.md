# 基于NeuralTalk2 图片标注 - web版

> web版图片标注示例
> 如有问题，欢迎拍砖，多多指教 😀

## 简要说明

本案例主要使用nerualtalks基于mscoco数据集训练后的模型,搭建flask网站进行上传图片进行标注,具体配置可参考[neuraltalk2](https://github.com/karpathy/neuraltalk2)进行操作.

### 环境配置  
ubuntu16 + python3 + lua + torch + flask

### 具体使用

- git clone https://github.com/RookieDay/neuraltalk2_web.git 
- 下载对应依赖包/配置环境
- 根据neuraltalk2 readme 下载对应的model checkpoint（有GPU版和CPU版本）
- python app.py --debug True (debug模式 默认使用CPU)
- 浏览器输入 http://0.0.0.0:5000/ 即可

## 网站效果预览

![初始页面](https://github.com/RookieDay/neuraltalk2_web/blob/master/01.png)

![动态展示](https://github.com/RookieDay/neuraltalk2_web/blob/master/02.gif)

## 开发计划

- [x] 上传单张图片标注；
- [x] 实现上传多张图片标注；
- [ ] 实现上传视频标注；
- [ ] 添加进度条；


## 许可

[MIT](./LICENSE) &copy; [RookieDay](https://github.com/RookieDay)
