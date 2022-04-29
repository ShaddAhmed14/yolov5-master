## <div align="center">YOLOv5 and Seq-NMS</div>

<div>
<p>
YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
</p>
 
<p>
This repo performs post-processing with Sequential NMS on yolov5 to improve results. The Seq-NMS method is adopted from <a href="https://arxiv.org/pdf/1602.08465.pdf">"Seq-NMS for Video Object Detection"</a>.
</p>
 
</div>


## <div align="center">Set-Up and Inference</div>

<div align-"left">Steps:</div>
<div>
1. Install
</div>
<div>
2. Inference
</div>
<div>
<p>
 </p>
 <i>Please follow commands in sections below.</i>
 <p>
 </p>
</div>
<div align="left">
<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/ShaddAhmed14/yolov5-master  # clone
cd yolov5-master
pip install -r requirements.txt  # install
```

</details>

<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from
the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

```bash
# Example

python detect.py --source  'https://www.youtube.com/watch?v=P1qHv44_wLQ'
```

</details>
</div>
