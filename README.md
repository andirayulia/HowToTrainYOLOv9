# HowToTrainYOLOv9

**Preparation**

Membuat konstanta (HOME) untuk menyimpan path direktori utama
```python
import os
HOME = os.getcwd()
print(HOME)
```

**Clone and Install**

Melakukan kloning untuk mendapatkan seluruh repository yolov9 dan installasi requirements 
```python
!git clone https://github.com/SkalskiP/yolov9.git
%cd yolov9
!pip install -r requirements.txt -q
```

Install Package roboflow untuk mengakses dan mengelola dataset dari Roboflow
```python
!pip install -q roboflow
```

**Download Model Weights**

Model yang digunakan adalah YOLO dan Gelan<br>
kapasitas dan akurasi model extended lebih tinggi dibandingkan dengan compact

c = compact
e = extended
```python
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt
```

Dari model tersebut, dapat dilihat bahwa model dengan versi extended memiliki ukuran memori yang lebih besar, dimana yolov9-e memiliki ukuran paling besar yaitu 140 MB
![image](https://github.com/user-attachments/assets/41e3334e-d529-4e36-a2a9-d3ecb3e6e51c)


**Download Example Data**

mendownload image file dan menyimpan pada folder data
```python
!wget -P {HOME}/data -q https://media.roboflow.com/notebooks/examples/dog.jpeg
!wget -P {HOME}/data -q https://cdn.rri.co.id/berita/Mamuju/o/1717043731797-_5d8e74a4-e657-46b6-8fa2-ec25653c4894/kstgu2qd72q6oiv.jpeg
```
**Detection with Pre-Trained COCO Model**

--gelan-c--
```python
!python detect.py --weights {HOME}/weights/gelan-c.pt --conf 0.1 --source {HOME}/data/dog.jpeg --device 0
```
```python
!python detect.py --weights {HOME}/weights/gelan-c.pt --conf 0.1 --source {HOME}/data/mendaki.jpeg --device cpu
```
![image](https://github.com/user-attachments/assets/99ded46f-253f-4101-b73d-bdfb6bc39eec)
![image](https://github.com/user-attachments/assets/6462fa3f-4a41-4a43-a69d-6bb027b47380)

--yolov9-e--
```python
!python detect.py --weights {HOME}/weights/yolov9-e.pt --conf 0.1 --source {HOME}/data/dog.jpeg --device 0
```
```python
!python detect.py --weights {HOME}/weights/yolov9-e.pt --conf 0.1 --source {HOME}/data/mendaki.jpeg --device cpu
```
![image](https://github.com/user-attachments/assets/8952a07d-b79b-4727-95e0-f522c2f57e29)
![image](https://github.com/user-attachments/assets/d06c76c9-5409-4fdf-8b00-10d584b0f9db)

Dari deteksi object diatas, didapatkan bahwa yolov9-e dapat mendeteksi object lebih banyak 

**Authenticate and Download the Dataset**

```python
import roboflow

roboflow.login()

rf = roboflow.Roboflow()

project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(8)
dataset = version.download("yolov9")
```

**Train Model**

Melakukan training menggunakan model gelan compact
```python
%cd {HOME}/yolov9

!python train.py \
--batch 16 --epochs 25 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data {dataset.location}/data.yaml \
--weights {HOME}/weights/gelan-c.pt \
--cfg models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml
```

#batch 16  : Menentukan ukuran batch, yaitu 16 gambar per batch. <br>
#epochs 25 : Menentukan jumlah epoch, yaitu 25 iterasi pelatihan. <br>
#img 640   : Menentukan ukuran gambar input yang digunakan dalam pelatihan, yaitu 640x640 piksel.

**RESULT**

metrics/mAP_0.5: Mean Average Precision (mAP) dengan threshold IoU 0.5. mAP merupakan metrik untuk mengukur performa model secara keseluruhan. Semakin tinggi nilai mAP, semakin baik performa model.
![image](https://github.com/user-attachments/assets/532be413-7fc3-443f-a80f-1f1e5a94ea23)

== CONFUSION_MATRIX ==

![image](https://github.com/user-attachments/assets/91f12034-c024-46bd-bd13-f56c1bbed9f1)
![image](https://github.com/user-attachments/assets/10732815-86ec-4977-9a5f-e246fc08fc0a)

**Validate Custom Model**

--img 640: Menentukan ukuran gambar input untuk evaluasi (640 pixel). <br>
--batch 32: Menentukan ukuran batch untuk evaluasi (32 gambar). <br>
--conf 0.001: Menentukan ambang batas confidence score (minimal 0.001 untuk menganggap deteksi valid). <br>
--iou 0.7: Menentukan ambang batas Intersection over Union (IoU) untuk menganggap deteksi valid (minimal 0.7).<br>

```python
%cd {HOME}/yolov9

!python val.py \
--img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 \
--data {dataset.location}/data.yaml \
--weights {HOME}/yolov9/runs/train/exp/weights/best.pt
```
![image](https://github.com/user-attachments/assets/bec1d244-dd00-4491-8b3d-6e58c560db39)

**Inference with Custom Model**
```python
!python detect.py \
--img 1280 --conf 0.1 --device 0 \
--weights {HOME}/yolov9/runs/train/exp/weights/best.pt \
--source {dataset.location}/test/images
```
```python
import glob

from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/yolov9/runs/detect/exp6/*.jpg')[:2]:
      display(Image(filename=image_path, width=600))
```
![image](https://github.com/user-attachments/assets/37a090a8-84f9-4bba-99fd-cb032a705422)
![image](https://github.com/user-attachments/assets/4f59cbb1-8e54-452e-bcd4-398a2fb099e8)






