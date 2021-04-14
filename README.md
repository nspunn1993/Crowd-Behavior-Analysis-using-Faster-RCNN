# Crowd-Behavior-Analysis-using-Faster-RCNN
Crowd Analysis for Congestion Control Early Warning System on Foot Over Bridge.
This follows from the tensorflow object detection API available [here](https://github.com/tensorflow/models/tree/r1.12.0)

# Steps to follow
- Set up the object detection directory structure and TensorflowGPU enable anaconda virtual envrironment.
- Generate the labelled dataset.
- Divide the labelled datset into trainig and test set.
- Generate label map and configure the training parameters.
- Train the object detector model and export the inference graph for testing.

# Step 1: Setup
Download the following files:
- Tensorflow object api repostiory from [here](https://github.com/tensorflow/models/tree/r1.12.0).
- An object detection model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) - we used [Faster RCNN inception V2 Coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) model.
- Download the files from this repository.
Extract the tensorflow object detection api repository and navigate to 
```
.model-master/research/object_detection/. 
```
Extract the object detection model and files from this repository there.

# Step 2: Anaconda virtual environment
I recommend to use miniconda instead of anaconda because of minimal already installed packages.
Open the anaconda command prompt and type the following:
```
- C:\> conda create -n objdet pip python=3.6
- C:\> activate objdet
- (objdet) C:\> pip install tensorflow-gpu==1.12
- (objdet) C:\> conda install -c anaconda protobuf
- (objdet) C:\> pip install pillow
- (objdet) C:\> pip install lxml
- (objdet) C:\> pip install Cython
- (objdet) C:\> pip install pandas
- (objdet) C:\> pip install contextlib2
- (objdet) C:\> pip install matplotlib
- (objdet) C:\> pip install opencv-python
- (objdet) C:\> pip install jupyter
```
Append the PYTHONPATH environment variable as PYTHONPATH=<Model master path>;<Model masterpath>\research;<Model master path>\research\slim;

# Citation
```
@inproceedings{punn2019crowd,
  title={Crowd analysis for congestion control early warning system on foot over bridge},
  author={Punn, Narinder Singh and Agarwal, Sonali},
  booktitle={2019 Twelfth International Conference on Contemporary Computing (IC3)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}

```
