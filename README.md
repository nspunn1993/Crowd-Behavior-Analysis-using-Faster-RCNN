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
Append the PYTHONPATH environment variable as

```PYTHONPATH=<Model master path>;<Model master path>\research;<Model master path>\research\slim;```

# Step 2: Anaconda virtual environment
I recommend to use miniconda instead of anaconda because of minimal already installed packages.

Open the anaconda command prompt and type the following:

Note: Before installing tensorflow-gpu you need to install CUDA and CuDNN with compatible versions. For more details you can refer [here](https://punndeeplearningblog.com/development/tensorflow-cuda-cudnn-compatibility/)
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
- (objdet) C:\> cd C:\<model-master>\research
- (objdet) C:\> protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
- (objdet) C:\<model-master>\research> python setup.py build
- (objdet) C:\<model-master>\research> python setup.py install
```
You can test the setup as follows:

``` (objdet) C:\<model-master>\research\object_detection> jupyter notebook object_detection_tutorial.ipynb ```

Run the jupyter notebook and it should run without any errors.

# Step 3: Dataset


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
