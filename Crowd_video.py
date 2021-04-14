
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from scipy.linalg import norm
from scipy import sum, average
from datetime import datetime
import math

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'Sequence 01_1.mp4'

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'training','label_map.pbtxt')

PATH_TO_VIDEO = os.path.join(CWD_PATH+'\\Videos',VIDEO_NAME)

PATH_TO_OUTPUT = os.path.join(CWD_PATH,'output')

NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

video = cv2.VideoCapture(PATH_TO_VIDEO)
width = 500
height = 800
fbgb = cv2.createBackgroundSubtractorMOG2()
area = [] #Area of the bounding boxes
new_area = [] #Order the area to compare in next frame: Object tracking
new_centroid = [] #Order the centroid to comapre in next frame: Object tracking
good_flag = 0 #Flag responsible for no stempede
bad_flag = 0 #Flag responsible for stempede
reset_status_count = 0 #Counter to check the window size of reset_status_flag
stempede_flag = 0 #Stempede flag
direction_flag = '' #To detect the direction of the crwod: can be removed later
agg_forward_count = 0 #For direction decision
agg_backward_count = 0 #For direction decision : obselete
agg_count_window = 20 #For resetting agg Forward count and agg Backward count : obselete
agg_count = 0 #Counter to check agg_count_window
message = 'Alert' #Displayed during stampede
max_still_frames = 10 #Max still frames : obselete
still_threshold = 0.8 #Detect still frame : obselete
still_count = 0 #Track number of still frames
img_write_flag = 0 #To save stempede alert frame

#Tune Parameters Start
area_diff_threshold = 0.2 #Min difference in area allowed
reset_status_flags = 20 #Window size to compare good and bad flags for stempede and then reset them
max_people_count = 5 #Max people allowed in frame
direction_sensitivity = 0.8 #Ranges between 0 and 1: 1 is max sensitivity
still_sensitivity = 0.4 #Ranges between 0 and 1: 0 is more sensitive
#Tune Parameters End

def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    d1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    d2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = normalize(d1)
    img2 = normalize(d2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    #z_norm = norm(diff.ravel(), 0)  # Zero norm
    m_norm = m_norm / d1.size
    return m_norm

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng
'''
def updateStatus(forward_count, backward_count, count):
    if forward_count > 0 or backward_count > 0:
        if (forward_count == 0 and backward_count == count) or (forward_count == count and backward_count == 0):
            good_flag += 2
        elif forward_count == backward_count:
            bad_flag += 2
        elif abs(forward_count - backward_count) < count/2:
            bad_flag += 1
        else:
            good_flag += 1
'''        
ret, frame = video.read()
pframe = frame
skip_flag = 1
while(video.isOpened()):
    count = 0
    forward_count = 0
    backward_count = 0
    centroid = []
    frame_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: frame_expanded})
    try:
        font = ImageFont.truetype('arial.ttf', 50)
    except IOError:
        font = ImageFont.load_default()
    for sc in scores:
        for sc1 in sc:
            if sc1 > 0.80:
                count += 1
            else:
                break
    
    pil_image = Image.fromarray(np.uint8(frame)).convert('RGB')
    im_width, im_height = pil_image.size
    draw = ImageDraw.Draw(pil_image)
    
    # motion direction Start
    prev_area = new_area
    prev_centroid = new_centroid
    area = [] 

    #no_of_boxes = sum(sum(i > .80 for i in scores))
    print(count)
    for i in range(count):
        ymin, xmin, ymax, xmax = boxes[0][i]
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
        #(left, top), (left, bottom), (right, bottom), (right, top)
        area.append(abs(left*bottom + right*top - left*top - right*bottom)) #Shoelace formula
        centroid.append(((left + right)/2,(top+bottom)/2))
        #area.sort(reverse=True)
    print('area ',area)
    print('centoid ',centroid)

    #centroid comparison
    #length = max(len(area), len(prev_area))
    new_area = [0] * len(prev_centroid)
    new_centroid = [(0,0)] * len(prev_centroid)
    if skip_flag == 1:
        skip_flag = 0
        prev_centroid = centroid[:]
    for i in range(len(prev_centroid)):
        temp = prev_centroid[i]
        #min = (math.inf,math.inf)
        min = math.inf
        index = 0
        for j in range(len(centroid)):
            #diff = tuple(abs(np.subtract(temp, centroid[j])))
            diff = np.linalg.norm(np.array(temp) - np.array(centroid[j]))
            #print('Difference ',diff,end=' ')
            #print(min)
            if diff < min:
                min = diff
                index = j
            j += 1
        try:
            new_area[i] = area[index]
            new_centroid[i] = centroid[index]
            centroid.pop(index)
            area.pop(index)
            i += 1
        except:
            continue
        #print(i)

    new_area.extend(area)
    new_centroid.extend(centroid)
    print('New area ',new_area)
    print('Prev area ', prev_area)
    
    for i in range(len(prev_area)):
        if i >= len(new_area) or new_area[i] <= 0:
            break
        if new_area[i] > 0:
            if abs(prev_area[i] - new_area[i]) > area_diff_threshold:
                if prev_area[i] > new_area[i]:
                    forward_count += 1
                elif prev_area[i] < new_area[i]:
                    backward_count += 1
            else:
                still_count += 1
        i += 1
        
    if forward_count > 0 or backward_count > 0:
        #For one direction
        if ((forward_count == 0 and backward_count == count) or (forward_count == count and backward_count == 0)) and count <= max_people_count:
            good_flag += 2
        #For both directions
        elif forward_count == backward_count and count > max_people_count:
            bad_flag += 2
        #For imbalanced directions
        elif abs(forward_count - backward_count) < count*direction_sensitivity and count > max_people_count:
            bad_flag += 1
        else:
            good_flag += 1

    if still_count > count*still_sensitivity:
        bad_flag += 1
    
    agg_forward_count += forward_count
    agg_backward_count += backward_count

    print('Good ',good_flag,' Bad ',bad_flag, 'Forward Count ',forward_count, 'Backward Count ',backward_count, 'Still Count', still_count)
    '''
    if count > 0 and agg_count >= agg_count_window:
        agg_count = 0
        if agg_forward_count > agg_backward_count:
            direction_flag = 'A'
        else:
            direction_flag = 'T'
        agg_forward_count = 0
        agg_backward_count = 0
        
    agg_count += 1
    
    if count == 0:
        direction_flag = ''
    '''
    if reset_status_count >= reset_status_flags:
        if bad_flag > good_flag:
            draw.text((0,0),'Head Count '+str(count)+' '+direction_flag+' '+message,fill='red',font=font)
            stempede_flag = 1
            img_write_flag = 1
        else:
            stempede_flag = 0
            img_write_flag = 0
        good_flag = 0
        bad_flag = 0
        reset_status_count = 0
        still_count = 0
    reset_status_count += 1
    # Motion direction End
    
    if stempede_flag == 1:
        draw.text((0,0),'Head Count '+str(count)+' '+direction_flag+' '+message,fill='red',font=font)
        #pil_image.save(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f')+'.jpg')
    else:
        draw.text((0,0),'Head Count '+str(count)+' '+direction_flag,fill='red',font=font)
    np.copyto(frame, np.array(pil_image))
    # Draw the results of the detection (aka 'visulaize the results')
    if count > 0:
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.80,
            max_boxes_to_draw=1000)

    if stempede_flag == 1 and img_write_flag == 1:
        cv2.imwrite(os.path.join(PATH_TO_OUTPUT,datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f')+'.jpg'), frame)
        img_write_flag = 0
    
    #frame = cv2.resize(frame,(width,height))
    cv2.imshow('Object detector', cv2.resize(frame,(width,height)))
    #cv2.imshow('Masked', videoMasked)
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    pframe = frame
    ret, frame = video.read()

# Clean up
video.release()
cv2.destroyAllWindows()
