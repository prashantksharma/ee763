import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

model_path = os.path.join('snapshots', 'resnet50_csv_50.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'Pedestrian', 1: 'Biker'}

video_path = 'project.avi'
output_path = 'hyang_output_actual.avi'
fps = 6


vcapture = cv2.VideoCapture(video_path)

width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))  # uses given video width and height
height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
vwriter = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'MJPG'),fps, (width, height)) #

num_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
print("Number of Frames: ", num_frames)
print("Original Width, Height: ", width, height)


def run_detection_video(video_path):
    count = 0
    success = True
    start = time.time()
    while success:
        if count % 100 == 0:
            print("frame: ", count)
        count += 1  # see what frames you are at
        # Read next image
        success, image = vcapture.read()
        
        if success:
            
            # so we can keep orig image scale
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
             # preprocess image for network
            image = preprocess_image(image)
            image, scale = resize_image(image)
            
            # Do compute
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            
            # correct for image scale
            boxes /= scale
            
            
             # visualize detections
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.4:
                    break

                color = label_color(label)

                b = box.astype(int)
                draw_box(draw, b, color=color)

                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)
            
            vwriter.write(draw) # overwrites video slice


    vcapture.release()
    vwriter.release() # 
    end = time.time()
    
    print("Total Time: ", end - start)


run_detection_video(video_path)