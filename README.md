# Traffic-Sign-Detection-using-Faster-R-CNN
Turkish Traffic Sign Detection and Classification using Faster R CNN Inception v2

#Step by Step guide to install and run the files needed for WINDOWS

# 1-) Install Models from https://github.com/tensorflow/models
Create a file and name it as 'tensorflow1' in your C:/
Unzip the files you downloaded and name unzipped file as 'models'
When these steps are finished there should be a path like this 'C:\tensorflow1\models\research\object_detection'


# 2-) Download Tensorflow FASTER R-CNN API: faster_rcnn_inception_v2_coco from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
Unzip the file and copy it to C:\tensorflow1\models\research\object_detection folder or copy it to place where unzipped the models file in step 1.


# 3-) Download one of the object_detection.zip or object_detection.rar
Unzip the files and copy them to the C:\tensorflow1\models\research\object_detection or copy it to place where unzipped the models file in step 1. Replace all files.

# 4-) Download and install Anaconda from https://www.anaconda.com/

# 5-) Create a virtual environment of anaconda. Type this command to cmd. conda create -n tensorflow1 pip python=3.6

# 6-) Download and install python libraries needed. First, activate your virtual environment using this command in cmd; activate tensorflow1
  It should be look like this; (tensorflow1) C:\>
   pip install --ignore-installed --upgrade tensorflow-gpu
   conda install -c anaconda protobuf
   pip install pillow
   pip install lxml
   pip install jupyter
   pip install matplotlib
   pip install pandas
   pip install opencv-python
  
  If you get any errors while installing libraries, try to install them from another source.
  If you want to work with cpu, install tensorflow for cpu.
  Make sure all libraries installed successfully
# 7-) !! IMPORTANT. type this command to cmd; set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
  If you crated your models file in anoter location, change the directory in the command.
  WHENEVER YOU ACTIVATE THE VIRTUAL ENVIRONMENT tensorflow1, YOU NEED TO TYPE THIS COMMAND AGAIN !!!!.
 
# 8-) Compile the protobufs, type this command to change directory cd C:\tensorflow1\models\research, 
 Then type this to cmd, protoc -- python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto

# 9-) Execute the setup files. Type these command to cmd;
      first type this to cmd: python setup.py build
      second type this to cmd: python setup.py install
      
# 10-) Download the train and test set, and their labels from https://drive.google.com/file/d/1m6I1qJ1r9JtDTBAfm8YlSA81zH40vQPL/view
      Unzip the file and create in the C:\tensorflow1\models\research\object_detection\images directory, create a new file named train         and create another file named test.
      Put train photos and their xml labels into train file and test photos and their xml labels into test file.
      
      # !! YOU CAN CREATE YOUR OWN DATASET USING LABEL IMG IN THE LINK https://github.com/tzutalin/labelImg 
      
      
# 11-) To create xml files of all labeled images, make sure you are in tensorflow1 environment and in the object_detection directory in cmd
       Run this command: python xml_to_csv.py
       It should be look like this: (tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
       
# 12-) Then, go back to the object_detection file and open generate_tfrecord.py

    This is how many class you have in your data. You can change class names arbitrary (instead of class1, type something of your own class names)
    def class_text_to_int(row_label):
    if row_label == 'class1':
        return 1
    elif row_label == 'class2':
        return 2
    elif row_label == 'class3':
        return 3
    elif row_label == 'class4':
        return 4
    elif row_label == 'class5':
        return 5
    elif row_label == 'class6':
        return 6
    else:
        None
        
     You must change this according to your own dataset. For example, I have only 1 traffic sign and it looks like this;
     if row_label == 'class1':
        return 1
     else:
        None
        
 # 13-) Then, open cmd again and activate tensorflow1 environment and go to the directory object_detection using cd command
        Type this command; python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
        Then type this command; python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
        
        This is how you  crate your data records to give as input.
        
  # 14-)  In the object_detection/training folder open create a new file as 'labelmap.pbtxt', then open it
          Type your own items. The number of items is how many classes you have in your own dataset.
          For example, if you have 2 classes, type something like this;
          item {
          id: 1
          name: 'class1'
          }
 
          item {
          id: 2
          name: 'class2'
          }
          !!WARNING, NAMES AND ITEM COUND SHOULD BE EQUAL TO THE NUMBERS AND NAMES YOU DEFINED IN STEP 12
          
  # 15-) To start training, you must do this;
         Go to the C:\tensorflow1\models\research\object_detection\samples\configs
         Copy faster_rcnn_inception_v2_pets.config and paste it to object_detection\training folder.
         Open this folder with text editor.
         
         Change these lines;
         
         LINE 9: TYPE HOW MANY CLASSES YOU HAVE
         
         LINE 110: it should be like this;  fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
         you can change directory to where your faster r cnn file is;
         
         LINE 126 AND 128: input_path : "C:/tensorflow1/models/research/object_detection/train.record"
                           label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
 
        
         LINE 132: TYPE HOW MANY TEST EXAMPLES YOU HAVE IN YOUR images/test folder. (TYPE HOW MANY .JPG FILE YOU HAVE)
         
         LINE 140 AND 142: input_path : "C:/tensorflow1/models/research/object_detection/test.record"
                           label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

# 16-) I KNOW IT TOOK SO LONG BUT BE PATIENT.

       TO START TRAINING, TYPE THIS COMMAND TO CMD; python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
       
       WAIT FOR A WHILE AND WAIT UNTIL YOUR TRAINING ENDS. YOU CAN END YOUR TRAINING ANYTIME. IT PUTS CHECKPOINTS WHILE TRAINING.
       
 # 17-) CREATE INFERENCE GRAPH, TYPE THIS COMMAND TO CMD BUT BEFORE RUNNING THE COMMAND, READ THE WARNING BELOW.
        python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
        
        WARNING:
        XXXX SHOULD BE THE NUMBER OF YOUR LAST CHECKPOINT
        YOUR LAST CHECKPOINT IS in training folder. Replace the biggest number of checkpoints with XXXX in the command.
        
  # 18-) To test your model, open cmd and tpye idle
         After idle opened, Select the Object_detection_image.py from the files at upper-left corner.
         
         In the Object_detection_image.py change NUM_CLASSES variable and type how many classes you have
         You can change test image.
         
