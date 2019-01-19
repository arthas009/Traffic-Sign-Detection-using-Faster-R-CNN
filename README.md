# Traffic-Sign-Detection-using-Faster-R-CNN
Turkish Traffic Sign Detection and Classification using Faster R CNN Inception v2

# Step by Step guide Installation on Windows. It's a long road, so please stay in patient and do whatever I say in the steps.
# (ADIM ADIM ANLATIM. DEDİĞİM HERŞEYİ HARFİ HARFİNE UYGULAYIN)

# Youtube link here; https://www.youtube.com/watch?v=hfUQ3X_CFLE. I tried to explaing as possible as I can. Sorry for my bad speech by the way :)


# 1-) Install Models from https://github.com/tensorflow/models as .zip file
# 1-) (https://github.com/tensorflow/models adresindeki dosyayı .zip olarak indirin)
    Create a file and name it as 'tensorflow1' in your C:/
    Unzip the files you downloaded and name unzipped file as 'models'
    When these steps are finished there should be a path like this 'C:\tensorflow1\models\research\object_detection'
    (C:/ DİZİNİNDE 'tensorflow1' isimli bir dosya açın ve models klasörünü oraya taşıyın
     Tüm bunların sonunda 'C:\tensorflow1\models\research\object_detection' isimli bir klasöre gidiyo olabilmeniz lazım.)


# 2-) Download Tensorflow FASTER R-CNN API: faster_rcnn_inception_v2_coco from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.
# 2-) https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md ' dan Tensorflow FASTER R-CNN API: faster_rcnn_inception_v2_coco modelini indirin.
    Unzip the file and copy it to C:\tensorflow1\models\research\object_detection folder or copy it to place where unzipped the models     file in step 1.
    (ZIP dosyasındakileri çıkartın ve C:\tensorflow1\models\research\object_detection klasörüne veya klasörleri nerde oluşturduysanız       oraya atın)

# AFTER 2 STEPS, YOUR FILE SHOULD LOOK LIKE THIS
# 2 ADIMIN SONUNDA DOSYANIZIN ŞU TARZ GÖZÜKÜYO OLMASI LAZIM
![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/img/modelsfile.JPG)

# 3-) Download one of the object_detection.zip or object_detection.rar
# 3-) YUKARDAKİ object_detection.zip veya object_detection.rar dosyalarından birini indirin.
    Unzip the files and copy them to the C:\tensorflow1\models\research\object_detection or copy it to place where unzipped the models     file in step 1. Replace all files.
    (DOSYALAR İNDİKTEN SONRA ZİP VEYA RARDAN ÇIKARTIN VE C:\tensorflow1\models\research\object_detection klasörüne atın)
# AFTER 3. STEPS, YOUR FILE SHOULD LOOK LIKE THIS
# 3. ADIMDAN SONRA DOSYANIZIN ŞUNUN GİBİ GÖZÜKÜYO OLMASI LAZIM
![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/img/object_detectionfile.JPG)

# 4-) Download and install Anaconda from https://www.anaconda.com/
# 4-) Anacondayı indirin https://www.anaconda.com/

# 5-) Create a virtual environment of anaconda. Type this command to cmd. conda create -n tensorflow1 pip python=3.6
# 5-) Anacondayı kurduktan sonra cmd'ye conda create -n tensorflow1 pip python=3.6 yazarak sanal ğython environmenti oluşturun.

# 6-) Download and install python libraries needed. First, activate your virtual environment using this command in cmd; activate tensorflow1.
# 6-) Aşağıdaki Python kütüphanelerini kurun. Kendiniz dışardan başka komutlada kurabilirsiniz. tensorflow1 isimli environmentin açık olması lazım. Bunun için ilk önce cmd ye activate tensorflow1 yazın.
    It should be look like this; (tensorflow1) C:\>
    1-) pip install --ignore-installed --upgrade tensorflow-gpu
    2-) conda install -c anaconda protobuf
    3-) pip install pillow
    4-) pip install lxml
    5-) pip install jupyter
    6-) pip install matplotlib
    7-) pip install pandas
    8-) pip install opencv-python
  
    If you get any errors while installing libraries, try to install them from another source.
    If you want to work with cpu, install tensorflow for cpu.
    Make sure all libraries installed successfully
    (tensorflow'u cpu da kullanmak istiyorsanız gpu yerine cpu versiyonunu indirin. Tüm kütüphanelerin düzgün indiğinden emin olun)
  
# 7-) !! IMPORTANT. Type command below on cmd to set your PYTHONPATH:
# 8-) !! ÖNEMLİ, Cmd ye pythonpathi  belirtmeniz lazım. Aşağıdaki komutu cmd'ye yazın.
    cmd command: set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
    
    If you crated your models file in anoter location, change the directory in the command.
    WHENEVER YOU ACTIVATE THE VIRTUAL ENVIRONMENT tensorflow1, YOU NEED TO TYPE THIS COMMAND AGAIN !!!!.
    (Eğer dosyaları başka konumda açtıysanız o konumdaki models\research\slim adresini verin
    !!! CMD Yİ TEKRARDAN HER AÇTIĞINIZDA PYTHONPATHİ TEKRARDAN BELİRLEMENİZ LAZIM. ONUN İÇİN YUKARDAKİ KOMUTU BİR YERE KAYDETSENİZ İYİ     OLUR)
  
 ![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/img/activateenvandpythonpath.JPG)

# 8-) Compile the protobufs, type this command to change directory cd C:\tensorflow1\models\research, 
# 8-) Protobufları derleyin. Bunun için aşağıdaki komutu yazın. Komutu yazmadan önce, cmd'de cd kullanarak dizinizi C:\tensorflow1\models\research olarak ayarlayın. Eğer dosyaları farklı konumda açtıysanız ona göre dizini değiştirin.
        Then type this to cmd;
    cmd command: 
    protoc -- python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto  .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto

# 9-) Execute the setup files. Type these commands to cmd;
# 9-) Kurulum dosyalarını çalıştırın. Aşağıdaki komutları sırası ile yazın.
      First type this to cmd:  python setup.py build
      Second type this to cmd:  python setup.py install
      
# 10-) Download the train and test set, and their labels from https://drive.google.com/open?id=1mcevi7aFmhip2W9vRNtVA8sA-jfq8S8W
# 10-) https://drive.google.com/open?id=1mcevi7aFmhip2W9vRNtVA8sA-jfq8S8W adresinden Veri setini indirin. Xml dosyaları onların labelları olarak yanlarında bulunmaktadır.
      This is my own dataset. I collected it from google maps and i labeled them with LABELIMG in the link below.
      You can reorder the train and test sets arbitrary or can make your own dataset.
      Unzip the file and create in the C:\tensorflow1\models\research\object_detection\images directory, create a new file named train         and create another file named test.
      Put train photos and their xml labels into train file and test photos and their xml labels into test file.
      
      # !! YOU CAN CREATE YOUR OWN DATASET USING LABEL IMG IN THE LINK https://github.com/tzutalin/labelImg 
      # !!While labeling your images to xml file, make sure you are using the same names and use upper charecters and dont't forget the           names because we are goint the use them in the steps below.
      (Bu benim kendi verisetim. Kendim google maps üzerinden topladım. LABELIMG programı ile tek tek etiketleyip XML dosyalarını      çıkarttım. Kendi datasetinizi oluşturup onları kullanabilirsiniz. Fakat etiketlerken kullandığınız sınıf isimlerini unutmayın. Onlar ilerde lazım olacak. Eğitim datasetini C:\tensorflow1\models\research\object_detection\images dizininde train isimli bir dosya oluşturup oraya atın. Test içinde yine aynı dizinde test isimli bir dosya oluşturup test datasetini oraya atın.)

# YOU CAN FIND IMAGE LABELING PROGRAM IN REPO 'LABELIMG.ZIP'

# UNTIL HERE YOUR images/train file should be something like this, same for test
# BU NOKTAYA KADAR, images/train dosyanız aşağıdaki gibi fotoğraflardan ve XML dosyalarından oluşmalıdır. Test klasörüde aynı şekilde.
![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/img/sampletraininingset.JPG)


# EXAMPLE LABEL IMAGING
# ÖRNEK FOTOĞRAF ETİKETLEME İŞLEMİ
![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/img/samplelabelimage.JPG)

# DON'T FORGET THE CLASSNAMES YOU GIVE WHILE LABELING. WE ARE GOING TO USE SAME NAMES IN THE STEPS BELOW !!
# YAZDIĞINIZ SINIF İSİMLERİNİ UNUTMAYIN. ONLARI İLERDE KULLANACAĞIZ !!

# 11-) To create .csv files of all labeled images, make sure you are in tensorflow1 environment and in the object_detection directory in cmd
# 11-) Etiketleri .csv dosyasına dökmek için aşağıdaki kodu çalıştırın. CSV dosyaları images klasöründe belirecektir.

       Run this command: python xml_to_csv.py
       It should be look like this: (tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
       
# 12-) Then, go back to the object_detection file and open generate_tfrecord.py
# 12-) object_detection klasörüne tekrar gidin  ve generate_tfrecord.py dosyasını bir text editörü ile açın
    This is how many class you have in your data. You can change class names arbitrary (instead of class1, type something of your own class names)
    
    (Burada sınıf sayınız, isimleri ve numaraları bulunmaktadır. Keyfi olarak düzenlebilirsiniz. Yani kendi verisetinizdeki sınıf sayısına göre isimleri ve numaraları düzenleyin).
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
     (Örnek, verisetinizde 1 sınıf var ise, adıda class1 ise şöyle yazın)
     if row_label == 'class1':
        return 1
     else:
        None
        
 # 13-) Then, open cmd again and activate tensorflow1 environment and go to the directory object_detection using cd command
 # 13-) CMD ye tekrar gelin ve aşağıdaki kodları sırasıyla yazarak input recordlarını derleyin.
        Type this command; python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --  output_path=train.record
        Then type this command; python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
        
        This is how you  crate your data records to give as input.
        
  # 14-)  In the object_detection/training folder open create a new file as 'labelmap.pbtxt', then open it
  # 14-)  object_detection/training dosyasını açın ve labelmap.pptxt isimli bir dosya oluşturun. Eğer bu dosya var ise dosyayı açın.
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

          (generate_tfrecords.py dosyasındaki gibi sınıf isimlerini ve numaralarını kendi verisetinize uyarlı bir şekilde sırası ile             yukardakine benzer şekilde bu dosyaya yazın)
   
   
  # EXAMPLE generate_tfrecord.py and labelmap.pbtxt
  # ÖRNEK generate_tfrecord.py dosyası ve labelmap.pbtxt dosyası
  ![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/img/generatetfrecord.JPG)
  
  ![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/img/labelmap.pbtx.JPG)
   
   
  # 15-) To start training, you must do this;
  # 15-) Eğitimi başlatmak için bunu yapmalısınız.
         Go to the C:\tensorflow1\models\research\object_detection\samples\configs
         Copy  faster_rcnn_inception_v2_pets.config and paste it to object_detection\training folder.
         Open this file (faster_rcnn...... .config)  with text editor.
         
         (C:\tensorflow1\models\research\object_detection\samples\configs dosyasına gidin ve faster_rcnn_inception_v2_pets.config
         dosyasını object_detection\training dizinine kopyalayın ve kopyaladığınız dosyayı text editörü ile açın.)
         
         
         Change these lines;
         
         LINE 9 (SATIR 9): TYPE HOW MANY CLASSES YOU HAVE (KAÇ SINIFINIZ OLDUĞUNU YAZIN)
         
         LINE 110 (SATIR 110) : it should be like this;  fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
         you can change directory to where your faster r cnn file is; 
         (Şöyle gözükmeli fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt" veya faster r cnn modeliniz nerdeyse o dosyanın yolunu verin)
         
         
         LINE 126 AND 128: input_path : "C:/tensorflow1/models/research/object_detection/train.record"
                           label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
 
        
         LINE 132: TYPE HOW MANY TEST EXAMPLES YOU HAVE IN YOUR images/test folder. (TYPE HOW MANY .JPG FILE YOU HAVE) 
         (KAÇ ADET TEST FOTOĞRAFINIZ VARSA ONUN SAYISINI YAZIN)
         
         LINE 140 AND 142: input_path : "C:/tensorflow1/models/research/object_detection/test.record"
                           label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

# 16-) I KNOW IT TOOK SO LONG BUT BE PATIENT.
# 16-) ÇOK UZUN SÜRDÜ BİLİYORUM. SABREDİN.
    !! Before start training, open training folder and delete these selected or whatever you have in your folder. Only 4 file unselected in the photo should stay in the file.
    !! BAŞLAMADAN ÖNCE training KLASÖRÜNDEN AŞAĞIDAKİ SEÇİLİ OLMAYAN DOSYALAR DIŞINDAKİLERİ SİLİN
![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/img/trainingfile.JPG)  

    !! Then go to the inference_graph file and delete all files in it
    !! GERİ GİDİN VE inference_graph DOSYASINDAKİ HERŞEYİ SİLİN

![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/img/inferencegraph.jpg)  

       TO START TRAINING, TYPE THIS COMMAND TO CMD; python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
       
       WAIT FOR A WHILE AND WAIT UNTIL YOUR TRAINING ENDS. YOU CAN END YOUR TRAINING ANYTIME. IT PUTS CHECKPOINTS WHILE TRAINING.
   
      (EĞİTİME BAŞLAMAK İÇİN CMD YE ŞUNU YAZIN VE BEKLEYİN: python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
      EĞİTİM BİTENE KADAR CMD Yİ KAPATMAYIN. EĞİTİMİ İSTEDİĞİNİZ ZAMAN  CMD Yİ KAPATARAK BİTİREBİLİRSİNİZ. MODEL EĞİTİLİRKEN CHECKPOİNTLER OLUŞMAKTADIR)
 ![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/img/training.JPG)  
 
     If you want to keep tracking your loss values on tensorboard, open another cmd, set pythonpath and go to the object_detection file      and type this tensorboard --logdir=training
     then go to your web browser and type localhost:6006
    (TENSORBOARD İLE LOSS DEĞERLERİNİ TAKİP ETMEK İÇİN BAŞKA BİR CMD EKRANI AÇIN, PYTHONPATHİ BELİRLEYİN, object_detection KLASÖRÜNE CD KOMUTU İLE GİDİN VE tensorboard --logdir=training YAZIN. KOMUT BİTİNCE WEB BROWSERİNİZİ AÇIN VE localhost:6006 YAZIN)
 # 17-) CREATE INFERENCE GRAPH, TYPE THIS COMMAND TO CMD BUT BEFORE RUNNING THE COMMAND, READ THE WARNING BELOW.
 # 17-) INFERENCE GRAPH OLUŞTURUN. AŞAĞIDAKİ KOMUTU CMD DE ÇALIŞTIRIN FAKAT ÇALIŞTRMADAN ÖNCE UYARIYI OKUYUN.
        python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
        
        WARNING:
        XXXX SHOULD BE THE NUMBER OF YOUR LAST CHECKPOINT
        YOUR LAST CHECKPOINT IS in training folder. Replace the biggest number of checkpoints with XXXX in the command.
        
        (XXXX YAZAN YERE training dosyasındaki en sonun checkpoint dosyasının numarasını verin)
  # 18-) To test your model, open cmd and tpye idle
  # 18-) Modelinizi test etmek için cmd ye idle yazın ve sol üst köşeden Object_detection_image.py dosyasını seçin.
         After idle opened, Select the Object_detection_image.py from the files at upper-left corner.
         
         In the Object_detection_image.py change NUM_CLASSES variable and type how many classes you have
         You can change test image.
         
         (Object_detection.py dosyasındaki, NUM_CLASSES değişkenine sınıf sayınızı yazın ve test etmek istediğin değişkeninde yolunu aşağısındaki değişkene verin)
         
# RESULT OF MY OWN DATASET
# KENDİ VERİSETİMİN SONUÇLARI

![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/Result/result1.JPG)
![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/Result/result2.JPG)
![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/Result/result3.JPG)
![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/Result/testtgirisiolmayan2.JPG)
![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/Result/test1all.jpg)
![alt text](https://github.com/arthas009/Traffic-Sign-Detection-using-Faster-R-CNN/blob/master/Result/testall1correcting.JPG)
