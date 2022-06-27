# Audio-vision-scene-classificatioin
Scene classification using both audio and vision feature.  
we use openl3 to extract the feature from both audio and video.  
The model backbones are MLPs or LSTM.
## Early Fusiion
run the following command to conduct early fusion of audio and video feature.

     python train.py --model early_fusion

## Late Fusiion
run the following command to conduct late fusion of audio and video feature (decision level).

     python train.py --model late_fusion

## LSTM backbone
run the following command to conduct LSTM (mid fusion).

       python train_only.py
       
# Results
![image]:
