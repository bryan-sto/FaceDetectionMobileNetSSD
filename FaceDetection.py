from tensorflow.keras.models import load_model
import cv2
import time
import tensorflow as tf 
import numpy as np
facetracker = load_model('facetracker.h5')

new_frame_time = 0 
prev_frame_time = 0 

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    font = cv2.FONT_HERSHEY_SIMPLEX = 1
    abc = "FPS : "
    hehe = "coordinate : "
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        center = (sample_coords + sample_coords//2)
        cv2.putText(frame, abc, (5,20), font, 1, (139,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, fps, (50,20), font, 1, (139,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, hehe, (6,50), font, 1, (139,00,00), 1, cv2.LINE_AA)
        cv2.putText(frame, str(center), (105,50), font, 1, (139,00,00), 1, cv2.LINE_AA)
        
    
    cv2.imshow('EyeTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()