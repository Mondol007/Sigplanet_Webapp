# SigPlanet

![Workflow](https://github.com/Mondol007/Sigplanet_Webapp/blob/40c97d98bf48e263d33dffd0cff7a8515983f39d/Final-1.png)


## **Overview:**
- Developed a hand gesture recognition system using Mediapipe and LSTM-based deep learning for alphabet and gesture prediction in real time.

## **Data Preprocessing:**
- Utilized Mediapipe to detect hand landmarks and extract 3D coordinates.
- Processed video input via RTSP and converted it for real-time analysis.
- Stored keypoints data as numpy arrays for model training, with labels mapped to each gesture.

## **Methodology:**
- Defined a sequential LSTM model with multiple LSTM layers and dense layers.
- Trained the model with categorical cross-entropy loss and optimized using Adam.
- Implemented real-time prediction, updating a sentence display based on gesture detection confidence.
  
## **Results:**
- Predicted gesture labels were displayed on screen in real-time, with high-accuracy predictions forming sentences.

