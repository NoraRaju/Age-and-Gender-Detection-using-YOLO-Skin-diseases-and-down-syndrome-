import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import yaml
from yaml.loader import SafeLoader
import tensorflow.compat.v1 as tf

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml, gender_model_path, downsyndrome_model_path):
        # Load YAML file
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        # YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Gender model
        self.gender_model = load_model(gender_model_path)
        
        # Down syndrome model
        self.downsyndrome_model = tf.keras.models.load_model(downsyndrome_model_path)

    def predictions(self, image):
        row, col, d = image.shape
        # get the YOLO predictions from the image
        # step-1 convert image to square image (array)
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image
        # step-2 get predictions from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()  # detection or prediction from YOLO

        # non-maximum suppression
        # step1: filter detection based on confidence (0.4) and probability score(0.25)
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []
        # width and height of the image(input image)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # confidence of detection of face
            if confidence > 0.4:
                class_score = row[5:].max()  # maximum probability from 1 object
                class_id = row[5:].argmax()

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # construct bounding box from four values
                    # left, top, width and height
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])

                    # append values into list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # clean
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # NMS
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()

        # draw the bounding box
        for ind in index:
            # extract bounding box
            x, y, w, h = boxes_np[ind]
            bb_conf = int(confidences_np[ind] * 100)
            classes_id = classes[ind]
            class_name = self.labels[classes_id]
            colors = self.generate_colors(classes_id)
            x1 = int(y)
            y1 = int(x)
            x2 = int(w + x1)
            y2 = int(h + y1)

            roi = image[y1:y2, x1:x2]
            gender_label = ""
            age_label = ""
            downsyndrome_label = ""

            if (roi.shape[0] > 10) or (roi.shape[1] > 10):

                gender_label, age_label = self.predict_age_and_gender(roi)
                resized_image = cv2.resize(image, (300, 300))
                downsyndrome_label = self.predict_downsyndrome(resized_image)

            print(f"Predicted Gender: {gender_label}, Age: {age_label}, Down Syndrome: {downsyndrome_label}")  # Debugging print
            text = f'{class_name}: {bb_conf}% {gender_label}: {age_label}, Down Syndrome: {downsyndrome_label}'
            cv2.rectangle(image, (x, y), (x+w, y+h), colors, 2)
            cv2.rectangle(image, (x, y-30), (x+w, y), colors, -1)
            cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
            

        
        return image

    def generate_colors(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(colors[ID])

    def predict_age_and_gender(self, image):
        # Preprocess the image for gender model
        img = Image.fromarray(image)
        img = img.resize((128, 128))
        if img.mode != 'L':  # Convert to grayscale if not already
            img = img.convert('L')
        img = np.array(img)
        img = img.reshape(1, 128, 128, 1)  # Reshape for grayscale image (1 channel)
        img = img / 255.0
        
        # Predict gender and age
        gender_pred = self.gender_model.predict(img)
        
        gender_mapping = {1: 'Female', 0: 'Male'}
        gender = gender_mapping[round(gender_pred[0][0][0])]
        age = round(gender_pred[1][0][0])
        
        return gender, age
    
    def predict_downsyndrome(self, img):
        if img.shape[0] < 10 or img.shape[1] < 10:
          return "Invalid"
        # Preprocess the image if needed
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        #preprocessed_image = resized_image.astype('float32') / 255.0
        class_names = ['Yes', 'No', '{model_version}']
        # Make predictions
        prediction = self.downsyndrome_model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction[0])]
        return predicted_class

        # Assuming the model predicts a binary classification
        #if prediction[0][0] > 0.5:
            #return "Yes"  
        #else:
            #return "No "


tf.compat.v1.enable_eager_execution()

# Example usage
#yolo_model = YOLO_Pred("path/to/your/yolo_model.onnx", "path/to/your/data.yaml", "trained_model.h5", "{model_version}")

