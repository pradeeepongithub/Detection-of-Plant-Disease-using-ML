import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
import smtplib
# here the model will load
filename = 'plant_disease_classification_model.pkl'
model = pickle.load(open(filename, 'rb'))

# here the labels will load
filename = 'plant_disease_label_transform.pkl'
image_labels = pickle.load(open(filename, 'rb'))

# Dimension of resized image, means cropped image
DEFAULT_IMAGE_SIZE = tuple((256, 256))

# here the image will be converted to
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def predict_disease(image_path):
    image_array = convert_image_to_array(image_path)
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image,0)
    plt.imshow(plt.imread(image_path))
    result = model.predict_classes(np_image)
    print((image_labels.classes_[result][0]))
    return image_labels.classes_[result][0]
def send_email(content):
    to='kumarpradeep9435@gmail.com'
    server = smtplib.SMTP('smtp.gmail.com', 587)  # simple mail transfer protocol
    server.ehlo()
    server.starttls()
    server.login('...Sensder Mail..., '...password...')
    server.sendmail('...Sensder Mail...', to, content)
    server.close()
# here the path will be given to image to predict the disease
predicted=predict_disease('0b13b997-9957-4029-b2a4-ef4a046eb088___UF.GRC_BS_Lab Leaf 0595.JPG')
if(predicted=='Potato___Late_blight'):
    content='Subject: Disease has been detected\nAgricultural Drone has detected '+predicted+' to prevent you can\nUse Bacillus subtilis or copper-based fungicides\nUse certified pathogen-free seeds\nRemove bottom leaves that are too close'
    send_email(content)
elif(predicted=='Potato___Late_blight'):
    content='Subject: Disease has been detected\nAgricultural Drone has detected '+predicted+' to prevent you can\nUse Foliar sprays\nUse healthy seeds\nAvoid excessive fertilization with nitrogen'
    send_email(content)
elif(predicted=='Tomato__Tomato_YellowLeaf__Curl_Virus'):
    content='Subject: Disease has been detected\nAgricultural Drone has detected '+predicted+' to prevent you can\nUse Control whitefly population\nPractice crop rotation\nUse net to cover seedbeds'
    send_email(content)
elif(predicted=='Tomato_Bacterial_spot'):
    content='Subject: Disease has been detected\nAgricultural Drone has detected '+predicted+' to prevent you can\nUse Copper-containing bactericides\nUse resistant varieties\nAvoid injuring transplant'
    send_email(content)
elif(predicted=='Tomato_Early_blight'):
    content='Subject: Disease has been detected\nAgricultural Drone has detected '+predicted+' to prevent you can\nUse Bacillus subtilis or copper-based fungicides\nUse drip irrigation system\nUse pathogen free seeds'
    send_email(content)
elif(predicted=='Tomato_Late_blight'):
    content='Subject: Disease has been detected\nAgricultural Drone has detected '+predicted+' to prevent you can\nTry to keep plant in dry through good drainage\nUse plant fortifier'
    send_email(content)