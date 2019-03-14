from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

# dimensions of our images
img_width, img_height = 300, 300

# load the model we saved
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")



for i in range(7):
    img = image.load_img('data/test/' + str(i+1) +'.jpg', target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)

    print(classes)
# predicting images


# print the classes, the images belong to