from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# model = load_model('projects_pretrained_change_setps_and_val_steps4(256size)_add_class3(epoch50_and_data(each100)flatten_adam_softmax)before_fine_tuning.h5')
model = load_model('projects_pretrained_change_setps_and_val_steps4(256size)_add_class3(epoch50_and_data(each100)flatten_Dropout_adam_softmax)before_fine_tuning.h5')
# img_path='/home/mskjhs/PycharmProjects/untitled2/projects/castle/test/korea/250korea.jpg'
# img_path='/home/mskjhs/PycharmProjects/untitled2/projects/castle/test/china/279china.jpg'
img_path='/home/mskjhs/PycharmProjects/untitled2/projects/dataset/alldata/13japan.jpg'
img =cv2.imread(img_path)
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = image.load_img(img_path, target_size=(256,256))
img_tensor=image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = preprocess_input(img_tensor)
pre=model.predict(img_tensor)
# img_edit = [pre[0][0], pre[0][1], pre[0][2]]
# print(['{:f}'.format(x) for x in img_edit])
label=np.array(["china","japan","korea"])
# merge=np.vstack((label,['{:f}'.format(x) for x in img_edit]))
merge=np.vstack((label,pre))
print(merge)
# pre=np.array(pre)
# x ="{:.4f}".format(pre)
# np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(img_tensor)})