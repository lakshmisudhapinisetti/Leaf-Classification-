from keras.models import load_model
from keras import utils as np_utils
from keras.preprocessing.image import load_img,img_to_array
import os
import numpy as np
import keras
import matplotlib.pyplot as plt

def load_data(datadir):
    imagelist = os.listdir(datadir)
    data_list = []
    label_list = []
    i = 0
    for image in imagelist:
        inner_path = os.path.join(datadir, image)
        img = load_img(inner_path,target_size=(224,224))
        x = img_to_array(img)
        data_list.append(x)
        label_list.append(int(image[0])-1)
        i = i+1
        print("%d loaded"%(i))
    print("All images have been loaded")
    return np.array(data_list),np.array(label_list)

def visualization(model, x_test, y_test, class_names):
    fig = plt.figure(figsize=(10, 8))
    y_test = np.dot(y_test, [0, 1, 2, 3, 4, 5]).astype(int)
    y_hat = model.predict(x_test)
    ## Only show 10 pictures at a time
    for i in range(10):
        ax = fig.add_subplot(5, 2, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_test[i]))
        pred_idx = np.argmax(y_hat[i])
        true_idx = y_test[i]
        ax.set_title("{}({})".format(class_names[pred_idx], class_names[true_idx]),
                     color=("green" if pred_idx == true_idx else "red"))
    plt.savefig("prediction")

    return

def confusion(model, x_test, y_test):
    y_truth = np.dot(y_test, [0, 1, 2, 3, 4, 5])
    y_hat = model.predict(x_test)
    y_pre = np.zeros([6, 6])
    for i in range(len(x_test)):
        m = int(y_truth[i])
        n = np.argmax(y_hat[i])
        y_pre[m, n] += 1
    return y_pre


model = load_model("C:/Users/sudha/OneDrive/Desktop/this/work_dirs/leaves8_model11.hdf5")

class_names =[ "Lotus magnolia leaves,Safflower Following Konoha", "maple leaves", "camphor leaves", "cypress leaves", "ginkgo"]
class_names_E = ["Lotus magnolia tree", "Redrlowered Loropetalum", "maple", "Camphora officinarum", "cypress", "ginkgo"]

testdir = "./test"
x_test, y_test = load_data(testdir)
y_test = keras.utils.np_utils.to_categorical(y_test, 6)
x_test = x_test.astype('float32') / 255.0

score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

# print(confusion(model,x_test,y_test))
# visualization(model,x_test,y_test,class_names_E)

