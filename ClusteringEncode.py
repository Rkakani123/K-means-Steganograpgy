import cv2
import numpy as np
from ClusteringLSB import ClusterImages, LSBteq

input_file = input("Enter Input File : ")
data = input("Enter Your Secret Data :")
output_file = input("Enter Name For Output File (Without Extension):")
output_file = output_file + ".png"

image = cv2.imread(input_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image1 = image.reshape(image.shape[0] * image.shape[1], 3)
labels, centers, bar, maxCluster = ClusterImages(image1).getCluster()

labels = labels.reshape(image.shape[0], image.shape[1])
index1 = np.argwhere(labels == maxCluster).tolist()
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
steg = LSBteq(image, index1)
cv2.imwrite(output_file, steg.encode_text(data))


