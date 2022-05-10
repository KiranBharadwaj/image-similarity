import pandas as pd
import os
from PIL import Image
import torch
from torchvision import models
from torchvision import transforms

from tqdm import tqdm

#directory structure
data_directory = "cars_data"
train_df = pd.read_csv(data_directory+"/cars_train_data.csv")
train_images = "cars_data/cars_train/cars_train"
input_dir_cnn = data_directory + "/images/input_images_cnn"

#Resizing all images to the size of 224 X 224 required by ResNet18 model
input_dim = (224,224)

os.makedirs(input_dir_cnn, exist_ok = True)

transformation_for_cnn_input = transforms.Compose([transforms.Resize(input_dim)])

for image_name in os.listdir(train_images):
    I = Image.open(os.path.join(train_images, image_name))
    newI = transformation_for_cnn_input(I)

    newI.save(os.path.join(input_dir_cnn, image_name))

    newI.close()
    I.close()


#Class for getting image embedding vectors after passing through ResNet18

#Each image is converted to a vector of size 512
class Img2VecResnet18():
    def __init__(self):
        self.device = torch.device("cpu")
        self.numberFeatures = 512
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor() 
        #Rescaling pixel values between 0 and 1. The values for mean and std was precomputed based on ImageNet dataset
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 

    def getFeatureLayer(self):
        cnnModel = models.resnet18(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512

        return cnnModel, layer
    #getVec method for getting the corresponding image embeddings
    def getVec(self, img):
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)
        def copyData(m, i, o): embedding.copy_(o.data)
        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()
        return embedding.numpy()[0, :, 0, 0]


#Creating vectors for every image in the train set
img2vec = Img2VecResnet18()
allVectors = {}

#for larger files we can sequentially write to memory
#since the dataset was small, I'm first storing the values in a dictionary
for image in tqdm(os.listdir(input_dir_cnn)):
    I = Image.open(os.path.join(input_dir_cnn, image))
    try:
        vec = img2vec.getVec(I)
        allVectors[image] = vec
        I.close()
    except:
        pass

#Persisting the vectors in harddisk in csv format

pd.DataFrame(allVectors).transpose().to_csv(data_directory + '/input_data_vectors.csv')