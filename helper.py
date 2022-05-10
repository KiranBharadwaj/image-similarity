import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from PIL import Image

import torch
from torchvision import models
from torchvision import transforms

spark = SparkSession.builder.getOrCreate()

data_directory = "cars_data"
input_df = spark.read.option('inferSchema', True).csv(data_directory + '/input_data_vectors.csv')

vector_columns = input_df.columns[1:]
assembler = VectorAssembler(inputCols=vector_columns, outputCol="features")

output = assembler.transform(input_df)
output = output.select('_c0', 'features')


brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", numHashTables=200, bucketLength=2.0)
model = brp.fit(output)
lsh_df = model.transform(output)

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

###Converting image to vector


def return_closest_images(filename):
    img2vec = Img2VecResnet18()
    I = Image.open(filename)
    test_vec = img2vec.getVec(I)
    I.close()
    test_vector = Vectors.dense(test_vec)
    result = model.approxNearestNeighbors(lsh_df, test_vector, 5)
    return list(result.select('_c0').toPandas()['_c0'])
