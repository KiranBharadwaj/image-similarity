## Distributed Image Similarity Detection with Deep Learning and PySpark

Humans have a natural instinct for recognizing images that are similar to one another, but it is a difficult computing operation. It becomes considerably more difficult when the project grows in size. In our project, we used Locality Sensitive Hashing algorithm along with approximate nearest neighbor search to tag similar car photos. To translate visual data into a numerical vector representation, we use deep learning. Specifically, we applied the embedding generated from the final layer of ResNet18 model to categorize the vectors.  The generated vectors will be subjected to PySpark's LSH method, which will allow us to locate comparable images given a fresh input image.

On a high level, our project resembles the algorithms used for image similarity identification by photo-sharing apps like Instagram and Pinterest. Our project also shows how PySpark's scalability can aid in scaling a deep learning model.
## Dataset

We are using [Stanford AI Car dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html). Our code is agnostic to the dataset used and can be easily extended to other datasets provided the same directory structure is followed.

The dataset for this task can be downloaded using the below command and unzipped to cars_data/cars_train
```
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
wget https://github.com/BotechEngineering/StanfordCarsDatasetCSV/blob/main/cardatasettrain.csv
```
## Tech
- PyTorch (CPU version is sufficient)
- Spark 3.1.2
- PySpark
- Jupyter Notebook
- Streamlit 

## Results

YouTube demo of UI [here](https://www.youtube.com/watch?v=oM0cOoeXVzI)

## Installation

Install the dependencies using the requirements.txt file

```sh
pip install -r requirements.txt
```
Before running any of the below commands, start a spark-shell

## Creating the embeddings

```sh
python embedding_generator.py
```

## Starting the Jupyter Notebook

```sh
jupyter notebook
```
## Running the UI

```sh
streamlit run ui.py
```
## License
MIT


