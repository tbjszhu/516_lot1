File description:

1.visualWordsGen.py

This script generates descpritors for all the images in the train dataset. These descriptors are then used to train a kmeans model. Finally, the kmeans models will be saved for further usage.

run example: python visualWordsGen.py -n 100 -c 50 -d orb

-n is to control the quantity of the feature point in the descriptor generation algo
-c is to specify the quantity of the cluster while training the Kmeans model
-d is to specify the type of the descriptor

2. test.py

This script generates histogram (global descriptors) for the input image from test set or it loads from the existed histogram previously saved. Then, the pretrained kmeans model is loaded and used. The distance between the test_image and images from the data set will be calculated and be ranked(default: euclidean distance). The top 20 nearest neighbors in terms of distances from the target (index and image name) are returned and saved as the image result.

run example: python test.py -n 100 -c 50 -d orb -m 1 -i 266_c

-n,-c,-d as in the visualWordsGen.py
-i for the name of the test image
- m is to specify the running model. Here we use model 1 to output the ranking result for the input test image named by -i option. Other models are used to generate csv file/Prediction-Recall image for one/all images.

3. utils.py

Tool functions to be imported by visualWordsGen.py and test.py
