# 516_lot1: find similar images using global descriptors
todo list:<br> 
(1) descriptors generation <br> 
(2) kmeans clustering to generate vocabulary (mean of each cluster) 
(3) generate histogramme for each image in the base, save the histogrammes <br> 
(4) given a new test image, calculate hist, compare it to all the hists in the base <br> 
(5) get the ranking of closest hist <br> 
(6) changes parameters and run tests <br>


Guide to use: <br>

  (1)first, run 
  ```python
  python visualWordsGen.py
  ```
  this code will generate descpritors for the train dataset (if descpts_addr = '') and use them to train a kmeans model (parameters can be <br>
  configurated in line 51), then the kmeans models (~ 1 hr for training) will be saved for further usage. <br>
   
  (2) second, test for a target image, run 
  ```python 
  python test.py
  ```
  this code will generate histogram (global descriptors) for each image from trainset (if hist_addr = '', or it will load <br> from the existed hists previously saved (the pretrained kmeans models will be loaded and used). <br>
  Then, we just calculate the distance (default: euclidean distance), return the top 10 nearest neighbors in terms of distances from the target (index and image name).
