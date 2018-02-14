# 516_lot1: find similar images using global descriptors
todo list:
(1) descriptors generation
(2) kmeans cluserting to generate vocabulary (mean of each cluster) <--  we are now here.
(3) generate histogramme for each image in the base, save the histogrammes
(4) given a new test image, calculate hist, compare it to all the hists in the base
(5) get the ranking of closest hist
