import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
y_test = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # label
y_score = [24.3635193888, 28.1578203521, 33.4888123156, 31.8134211098, 20.5152435803
, 19.2871678992, 19.8124604819, 18.2740310065, 9.0487417878, 14.463933792] # prediction

precision, recall, _ = precision_recall_curve(y_test, y_score)
print precision
print recall
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
area = auc(recall,precision)
print area
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()