import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
y_test = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # label
y_score = [8001.998002, 8001.998002, 8001.998002, 8001.998002, 8001.998002
, 34.445225163, 20.5347787411, 32.7220393887, 15.0246640715, 42.4370951403] # prediction

precision, recall, _ = precision_recall_curve(y_test, y_score)
print precision
print recall
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
area = auc(recall, precision)
print area
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()