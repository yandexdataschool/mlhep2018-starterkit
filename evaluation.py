import tables
from sklearn.metrics import roc_auc_score
import numpy as np
from collections import OrderedDict

model = # initialize your model here

train_f = tables.open_file('train.hdf5')


scores = OrderedDict([('acc@80',0.0), ('acc@50',0.0), ('acc@mean',0.0)])

accs = []
for i in range(train_f.root.data.shape[0]):
    labels = train_f.root.label[i]
    mask = labels > 0
    if mask.sum() > 0: # non-empty event
        preds = model.predict(train_f.root.data[i])[mask].flatten() # predictions must be already rounded to discrete values!
        labels = labels[mask].flatten()
        acc = (preds == labels).mean()
        accs.append(acc)

scores['acc@80'] = np.percentile(accs, 80)
scores['acc@50'] = np.median(accs)
scores['acc@mean'] = np.mean(accs)

for score_name, score in scores.items():
    output_str = "======= Score(" + score_name + ")=%0.12f =======" % score
    print(output_str)
