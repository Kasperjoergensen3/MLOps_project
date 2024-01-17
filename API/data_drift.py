import pandas as pd
import torch
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import *

class_dict = {0: "glioma", 1: "meningioma", 2: "no-tumor", 3: "pituitary"}
num_dict = {'glioma': 0, 'meningioma': 1, 'no-tumor': 2, 'pituitary': 3}

#Load ref data (train)
ref_data = torch.load("data/processed/train.pt")
features, labels = ref_data.tensors
features_np = features.numpy()
features_mean = features_np.mean(axis=(1,2,3))
features_contrast = features_np.std(axis=(1,2,3))
labels_np = labels.numpy()
#ref_df = pd.DataFrame({"feature_mean": features_mean, "label": [class_dict[label] for label in labels_np]})
ref_df = pd.DataFrame({"feature_mean": features_mean, "feature_contrast": features_contrast,
                        "label": [int(label) for label in labels_np]})

ref_df.to_csv("API/app/train_features.csv", index = False)

#Load current data
cur_data = pd.read_csv("API/app/prediction and feature_database.csv")
cur_data.columns = ["time", "feature_mean", "feature_contrast", "label"]
cur_data.drop(["time"], axis = 1, inplace=True)

#Generate report
report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
report.run(reference_data=ref_df, current_data=cur_data)
report.save_html('API/report.html')


#Load test data
#test_data = torch.load("data/processed/test.pt")
#features, labels = test_data.tensors
#features_np = features.numpy()
#features_mean = features_np.mean(axis=(1,2,3))
#features_contrast = features_np.std(axis=(1,2,3))
#labels_np = labels.numpy()
#test_df = pd.DataFrame({"feature_mean": features_mean, "feature_contrast": features_contrast, 
#                        "label": [int(label) for label in labels_np]})

#Generate report for test data
#report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
#report.run(reference_data=ref_df, current_data=test_df)
#report.save_html('API/testdata_report.html')

