import matplotlib.pyplot as plt
import sklearn
import sys
from tsfresh import extract_features
from tsfresh import extract_relevant_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def quaternion_to_euler(x, y1, z, w):
    import math
    t0 = +2.0 * (w * x + y1 * z)
    t1 = +1.0 - 2.0 * (x * x + y1 * y1)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y1 - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y1)
    t4 = +1.0 - 2.0 * (y1 * y1 + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

if __name__ == '__main__':
    train = pd.read_csv('C:/Users/bhunt\OneDrive/Documents/KaggleData/Robot Career Competition/career-con-2019/X_train.csv')
    y = pd.read_csv('C:/Users/bhunt\OneDrive/Documents/KaggleData/Robot Career Competition/career-con-2019/y_train.csv')
    test = pd.read_csv('C:/Users/bhunt\OneDrive/Documents/KaggleData/Robot Career Competition/career-con-2019/X_test.csv')
    y = y.drop(["series_id", "group_id"], axis=1)
    labels = LabelEncoder()
    labels.fit(list(y.values))
    y = labels.transform(list(y.values))
    train_extra_vars = pd.DataFrame()
    angle_x = []
    angle_y = []
    angle_z = []
    linear_acc_mag = []
    angular_vel_mag = []
    angle_mag = []
    acc_vs_vel = []
    for index, row in train.iterrows():
        xang, yang, zang = quaternion_to_euler(row['orientation_X'], row['orientation_Y'], row['orientation_Z'], row['orientation_W'])
        accel = (row['linear_acceleration_X']**2 + row['linear_acceleration_Y']**2 + row['linear_acceleration_Z']**2)**0.5
        linear_acc_mag.append(accel)
        vol_mag = (row['angular_velocity_X'] ** 2 + row['angular_velocity_Y'] ** 2 + row['angular_velocity_Z'] ** 2) ** 0.5
        angular_vel_mag.append(vol_mag)
        angle_mag.append((xang**2 + yang**2 + zang**2)**0.5)
        acc_vs_vel.append(accel/vol_mag)
        angle_x.append(xang)
        angle_y.append(yang)
        angle_z.append(zang)

    train['angle_X'] = np.asarray(angle_x)
    train['angle_Y'] = np.asarray(angle_y)
    train['angle_Z'] = np.asarray(angle_z)
    train['angle_mag'] = np.asarray(angle_mag)
    train['acc_vs_vel'] = np.asarray(acc_vs_vel)
    train['linear_acc_mag'] = np.asarray(linear_acc_mag)
    train['angular_velocity_mag'] = np.asarray(angular_vel_mag)
    train['series_id'] = test['series_id']
    train['measurement_number'] = test['measurement_number']

    y = pd.Series(np.array(y).reshape(len(y)))
    train = train.drop(["row_id"], axis=1)
    test = test.drop(["row_id"], axis=1)
    print(train.shape[0])
    extracted_features = extract_features(train, column_id="series_id", column_sort="measurement_number")
    impute(extracted_features)
    print(extracted_features.shape)
    print(y.shape)
    extracted_features.to_csv(path_or_buf='C:/Users/bhunt\OneDrive/Documents/KaggleData/Robot Career Competition/career-con-2019/X_train_extraction3.csv')
    features_filtered = select_features(extracted_features, y)
    features_filtered_direct = extract_relevant_features(train, y, column_id='series_id', column_sort='measurement_number')

    features_filtered_direct.to_csv(path_or_buf='C:/Users/bhunt\OneDrive/Documents/KaggleData/Robot Career Competition/career-con-2019/X_train_newfeat3.csv')
