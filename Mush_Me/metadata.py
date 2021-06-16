import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Data:
    def __init__(self, _set='mini_train'):

        ### _set should be the set you'd like to have
        ### You can pick between 4 datasets = 'train', 'test', 'mini_train', 'mini_test'
        local_path_to_mush = '/'.join(os.getcwd().split('/')[:6])
        self._set = _set
        self.path = os.path.join(local_path_to_mush, 'Mush_Me', 'data',
                                 'raw_data', self._set)
        self.path_to_metadata = os.path.join(local_path_to_mush, 'Mush_Me', 'data', 'metadata',
                                             ('DanishFungi2020_' + _set + '_metadata.csv'))

    def get_metadata(self):
        data = pd.read_csv(self.path_to_metadata)
        self.len_initial = len(data)
        return data

    def get_clean_metadata(self, columns=False):

        ### Filtering the data based on the kingdom Fungi, CoorUncert small than 1000m and collected in 2000 or later
        data = self.get_metadata()

        data_filtered = data[data['kingdom']=='Fungi']\
                            [data['CoorUncert']<1000]\
                            [data["year"] > 1999].reset_index()

        ### Dropping useless columns

        data_filtered.drop(columns=[
            'rightsHolder', 'image_url', 'identifiedBy',
            'infraspecificEpithet', 'level0Gid', 'level0Name', 'level1Gid',
            'level1Name', 'level2Gid', 'level2Name', 'gbifID', 'kingdom',
            'taxonRank', 'locality'],inplace=True)

        ### Rename the column eventDate to date

        data_filtered = data_filtered.rename(columns={'eventDate': 'date'})

        ### Turning the date object into a datetime object

        data_filtered['date'] = pd.to_datetime(data_filtered['date'])

        ### Drop the remaining null values

        data_filtered.dropna(inplace=True)

        ### Correcting the image names to keep only the image names

        data_filtered['image_path'] = data_filtered['image_path'].map(lambda x: x.replace('/Datasets/DF20/', ''))

        ### Reset index

        data_filtered.reset_index(inplace=True)

        ### Select specific columns

        if columns:
            print(data_filtered.columns)
            num_columns = int(input('How many columns would like to keep? '))
            cols = []
            for _ in range(num_columns):
                  col = input('Give a column name: ')
                  cols.append(col)

            data_filtered = data_filtered[cols]

        ### Return data_filtered
        self.clean_metadata = data_filtered

        return self.clean_metadata

