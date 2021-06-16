import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from Mush_Me.utils import log_progress

class Data:
    def __init__(self, _set='mini_train'):

        ### _set should be the set you'd like to have
        ### You can pick between 4 datasets = 'train', 'test', 'mini_train', 'mini_test', 'all'
        self.local_path_to_data = os.path.join('/'.join(os.getcwd().split('/')[:6]),'Mush_Me','data')
        if _set in ['train', 'test']:
            answer = input('You have selected a big dataset, are you sure you\'d like to proceed? \
                            (This might present some limitations further) [y/n]')
            if answer == 'y':
                self._set = _set
            else:
              self._set = input('Please enter another set [mini_train, mini_test]')
        self.path = os.path.join(self.local_path_to_data,'raw_data', self._set)
        self.path_to_metadata = os.path.join(self.local_path_to_data, 'metadata','01_initial_metadata',
                                             ('DanishFungi2020_' + _set + '_metadata.csv'))

    def get_metadata(self):
        data = pd.read_csv(self.path_to_metadata)
        self.len_initial = len(data)
        return data


    def get_clean_metadata(self, columns=False):

        ### Filtering the data based on the kingdom Fungi, CoorUncert small than 1000m and collected in 2000 or late

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

        data_filtered.drop(columns=['level_0', 'index'], inplace=True)

        ### Select specific columns

        if columns:
            print(data_filtered.columns)
            num_columns = int(input('How many columns would like to keep? '))
            cols = []
            for _ in range(num_columns):
                col = input('Give a column name: ')
                if col == 'image_path':
                    print('This column is included by default, no need to input it!')
                    continue
                cols.append(col)

            cols.append('image_path')

            data_filtered = data_filtered[cols]

        ### Return data_filtered
        self.clean_metadata = data_filtered

        return self.clean_metadata

    def create_csv(self, new_directory = False):
        self.path_clean_metadata = os.path.join(self.local_path_to_data,
                                                'metadata',
                                                '02_cleaned_metadata')

        number_of_files = len(os.listdir(self.path_clean_metadata))

        if new_directory:

            version = 'v'+str(number_of_files+1)

            path_new_dir = os.path.join(self.path_clean_metadata,version)

            os.mkdir(path_new_dir)

            pd.to_csv(path_new_dir+self._set+'_'+version)

        else:

            version = 'v'+str(number_of_files)

            path_dir = os.path.join(self.path_clean_metadata, version)

            pd.to_csv(path_dir + self._set + '_' + version)

        return 'CSV created successfully'

    def select_images(self, create_new_directory=True):

        data = self.get_clean_metadata()

        image_paths = list(data['image_path'])

        print('Be careful, you should only run this function once!')

        if self._set in ['train','test']:
            print('Do not run this method with a full dataset')
            return None

        path_to_images = os.path.join(self.local_path_to_data, 'raw_data','pictures')

        number_of_files = len(os.listdir(path_to_images))

        if create_new_directory:

            version = f'0{str(number_of_files)}_Dataset'

            path_new_dir = os.path.join(path_to_images,version,self._set)

            os.make_dir(path_new_dir)

            for path in log_progress(image_paths, every=1):

                origin = os.path.join(path_to_images, '00_initial_set',self._set, path)

                target = os.path.join(path_new_dir, path)

                shutil.copy2(origin,target)

            return 'Transfer successfully executed!'

        print(os.listdir(path_to_images))

        directory = input('Please select in which directory you\'d like to insert this new file:')

        path_dir = os.path.join(path_to_images,directory,self._set)

        os.mkdir(path_dir)

        for path in log_progress(image_paths, every=1):

            origin = os.path.join(path_to_images, '00_initial_set', path)

            target = os.path.join(path_dir, path)

            shutil.copy2(origin,target)

        return 'Transfer successfully executed'
