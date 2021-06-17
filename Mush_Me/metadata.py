import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from utils import log_progress
from struct import unpack


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
        self.marker_mapping = {
            0xffd8: "Start of Image",
            0xffe0: "Application Default Header",
            0xffdb: "Quantization Table",
            0xffc0: "Start of Frame",
            0xffc4: "Define Huffman Table",
            0xffda: "Start of Scan",
            0xffd9: "End of Image"
        }

    def decode(self):
        data = self.img_data
        while (True):
            marker, = unpack(">H", data[0:2])
            # print(self.marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2 + lenchunk:]
            if len(data) == 0:
                break

def is_JPEG(image_path, data_path):
    complete_path = os.path.join(data_path, image_path)
    image = JPEG(complete_path)
    try:
        image.decode()
        return 1
    except:
        return 0


def move_picture_to_species(species, path, origin_path, path_dir):

    species = species.replace(' ', '_').lower()

    if not os.path.isdir(os.path.join(path_dir, species)):

        os.mkdir(os.path.join(path_dir, species))

    source = os.path.join(origin_path, path)

    target = os.path.join(path_dir, species, path)

    shutil.copy2(source, target)


class Data:
    def __init__(self, _set='mini_train'):

        ### _set should be the set you'd like to have
        ### You can pick between 4 datasets = 'train', 'test', 'mini_train', 'mini_test', 'all'
        self.local_path_to_data = os.path.join('/'.join(os.getcwd().split('/')[:6]),'Mush_Me','data')

        ### As some methods only work with the mini-train and mini-test sets, if a bigger dataset is required
        ### the user is required to confirm whether he'd like to proceed.

        if _set in ['train', 'test']:
            answer = input('You have selected a big dataset, are you sure you\'d like to proceed? \
                            (This might present some limitations further) [y/n]'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                )
            if answer == 'y':
                self._set = _set
            else:
                self._set = input('Please enter another set [mini_train, mini_test]')
        else:
            self._set = _set

        ### creates a path to raw metadata

        self.path_to_raw_metadata = os.path.join(self.local_path_to_data, 'metadata','01_initial_metadata',
                                             ('DanishFungi2020_' + self._set + '_metadata.csv'))

        ### creates a path to the pictures folder

        self.path_to_images = os.path.join(self.local_path_to_data, 'raw_data','pictures')



    def get_metadata(self):
        data = pd.read_csv(self.path_to_raw_metadata)
        self.len_initial = len(data)
        return data

    def get_clean_metadata(self, columns=False, species_directory=False):

        ''' This method returns a cleaned data set. By default this method
            returns the complete df. With the argument columns specified to
            True, you will be asked for the columns you'd like to keep.
        '''


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

        ### Select specific columns according to user needs

        if columns:
            print(data_filtered.columns)
            num_columns = int(input('How many columns would like to keep? '))
            cols = []
            for _ in range(num_columns):
                col = input('Give a column name: ')

                ### image_path is automatically included since we use it all the time

                if col == 'image_path':
                    print('This column is included by default, no need to input it!')
                    continue
                cols.append(col)

            cols.append('image_path')

            data_filtered = data_filtered[cols]

        ### To only keep the species and image_path columns for when we create
        ### the directories based on the species

        if species_directory:
            data_filtered = data_filtered[['species', 'image_path']]

        return data_filtered

    def create_csv(self, new_directory = False):
        ''' Creates a csv with the clean metadata
            By default, this will create a csv in already existing
            directory. If you wish to create a new set of CSVs please make
            sure to specify: new_directory = True'''

        self.path_clean_metadata = os.path.join(self.local_path_to_data,
                                                'metadata',
                                                '02_cleaned_metadata')

        ### Checks the number of files in the 02_cleaned_metadata directory

        number_of_files = len(os.listdir(self.path_clean_metadata))

        ### In case of a new directory

        if new_directory:

            ### create a new version_name for the new directory

            version = 'v'+str(number_of_files+1)

            ### create a path for the new directory

            path_new_dir = os.path.join(self.path_clean_metadata,version)

            ### create the directory

            os.mkdir(path_new_dir)

            ### generate a csv and store it in the new directory

            pd.to_csv(path_new_dir+self._set+'_'+version)

        else:
            ### prints the existing directory names to allow the user to pick
            ### in which directory he will create his CSV

            print(os.listdir(self.path_clean_metadata))

            ### asks the user which directory he'd like to use

            version = input('In which existing directory would you like to store this CSV? ')

            ### creates the path to the chosen directory

            path_dir = os.path.join(self.path_clean_metadata, version)

            ### creates a file name for the CSV

            file_name = self._set + '_' + version

            ### if the file already exists asks the user to confirm

            if file_name in os.listdir(path_dir):

                answer = input('This file already exists, you will erase it, would you like to proceed[y/n]? ')

                ### in case of no, aborts

                if answer == 'n':

                    return "No file has been created"

            ### generates the csv in the directory of this choice

            pd.to_csv(path_dir + self._set + '_' + version)

        return 'CSV created successfully'

    def global_image_directory(self, create_new_directory=True):

        ''' This method creates a directory which stores the images
            selected from the cleaned metadata. By default this creates
            a new directory '''

        ### creates a cleaned metadata df with the columns species and image_path

        data = self.get_clean_metadata(species_directory=True)

        image_paths = list(data['image_path'])

        print('Be careful, you should only run this function once!')

        if self._set in ['train','test']:
            print('Do not run this method with a full dataset')
            return None

        number_of_files = len(os.listdir(self.path_to_images))

        if create_new_directory:

            version = f'0{str(number_of_files)}_Dataset'

            path_new_dir = os.path.join(self.path_to_images,version,self._set)

            os.mkdir(path_new_dir)

            for path in log_progress(image_paths, every=1):

                origin = os.path.join(self.path_to_images, '00_initial_set',self._set, path)

                target = os.path.join(path_new_dir, path)

                shutil.copy2(origin,target)

            return 'Transfer successfully executed!'

        print(os.listdir(self.path_to_images))

        directory = input('Please select in which directory you\'d like to insert this new file:')

        path_dir = os.path.join(self.path_to_images,directory,self._set)

        os.mkdir(path_dir)

        for path in log_progress(image_paths, every=1):

            origin = os.path.join(self.path_to_images, '00_initial_set', path)

            target = os.path.join(path_dir, path)

            shutil.copy2(origin,target)

        return 'Transfer successfully executed'

    def directory_by_species(self, create_new_directory=True):

        if self._set in ['train','test']:
            print('Do not run this method with a full dataset')
            return None

        data = self.get_clean_metadata(species_directory=True)

        path = os.path.join(self.path_to_images, '00_initial', self._set)

        data['is_JPEG'] = data['image_path'].map(lambda x: is_JPEG(x, path))

        data = data[data['is_JPEG']==1]

        number_of_files = len(os.listdir(self.path_to_images))

        if create_new_directory:

            version = f'0{str(number_of_files-1)}_By_Species_Dataset'

            path_new_dir = os.path.join(self.path_to_images, version)

            os.mkdir(path_new_dir)

            path_new_subdir = os.path.join(path_new_dir, self._set)

            os.mkdir(path_new_subdir)

            data.apply(lambda x: move_picture_to_species(x[0],x[1],path, path_new_dir), axis=1)

            return 'New species directories created with success'

        print(os.listdir(self.path_to_images))

        directory = input(
            'Please select in which directory you\'d like to insert this new file:'
        )

        path_new_dir = os.path.join(self.path_to_images, directory, self._set)

        os.mkdir(path_new_dir)

        data.apply(
            lambda x: move_picture_to_species(x[0], x[1], path, path_new_dir),axis=1)

        return 'Transfer successfully executed'
