import pandas as pd
from functools import reduce
import constants


def split_slash(x) -> str:
    """
    Split Name of subject by Slash
    :param x: string (full name of subject)
    :return: name of subject by number and abbreviations
    """
    return x.split("/")[0][1:]


def delete_apostrophes(x) -> str:
    """
    Deletes apostrophes in a word
    :param x: string
    :return: string without apostrophes
    """
    return x[1:-1]


class DataProcessor:
    """
    This class processes the data
    takes the pickle data which is
            |   r1  |   r2s    |    mt    |    tv  | diffusion | t2 |
    H20_AS  | dict{roi (int) : nd.array[values of voxels]}
    ..
    ->
    Converts it to a data frame:
    subject | ROI  |   r1  |   r2s    |    mt    |    tv  | diffusion | t2 |
    H20_AS  |  10 |  list[values of voxels]
    H20_AS  |  11 |  list[values of voxels]
    ...
    ...
    ...
    """
    def __init__(self, path_to_data, roi_dict=constants.SUB_CORTEX_DICT, wanted_rois=None):
        """
        Initialize a DataProcessor object
        :param path_to_data: path to the pickle's data
        :param roi_dict: ROI dictionary {roi number (int): roi name (str)}
        :param wanted_rois: ROI dictionary {roi number (int): roi name (str)} for wanted ROIs
        """
        self.roi_dict = roi_dict
        self.wanted_rois = wanted_rois
        self.df = self.get_raw_data_of_all_relevant_subjects(path_to_data)

    def get_data_proccessed(self) -> pd.DataFrame:
        """
        A getter to the data
        :return: pd.DataFrame
        """
        return self.df

    def get_raw_data_of_all_relevant_subjects(self, data_path) -> pd.DataFrame:
        """
        Get a Path to a pickle that save all info about the data, and returns it as df
        :param data_path: Path to the Pickle
        :return: DataFrame
        """
        data = pd.read_pickle(data_path)
        data.index.name = "subjects"
        data = self.create_data_frame_with_rois(data)
        return data

    def _edit_all_columns_of_parameters(self, data) -> pd.DataFrame:
        """
        edit the columns of the parameter - create a new column named ROI which contains the value of the dicitonary
        of each roi. If needed - also leaves only wanted ROIS by by given wanted_ROIs
        :param data: given data
        :return: the updated full data as df
        """
        df_to_concate = []
        for col_name in data.columns:
            # This part takes a column -> makes it a pd.Series and reset the indexes -> afterwards it takes the
            # data in each cell in the columns and create another columns named "ROI" which contain the key
            # value of the dictionary, and the values (list of voxels) leaves in the parameter columns
            # (see documentation of the class to understand better or just debug)
            df_to_concate += [data[col_name].apply(pd.Series).reset_index().melt(id_vars=["subjects"], var_name="ROI",
                                                                                 value_name=f"{col_name}").sort_values(
                                                                                 by=['subjects', "ROI"])]
        # Concatenate all df creates for each parameter
        full_data = reduce(lambda left, right: pd.merge(left, right, on=['subjects', 'ROI']), df_to_concate)

        # if only some ROIs are relevant - drop all the other rois
        if self.wanted_rois:
            full_data.drop(full_data[~full_data["ROI"].isin(list(self.wanted_rois.keys()))].index, inplace=True)

        return full_data

    def _add_columns(self, full_data):
        """
        Add columns to the data: subject, Age, Gender and ROI_name
        :param full_data: given df of full data
        :return: updated df with all new columns
        """
        names_col = ['subjects', 'Age', 'Gender']
        subject_info = pd.read_csv(constants.SUBJECTS_INFO_PATH, names=names_col)
        subject_info["subjects"] = subject_info["subjects"].apply(split_slash)
        subject_info["Gender"] = subject_info["Gender"].apply(delete_apostrophes)
        full_data = pd.merge(full_data, subject_info, on=['subjects'])
        full_data['ROI_name'] = full_data['ROI'].apply(lambda x: self.roi_dict[x])
        return full_data

    def create_data_frame_with_rois(self, data) -> pd.DataFrame:
        """
        Gets data (df of dictionaries) and convert it to df - which means take all dictionaris and seperate
        them by ROIs
        :param data: df
        :return: DataFrame where each there is a column for each ROI, and per parameter list of all values in all voxels
        in the specific ROI
        """
        full_data = self._edit_all_columns_of_parameters(data)
        full_data = self._add_columns(full_data)
        return full_data
        
    @staticmethod   
    def extract_outliers(data, param, chosen_rois_dict):
        outliers = {}
        threshold = 3

        for roi_value, roi_name in chosen_rois_dict.items():
            roi_to_check_data = data[data.ROI == roi_value]
            Q1 = roi_to_check_data[param].quantile(0.25)
            Q3 = roi_to_check_data[param].quantile(0.75)
            IQR = Q3 - Q1


            outliers_df = roi_to_check_data[(roi_to_check_data[param] < Q1 - threshold * IQR) | (roi_to_check_data[param] > Q3 + threshold * IQR)]

            if outliers_df.subjects.values.size > 0:
                outliers[roi_value] = outliers_df.subjects.tolist()

        return outliers
    
    @staticmethod
    def outliers_counter(data, params_to_work_with, chosen_rois_dict):
        outliers_counter = {}

        for param in params_to_work_with:
            subjects_outliers_counter = {}
            outliers = DataProcessor.extract_outliers(data, param, chosen_rois_dict)

            for roi, roi_outliers in outliers.items():
                    for outlier in roi_outliers:
                        if outlier in subjects_outliers_counter:
                            subjects_outliers_counter[outlier] += 1
                            outliers_counter[outlier] += 1
                        else:
                            subjects_outliers_counter[outlier]  = 1
                            if outlier not in outliers_counter:
                                outliers_counter[outlier] = 1

            subjects_outliers_counter = dict(sorted(subjects_outliers_counter.items(), key=lambda item: item[1], reverse=True))
            # print(f'{param} outliers: {outliers}')
            # print(f'{param} outliers counter: {subjects_outliers_counter}')
            # print(f'{param} outliers counter: {outliers_counter}')
            # print('----------------------------------------------------------')

        outliers_counter = dict(sorted(outliers_counter.items(), key=lambda item: item[1], reverse=True))
        print(f'outliers counter: {outliers_counter}')

        # data = data[~data.subjects.isin(['H047_DC', 'H054_AE', 'H037_YB', 'H036_EV'])]
        data.subjects.nunique()