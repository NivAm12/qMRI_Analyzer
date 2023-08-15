import pandas as pd
import numpy as np
import nibabel as nib
from scipy import stats
import os
import glob
from re import search
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression


# -------------------- MAPS and Segmentations paths -------------------- #
from constants import R1, R2S, MT, TV, T2, DIFFUSION, \
    MAP_DIFFUSION, MAP_MT, MAP_TV, MAP_R1, MAP_T2, MAP_R2S, \
    SEG_DIFFUSION, SEG_T2, BASIC_SEG
import constants

# -------------------- Statistical Methods Names -------------------- #
from constants import ROBUST_SCALING, Z_SCORE

# -------------- Save Addresses ---------------#
SAVE_ADDRESS = "/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/saved_versions/corr_by_means/" \
               "subcortical_updated/with_R1/raw_data_of_subjects/"

# ------------- File Names -------------------#
FILE_NAME_PURE_RAW_DATA = "raw_data3"
FILE_NAME_Z_SCORE_AVG = "raw_data_z_score_avg"
FILE_NAME_Z_SCORE_ON_BRAIN = "raw_data_z_score_on_brain3"
FILE_NAME_DMEDIAN = "raw_data_robust_scaling3"


def HUJI_subjects_preprocess(analysisDir):
    """
    # Description: preprocesses HUJI subjects by the consensus in the lab
    :param analysisDir: A path to the dir from which the data will be taken
    :return: np array shape: (num_of_subjects after preprocess, 1)
    """

    subject_names = os.listdir(analysisDir)
    subject_names = [i for i in subject_names if (search('H\d\d_[A-Z][A-Z]', i) or search('H\d\d\d_[A-Z][A-Z]', i))]
    # currently this regex is good for the 'H010_AG' and 'H60_GG' expressions
    subject_names.sort()
    del subject_names[1:8]

    subject_names.remove('H010_AG')
    subject_names.remove('H011_GP')
    subject_names.remove('H014_ZW')
    subject_names.remove('H029_ON')
    subject_names.remove('H057_YP')
    subject_names.remove('H60_GG')
    subject_names.remove('H061_SE')
    subject_names = np.array(subject_names)

    subject_names = subject_names.reshape(-1, 1)
    return subject_names


def get_subject_paths(analysis_dir, subject_names):
    """
    # this func finds the paths to the MRI scan data of the subjects and creates a np array of them
    # as well as a np array of the names of the subjects which have a path, returning both arrays
    # analysisDir: the directory of the dataset used
    # niba_file_name (nii.gz end ing)
    # will create an array of the sub_path of each subject
    # ex1: analysisDir: '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'
    # niba_file_name: 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz'
    # subject_names: numpy array of names of the subjects: shape:(num of subjects, 1), vector, return value from
    # HUJI_subjects_preprocess
    # ex2: analysisDir: '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'
    # niba_file_name: 'first_all_none_firstseg.nii.gz'
    :param analysis_dir: The input dir from which all the path will be brought.
    :param subject_names: all the subject names
    :return: tuple of lists, first list is all the subject paths and the second list is the names of the subjects.
    """

    subject_paths = []
    names = []

    for sub in range(len(subject_names)):
        subject = subject_names[sub]
        os.chdir(analysis_dir)
        scanedate = os.path.join(analysis_dir, subject[0])  # adds the to the string the second string so we have a path
        os.chdir(scanedate)  # goes to current path

        readmepath = os.path.relpath('readme',
                                     scanedate)  # This method returns a string value which represents the relative file path to given path from the start directory.
        if os.path.isfile(readmepath):  # has a read me
            file1 = open(readmepath, 'r')
            A = file1.read().splitlines()[0]
            sub_path = [scanedate + '/' + A]
        else:
            sub_path = glob.glob(scanedate + '/*/')

        if not sub_path:
            continue

        subject_paths.append(sub_path)
        names.append(subject[0])

    return subject_paths, subject_names


class DataReader:
    """
    This Class of DataReader gets the basic data from the scanning for all subject,
    gets all the relevant subject with the relevant qmri params (with its mapping and segmentations), normalize all
    the data according to normalizer on the whole brain. if given - also calculates the derivative of the given params.
    Afterwards by using the save_in_pickle_raw_data, we save the data (which is only the relevant params, subjects and
    normalized) as df and pickle.
    NOTICE: Derivative funcs still isn't accurate enough.
    """

    def __init__(self, analysis_dir, rois, qmri_params, choose_normalize, derivative_params=None, bin_data=None):
        """
        Initialize a DataReader object
        :param analysis_dir: path to the MRI scanning the will enable to collect all the relevant data.
        :param rois: list on ints of the rois.
        :param qmri_params: dictionary where the keys are the params, values are map of the param and its segmentation.
        :param choose_normalize: normalizer of all the collected data - None, Z-Score, Robust-Scaling
        :param derivative_params: dictionary where the key is a param name, and values are all the other params that
               together we derive them, for example - {TV: [R1,R2s]}.
        :param bin_data: the range divided to bins for the algorithm for the derivative
        """
        subject_names_processed = HUJI_subjects_preprocess(analysis_dir)
        self.subject_paths, self.subject_names = get_subject_paths(analysis_dir, subject_names_processed)
        self.rois = np.array(rois)
        self.qmri_params = qmri_params
        self.choose_normalize = choose_normalize
        self.derivative_params = derivative_params  # All derivative params represented as {param: [list of param to derviate with]}

        self.bin_data = np.linspace(0.00, 0.4, 36) if bin_data is None else bin_data
        self.all_subjects_raw_data = []  # This is a list where each cell in the list represents a subject, in each
        # cell there is a dictionary of {param: {roi: values}}
        self.data_extracted = False  # Indicates if the data was extracted
        self.idx_used_all_subject = []

    def extract_data(self):
        """
        Extract the data from the paths.
        :return: None
        """
        name_idx = 0
        sub_names = []

        for sub_path in self.subject_paths:
            measures, seg_dict = self.measures_per_subject(sub_path)

            if measures == -1:
                name_idx += 1
                continue

            sub_names.append(self.subject_names[name_idx][
                                 0])  # save names of subjects that have all the data relevant for analysis
            name_idx += 1
            self.all_subjects_raw_data.append((self.add_all_info_of_param_per_subject(measures, seg_dict)))

        self.subject_names = sub_names

        if self.derivative_params:
            self.add_derivative_params_to_data()

        self.data_extracted = True

    def save_in_pickle_raw_data(self, save_address):
        """
        Save the extracted data as a pickle.
        :param save_address: The address where to save the pickle.
        :return: None
        """
        if self.data_extracted:
            all_subjects = pd.DataFrame(self.all_subjects_raw_data,
                                        index=self.subject_names)

            all_subjects.to_pickle(save_address)
        else:
            raise ("Data Not Extracted, Please Extract Data First! (DataReader object. extract_data())")

    def measures_per_subject(self, sub_path):
        """
        # Creates dictionary where key is the parameter and values are the measures of the parameter, t1,r2s,mt, tv
        and seg file
        :param sub_path: sub path of the given subject
        :return: tuple: first is dictionary where key is the parameter and values are the measures of the parameter,
                        second is segmentation dict
        """

        measures = {}
        seg_dict = {}

        for param_name in self.qmri_params.keys():
            param_file_to_use = None
            seg_file_to_use = None

            # check sub folders of subject
            for path in sub_path:
                param_file = os.path.join(path, self.qmri_params[param_name][0])

                if not os.path.isfile(param_file):
                    continue

                param_file_to_use = param_file
                break

            for path in sub_path:
                seg_file = os.path.join(path, self.qmri_params[param_name][1])

                if not os.path.isfile(seg_file):
                    continue

                seg_file_to_use = seg_file
                break

            if param_file_to_use is None or seg_file_to_use is None:
                return -1, -1

            # print(f'param: {param_name}')
            # os.system(f'freeview -v {param_file_to_use} {seg_file_to_use}:colormap=lut &')

            seg = nib.load(seg_file_to_use).get_fdata()
            seg_dict[param_name] = seg

            param_data = nib.load(param_file_to_use).get_fdata()

            # if param_name == 'r1':
            #     r1 = 1 / param_data  # todo: problem with zero division, change later
            #     param_data = np.nan_to_num(r1, posinf=0.0, neginf=0.0)

            measures[param_name] = param_data

        return measures, seg_dict

    def normalize_raw_data_by_z_score(self, measures, seg_dict, param_name) -> None:
        """
        Normalize raw data by z_score
        :param measures: all measures of the brain of all parameters for the subject (dictionary = parameter:values of
                         the parameter over all the brain)
        :param seg_dict: the compatible segmentation map for the parameter
        :param param_name: the parameter name.
        :return:
        """
        roi_mask = np.where((np.isin(seg_dict[param_name], self.rois)) & (measures[param_name] > 0))
        measures[param_name][roi_mask] = stats.zscore(measures[param_name][roi_mask], nan_policy='omit')

    def normalize_raw_data_by_robust_scaling(self, measures, seg_dict, param_name) -> None:
        """
        Normalize raw data by robust scaling
        :param measures: all measures of the brain of all parameters for the subject (dictionary = parameter:values of
                         the parameter over all the brain)
        :param seg_dict: the compatible segmentation map for the parameter
        :param param_name: the parameter name.
        :return:
        """
        roimask = np.where((np.isin(seg_dict[param_name], self.rois)) & (measures[param_name] > 0))
        # print("zeros calculated robust:", len(measures[measure_idx][roimask][measures[measure_idx][roimask] < 0]))
        scaler = RobustScaler()
        mea_df = pd.DataFrame(measures[param_name][roimask])
        df_robust = pd.DataFrame(scaler.fit_transform(mea_df), columns=mea_df.columns)
        param = df_robust.to_numpy()
        measures[param_name][roimask] = param.reshape(param.shape[0], )

    def _add_only_voxels_from_rois(self, measures, seg_dict, param_name, sub_measure, how_to_normalize):
        """
        adds voxels info to all rois - which means that for each roi takes all the voxels that belong to this ROI
        (in addition deletes all values which are 0 if not normalized, or other problematic
        values).
        :param measures: all measures of the brain of all parameters for the subject (dictionary = parameter:values of
                         the parameter over all the brain)
        :param seg_dict: segmentation dict where key is a parameter and values is the seg map.
        :param param_name: parameter name
        :param sub_measure: dictionary where key is roi number and value is array of all values in voxels in this roi.
        :param how_to_normalize: how to normalize the data - None/Z-Score/Robust-Scaling
        :return:
        """
        for roi in self.rois:
            # all indices in seg file with value roi, all coordinates with same label (associated with same area)
            roi_mask = np.where(seg_dict[param_name] == roi)

            mea_masked = measures[param_name][roi_mask]  # slice to only include values with the same label,
            # keeps array only of values in a specific roi
            # TODO fix the shapes
            if how_to_normalize is None:
                non_zeroes = np.where(mea_masked > 0)
            else:
                non_zeroes = np.where(mea_masked != np.inf)

            sub_measure[roi] = mea_masked[non_zeroes]

    def add_all_info_of_param_per_subject(self, measures, seg_dict):
        """
        Add all info of paramter per subject - normalize data and takes only the data from the relevant rois.
        :param measures: all measures of the brain of all parameters for the subject (dictionary = parameter:values of
                         the parameter over all the brain)
        :param seg_dict: segmentation dictionary
        :return: Dictionary- key is parameter name, values are the dictionary of roi:list of values in the voxels in
                 this roi.
        """
        subject_params = {}
        for param_name in measures.keys():  # loop over all parameters ex: t1, r2s, tv..
            sub_measure = {}
            if self.choose_normalize == Z_SCORE:
                self.normalize_raw_data_by_z_score(measures, seg_dict, param_name)
            if self.choose_normalize == ROBUST_SCALING:
                self.normalize_raw_data_by_robust_scaling(measures, seg_dict, param_name)

            # this will hold the all subject's measures for a specific measurement
            self._add_only_voxels_from_rois(measures, seg_dict, param_name, sub_measure, self.choose_normalize)
            subject_params[param_name] = sub_measure

        return subject_params

    def derive_param_with_another_param(self, params):
        """
        Derive the parameters with other parameters
        :param params: given params
        :return:
        """
        sorted_tup = np.array(sorted(params))
        buckets = np.split(sorted_tup, np.bincount(np.digitize(sorted_tup[:, 0], self.bin_data)).cumsum())
        del_idx = []

        for i in range(len(buckets)):
            if len(buckets[i]) < 0.04 * len(params):
                del_idx.append(i)

        avg_points_bucket = list(map(lambda x: np.array([np.mean(x[:, 0]), np.mean(x[:, 1])]), buckets))
        avg_points_bucket = np.delete(avg_points_bucket, del_idx, axis=0)

        # rows_idx_contain_nan = [np.any(i) for i in np.isnan(avg_points_bucket)]
        # avg_points_bucket = np.delete(avg_points_bucket, rows_idx_contain_nan, axis=0)
        X = np.array(avg_points_bucket)[:, 0].reshape(-1, 1)
        y = np.array(avg_points_bucket)[:, 1].reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        coeffs = reg.coef_.tolist() * len(params)
        all_X_values = np.array(params)[:, 0].reshape(-1, 1)
        return np.array(coeffs).reshape(1, -1)[0], reg.predict(all_X_values).reshape(1, -1)[0]
        # todo: 1) check sum for each bucket + make avergae and than to all averages do linear regression
        # todo: 2) add another column which contain all values per the function we discovered by the informartion

    def add_derivative_params_to_data(self):
        """
        add the derivative params to the data
        :return:
        """
        for param_to_derive in self.derivative_params.keys():
            for subject_index in range(len(self.all_subjects_raw_data)):
                for second_param_to_derive in self.derivative_params[param_to_derive]:
                    self.all_subjects_raw_data[subject_index][f'Slope-{param_to_derive}-{second_param_to_derive}'] = {}
                    self.all_subjects_raw_data[subject_index][
                        f'D{param_to_derive}-{second_param_to_derive}-values'] = {}

                    for roi in self.rois:
                        param1_data = self.all_subjects_raw_data[subject_index][param_to_derive][roi]
                        param2_data = self.all_subjects_raw_data[subject_index][second_param_to_derive][roi]

                        min_len = min(len(param1_data), len(param2_data))  # TODO: Make it better solution!!
                        params_as_x_y = np.array([param1_data[:min_len], param2_data[:min_len]]).T.tolist()

                        self.all_subjects_raw_data[subject_index][f'Slope-{param_to_derive}-{second_param_to_derive}'][
                            roi], \
                            self.all_subjects_raw_data[subject_index][
                                f'D{param_to_derive}-{second_param_to_derive}-values'][roi] = \
                            self.derive_param_with_another_param(params_as_x_y)


if __name__ == "__main__":
    # The input dir containing all data of the subjects after MRI screening
    analysis_dir = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'

    # Can be changed - list of all ROIs' numbers from the segmentation
    rois = list(constants.ROI_CORTEX.keys())

    # Can be changed - this is the save address for the output
    save_address = '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/saved_versions/corr_by_means/2023_analysis/ROI_RIGHT_CORTEX_4_params/'

    # Can be changed - using other params - make sure to add another parameter as a name, and tuple of the
    # full path to the map of the parameter and the full path to the compatible segmentation
    params = {
        R1: (MAP_R1, BASIC_SEG),
        # R2S: (MAP_R2S, BASIC_SEG),
        MT: (MAP_MT, BASIC_SEG),
        TV: (MAP_TV, BASIC_SEG),
        T2: (MAP_T2, SEG_T2)
    }

    # Can be changed - add more sort of normalizer and fit to it the compatible name to the file
    normalizer_file_name = {None: FILE_NAME_PURE_RAW_DATA, Z_SCORE: FILE_NAME_Z_SCORE_ON_BRAIN,
                            ROBUST_SCALING: FILE_NAME_DMEDIAN}

    # ---- Here You Can Change the sort of normalizer ---- #
    choose_normalizer = None

    # ---- Here you can change the derivative_dict
    # derivative_dict = {TV: [R1, R2S]}
    derivative_dict = None

    # ---- Here You Can Change
    range_for_tv_default = np.linspace(0.00, 0.4, 36)

    # ---- RUN the Reader
    reader = DataReader(analysis_dir, rois, params, choose_normalizer, derivative_dict, range_for_tv_default)
    reader.extract_data()
    print(f'NUmber of subjects: {len(reader.all_subjects_raw_data)}')

    if not os.path.exists(save_address):
        os.mkdir(save_address)

    reader.save_in_pickle_raw_data(save_address + normalizer_file_name[choose_normalizer])
