import os, sys
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants
from Data_Reader import HUJI_subjects_preprocess, get_subject_paths
import nibabel as nib


def linear_register_maps(analysis_dir, ref_map_name, in_map_name, output_folder, output_file_name):
    subject_names_processed = HUJI_subjects_preprocess(analysis_dir)
    subject_paths, subject_names = get_subject_paths(analysis_dir, subject_names_processed)

    for sub_path in subject_paths:
        ref_file = os.path.join(sub_path[0], ref_map_name)
        in_file = os.path.join(sub_path[0], in_map_name)
        if not os.path.isfile(ref_file) or not os.path.isfile(in_file):
            continue

        output_file = os.path.join(sub_path[0], output_folder, output_file_name) + '.nii.gz'

        out_mat = os.path.join(sub_path[0], output_folder, f'{output_file_name}.mat')

        calcCmd = f'flirt -in {in_file} -ref {ref_file} -out {output_file} -omat {out_mat} -dof 6'
        subprocess.run(calcCmd, shell=True)

        applyCmd = f'flirt -in {in_file} -ref {ref_file} -out {output_file} -init {out_mat} -applyxfm'
        subprocess.run(applyCmd, shell=True)


if __name__ == "__main__":
    linear_register_maps(analysis_dir=constants.ANALYSIS_DIR, ref_map_name=constants.BASIC_SEG,
                         in_map_name=constants.MAP_T2, output_folder='T2_EMC',
                         output_file_name='T2_trans_map')

