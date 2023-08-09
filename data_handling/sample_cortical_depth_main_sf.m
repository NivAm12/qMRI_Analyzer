function [output2mrQ]=sample_cortical_depth_main_sf(subjects,ss,samples)

% samples=[-0.4:0.2:1.2]; % the distances along the coritical profile
sub=subjects{ss};
analysisdir='/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human/';
orig=['/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/freesurfer_subjects/' sub(1:7) '/mri/T1.mgz'];
ref=fullfile(analysisdir,sub,'/mrQ_2022/OutPutFiles_1/T1w/mprage_forFreesurfer.nii.gz');

%% create volumes from surface
output=[]; output2mrQ=[];
cmddisp=['mrview -load ' ref ' ' ];
for ii=1:length(samples) % run over all the required distances

    output{ii}=['/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/freesurfer_subjects/' sub(1:7) '/mri/corticaldepth_' num2str(ii) '.mgz'];

    % generate volume for the corical surface
    cmd=['mri_surf2vol --mkmask --hemi lh --surf white --projfrac ' num2str(samples(ii)) '  --identity ' sub(1:7) ' --subject ' sub(1:7) ' --o ' output{ii} ' --template ' orig]  ;
    system(cmd)

    % convert volume to nii
    output2mrQ{ii}=['/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/freesurfer_subjects/' sub(1:7) '/mri/corticaldepth_' num2str(ii) '2mrQ.nii.gz'];

    mgzIn  = output{ii};
    refImg = ref;
    niiOut = output2mrQ{ii};
    orientation = 'RAS';
    fs_mgzSegToNifti(mgzIn, refImg, niiOut,orientation);

    cmddisp=[cmddisp ' -roi.load ' output2mrQ{ii} ' ' ];
end

system(cmddisp)

end