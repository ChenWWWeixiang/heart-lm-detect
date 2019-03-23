import SimpleITK as sitk
import numpy as np
import glob
import os
import xlrd
import threading


#atlas_np = np.tile(atlas_np[np.newaxis,np.newaxis,:,:,:],[args.batch_size,1,1,1,1]).astype(np.float32)

#atlas_torch = torch.from_numpy(atlas_np).type(torch.float32).to(device)
#atlas_torch_half = F.interpolate(atlas_torch,scale_factor=0.5).to(device)
def get_boxes(box_dir):
    ExcelFile = xlrd.open_workbook(box_dir)
    sheet = ExcelFile.sheet_by_name('Sheet1')
    name_cols = sheet.col_values(0, 3)
    T2_box=[]
    T1_box = []
    a=list(range(1, 7))
    a.append(14)
    for i in a:
        t = sheet.col_values(i,3)
        for idx,x in enumerate(t):
            if x=='':
                t[idx]=0

        T2_box.append(t)
    a=list(range(8, 14))
    a.append(14)
    for i in a:
        t = sheet.col_values(i,3)
        for idx,x in enumerate(t):
            if x=='':
                t[idx]=0
        T1_box.append(t)
    T2_box=np.array(T2_box)
    T1_box=np.array(T1_box)
    box_dict=dict()
    for idx,one in enumerate(name_cols):
        real_id=one.split('.')[0]
        if box_dict.get(real_id)==None:
            box_dict[real_id]=[]
        if np.sum(T2_box[:,idx])>0:
            record=dict()
            record['box']=T2_box[:-1,idx]
            record['name']=one+'/T2.1.nii'
            record['type']='T2'
            record['nian'] = T2_box[-1,idx]
            box_dict[real_id].append(record)
        if np.sum(T1_box[:,idx])>0:
            record = dict()
            record['box']=T1_box[:-1,idx]
            record['name'] = one+'/T1.1.nii'
            record['type'] = 'T1'
            record['nian'] = T1_box[-1,idx]
            box_dict[real_id].append(record)
    keys=list(box_dict.keys())
    for one in keys:
        persondata=box_dict[one]
        if len(persondata)<=1:
            box_dict.pop(one)
        else:
            T1_flag=False
            T2_flag=False
            for item in persondata:
                if item['type']=='T1':
                    T1_flag=True
                if item['type'] == 'T2':
                    T2_flag = True
            if not (T1_flag and T2_flag):
                box_dict.pop(one)
    return box_dict

#T1_files = glob.glob('/home/data2/pan_cancer/T1/*.mha')#moved--T1
data_path='/home/data2/pan_cancer/0321/'

#T2_files = glob.glob('/home/data2/pan_cancer/T2/*.mha')#fixed--T2
boxdata_path='/home/data2/pan_cancer/datalabel.xlsx'
box_dict=get_boxes(boxdata_path)
OUTPUT_DIR = '/home/data2/pan_cancer/0321output/'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
all_keys=list(box_dict.keys())
for key in all_keys:

    p_data=box_dict[key]
    T1_files=[]
    box1=[]
    box2=[]
    T2_files=[]
    for line in p_data:
        if line['type']=='T1' and os.path.exists(data_path+line['name']):
            T1_files.append(data_path+line['name'])
            box1.append(line['box'])
        if line['type']=='T2' and os.path.exists(data_path+line['name']):
            T2_files.append(data_path+line['name'])
            box2.append(line['box'])
    T1num=len(T1_files)
    T2num = len(T2_files)
    for i in range(T1num):
        for j in range(T2num):
            fixed_image = sitk.ReadImage(T1_files[i], sitk.sitkFloat32)  ##
            moving_image = sitk.ReadImage(T2_files[j], sitk.sitkFloat32)

            print(T1_files[i])
            initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                                  moving_image,
                                                                  sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)
            moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0,
                                             moving_image.GetPixelID())
            sitk.WriteImage(moving_resampled,OUTPUT_DIR+key+'T2.mha')
            sitk.WriteImage(fixed_image, OUTPUT_DIR + key + 'T1.mha')





