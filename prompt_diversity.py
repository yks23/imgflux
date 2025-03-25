import shutil
import os
valid_file_path='./data/valid'
final_file_pth='./data/valid_new'
base_dir="/data115/video-diff/workspace/HOI-DiffusionAsShader"
output_path="./data/valid-data"
os.makedirs(final_file_pth,exist_ok=True)
os.makedirs(output_path,exist_ok=True)
files=os.listdir(valid_file_path)
for file in files:
    pathfilelist=os.path.join(final_file_pth,file)
    with open(pathfilelist,'w') as F:
        pathfile=os.path.join(valid_file_path,file)
        with open(pathfile,'r') as f:
            videolist=[s.strip() for s in f.readlines()]
        for i,pth in enumerate(videolist):
            if os.path.isfile(os.path.join(base_dir,pth)):
                middle_name="0"*(3-len(str(i)))+str(i)
                new_file_name=pth.split('/')[-1]
                os.makedirs(os.path.join(output_path,middle_name),exist_ok=True)
                shutil.copy(os.path.join(base_dir,pth),os.path.join(output_path,middle_name,new_file_name))
                F.write(os.path.join(middle_name,new_file_name)+'\n')
            else:
                middle_name="0"*(3-len(str(i)))+str(i)
                os.makedirs(os.path.join(output_path,middle_name),exist_ok=True)
                shutil.copytree(os.path.join(base_dir,pth),os.path.join(output_path,middle_name,'label'), dirs_exist_ok=True)
                F.write(os.path.join(middle_name)+'\n')
                
            
        