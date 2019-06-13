import os
import sys
argv = sys.argv
import json

file_name = argv[1]
script_name = file_name.split('.')[0]
script_suf = file_name.split('.')[1]
if script_suf=='py':
    kernel_type = 'script'
elif script_suf=='ipynb':
    kernel_type = 'notebook'
else:
    raise ValueError('Only python script or jupyter notebook are allowed.')

metadata = {
    'id': f'yujirokita/{script_name}',
    'title': script_name,
    'code_file': file_name,
    'language': 'python',
    'kernel_type': kernel_type,
    'is_private': 'true',
    'enable_gpu': 'false',
    'enable_internet': 'false',
    'competition_sources': ['instant-gratification'],
    'kernel_sources': []
}
with open('kernel-metadata.json', 'w') as fp:
    json.dump(metadata, fp)

command = 'kaggle kernels push'
#os.system(command)