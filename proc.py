import os
import sys
import shutil

def solve(path):
    
    os.makedirs(str(os.path.join(path, 'HR')), exist_ok = True)
    os.makedirs(str(os.path.join(path, 'LR')), exist_ok = True)

    for scale in [2, 3, 4]:
        if path.find('Urban100') != -1 and scale == 3:
            continue
        os.makedirs(str(os.path.join(path, 'HR', f'X{scale}')), exist_ok = True)
        os.makedirs(str(os.path.join(path, 'LR', f'X{scale}')), exist_ok = True)
        full_path0 = os.path.join(path, f'image_SRF_{scale}')
        
        for filename in os.listdir(full_path0):
            if filename.endswith('HR.png'):
                full_path = os.path.join(path, 'HR', f'X{scale}')
                f0 = os.path.join(full_path0, filename)
                f = os.path.join(full_path, filename.replace('_HR.png', '.png'))
                #print(f' > f0 = {f0}, f = {f}')
                shutil.copyfile(str(f0), str(f))
            if filename.endswith('LR.png'):
                full_path = os.path.join(path, 'LR', f'X{scale}')
                f0 = os.path.join(full_path0, filename)
                f = os.path.join(full_path, filename.replace('_LR.png', f'x{scale}.png'))
                shutil.copyfile(str(f0), str(f))

solve(sys.argv[1])