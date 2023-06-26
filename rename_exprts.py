import os



init_dir = 'results/'
exprts = [4, 5, 6]

for exprt in exprts:
    e_dir = init_dir + f'exprt_{exprt}/'
    subdirs = os.listdir(e_dir)
    for subdir in subdirs:
        ee_dir = e_dir + f'{subdir}/'
        if os.path.isdir(ee_dir):
            translation_table = str.maketrans('', '', ''.join(["'", ":", "{", "}", ","]))
            subdir_new = subdir.translate(translation_table).replace(' ', '_')
            #print(subdir_new)
            os.rename(ee_dir, e_dir + f'{subdir_new}/')
