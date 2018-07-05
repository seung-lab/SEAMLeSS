import sys
import ast
import glob
import os
import subprocess

def strip_name(s):
    if s[-1] == '/':
        s = s[:-1]
    if '/' in s:
        s = s[s.rindex('/')+1:]
    if s.endswith('.pt'):
        s = s[:-3]
    return s

def parse(contents):
    dup_idx = contents.rindex('Namespace')
    while dup_idx > 0:
        contents = contents[:dup_idx]
        dup_idx = contents.rindex('Namespace')
    contents = "{'"+ contents[10:-1] + "'}"
    contents = contents.replace("=", "':'")
    contents = contents.replace(", ", "', '")
    contents = contents.replace("''", "'")
    return ast.literal_eval(contents)

def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    return subprocess.check_output(['du','-sh', path]).split()[0].decode('utf-8')

#####

print('----------------------------------------------------------')
print('Select a network by its number, type a name, or type MORE,')
print('----------------------------------------------------------')
search_dir = "out/"
folders = filter(lambda x: not os.path.isfile(x), glob.glob(search_dir + "*"))
folders.sort(key=lambda x: os.path.getmtime(x))
folders = list(reversed([f[f.index('/')+1:] for f in folders]))

display_length = 10
for start_idx in range(0, len(folders), display_length):
    for idx, f in enumerate(folders[start_idx:start_idx + display_length]):
        aux_info = '[{}]'.format(du('out/{}'.format(f)) if os.path.exists('pt/{}.pt'.format(f)) else 'NO PT ARCHIVE')
        print '{}) {} {}'.format(start_idx + idx + 1, f, aux_info)
    print('----------------------------------------------------------')
    selection = raw_input()
    if selection.isdigit():
        name = folders[int(selection)-1]
        break
    elif 'more'.startswith(selection.lower()):
        continue
    else:
        name = selection
        break

name = strip_name(name)    

ARCHIVE_KEY = 'state_archive'
OTHER_PARAMS = ['lambda1', 'lambda2', 'lambda3', 'lambda4', 'lambda5', 'num_targets', 'lr']

files = []
aux_info_printouts = []
while True:
    files.append(name)
    try:
        f = open('out/' + name + '/args.txt')
        contents = parse(f.read())
        f.close()
        aux_info = ', '.join(['{}: {}'.format(k, contents[k] if k in contents else '<NOT FOUND>') for k in OTHER_PARAMS])
        aux_info_printouts.append(aux_info)
        
        next_name = contents['state_archive']
        if next_name is not None:
            name = strip_name(next_name)
        else:
            break
    except Exception as e:
        aux_info_printouts.append('~ No args.txt found ~')
        break

print('\nHistory for network {}:\n'.format(files[0]))
for idx, (f, aux) in enumerate(reversed(zip(files, aux_info_printouts))):
    print('{}) {}{}:\n  {}\n'.format(idx+1, f, ' [ROOT ARCHIVE]' if idx == 0 else '', aux))
