import pandas as pd
from functions import translate_data

# replace example.json to your file for translating
# replace the name of the folder where the file is stored in
name = 'example'
folder = ''
folder = folder + '/' if folder != '' else '' 
df = pd.read_json(f'{folder}{name}.json', lines=True)

# type_trans:
# using translator for fast translate but not stable
# using deep_translator for stable but slow

# checkpoint:
# if it translates checkpoint data, the file will be saved as name_translated.json in the same directory

# startpos:
# where to start translating

# save_file:
print(translate_data(df, name, type_trans='translator', checkpoint=100))