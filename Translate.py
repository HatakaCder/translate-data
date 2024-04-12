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

# Nếu đang chạy mà lỗi bởi api và muốn chạy tiếp thì chỉnh tên file và thư mục theo file đã translated theo checkpoint và đặt startpos theo checkpoint đó
# Ví dụ nếu file a lỗi ở 22 và checkpoint = 10 thì cần chỉnh tên file a_translated và folder = 'translated' và set startpos = 20

# save_file:
print(translate_data(df, name, type_trans='translator', checkpoint=100, startpos=5))