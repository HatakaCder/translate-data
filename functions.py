from deep_translator import GoogleTranslator
from tqdm import tqdm
import translators as ts
import numpy as np

drop_columns = ['answer',
 'ccid',
 'context_id',
 'course_id',
 'course_order',
 'create_time',
 'end',
 'enroll_time',
 'exercise_id',
 'gender',
 'graph_predict',
 'ground_truth',
 'id',
 'language',
 'location',
 'name_en',
 'problem_id',
 'resource_id',
 'score',
 'sign',
 'start',
 'text_predict',
 'type',
 'user_id',
 'year_of_birth']
drop_keys = ['resource_id', 'chapter']

def save_file(df, name):
    df.to_json(f'{name}_translated.json', orient='records', force_ascii=False)
    return df
def translate_text(text, type_trans):
    if text != None:
        if type_trans == 'deep_translator':
            text = GoogleTranslator(source='auto', target='vi').translate(text=text)
        elif type_trans == 'translator':
            text = ts.translate_text(text, to_language='vi', if_ignore_empty_query=True)
        else: pass
    return text

def translate_list(lst, type_trans):
    if len(lst)!=0:
        valid_data = [item for item in lst if item is not None]
        random_sample = valid_data[0]

        type_rnd = type(random_sample).__name__
        if type_rnd == 'str':
            lst = [translate_text(item, type_trans) for item in lst]
        elif type_rnd == 'dict':
            lst = [translate_dict(item, type_trans) for item in lst]
        else: pass
    return lst

def translate_dict(dct, type_trans):
    lst_keys = list(dct.keys())

    if len(lst_keys) != 0:
        lst_keys = np.setdiff1d(lst_keys, drop_keys)
        for item in lst_keys:
            type_keys_dict = type(dct[item]).__name__

            if type_keys_dict == 'list':
                dct[item] = translate_list(dct[item], type_trans)
            elif type_keys_dict == 'str':
                dct[item] = translate_text(dct[item], type_trans)
    return dct

def translate_data(df, name, type_trans, checkpoint=None):
    if checkpoint is None: checkpoint = len(df)
    old_idx = 0

    col_df = np.setdiff1d(np.array(df.columns), drop_columns)
    col_types = df[col_df].apply(lambda col: type(col[0])).to_numpy()
    col_df_dict = dict(zip(col_df, [col_type.__name__ for col_type in col_types]))

    for idx in tqdm (range (len(df)), desc="Translating..."):
        for col_name, col_type in col_df_dict.items():
            if col_type=='str':
                df.at[idx, col_name]=translate_text(df.at[idx, col_name], type_trans)
            elif col_type=='list':
                df.at[idx, col_name]=translate_list(df.at[idx, col_name], type_trans)
            elif col_type=='dict':
                df.at[idx, col_name]=translate_dict(df.at[idx, col_name], type_trans)
            else: continue

        if (idx+1) % checkpoint == 0:
            df.iloc[old_idx:idx] = save_file(df, name).iloc[old_idx:idx]
            old_idx = idx

    df.iloc[old_idx:] = save_file(df, name).iloc[old_idx:]
    return df