from deep_translator import MicrosoftTranslator
from tqdm import tqdm
import numpy as np
import os

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
    if os.path.isdir('translated') is False:
        os.mkdir('translated')
    df.to_json(f'translated/{name}_translated.json', orient='records', force_ascii=False)
    return df


def translate_text_new(texts, type_trans):
    if len(texts) != 0:
        if type_trans == 'dt-microsoft':
            texts = MicrosoftTranslator(api_key='823b2fcb911f41b0960aafaa0cf3c1aa', region='southeastasia',
                                        source='zh-hans', target='vi').translate_batch(list(texts))
        elif type_trans == 'dt-mymemory':
            pass
    return texts


def translate_list_new(lst, type_trans):
    if len(lst) != 0:
        valid_data = [item for item in lst if item is not None]
        random_sample = valid_data[0]

        type_rnd = type(random_sample).__name__
        if type_rnd == 'list':
            lst = [translate_text_new(item, type_trans) for item in lst]
        elif type_rnd == 'dict':
            lst = [translate_dict_new(item, type_trans) for item in lst]
        else:
            pass
    return lst


def translate_dict_new(dct, type_trans):
    lst_keys = list(dct.keys())

    if len(lst_keys) != 0:
        lst_keys = np.setdiff1d(lst_keys, drop_keys)
        for item in lst_keys:
            type_keys_dict = type(dct[item]).__name__

            if type_keys_dict == 'list':
                dct[item] = translate_list_new(dct[item], type_trans)
            elif type_keys_dict == 'str':
                dct[item] = translate_text_new(dct[item], type_trans)
    return dct

def translate_data(df, name, type_trans, checkpoint=None, startpos=0, block=10):
    if os.path.exists(f'{name}_translated.json'):
        ans = input(f"Do you want to overwrite the exist file: {name}_translated.json? (Y/N)")
        if ans.lower() != 'y': return df

    if checkpoint is None: checkpoint = len(df)
    old_idx = startpos

    col_df = np.setdiff1d(np.array(df.columns), drop_columns)
    col_types = df[col_df].apply(lambda col: type(col[0])).to_numpy()
    col_df_dict = dict(zip(col_df, [col_type.__name__ for col_type in col_types]))

    print("Start pos:", startpos)
    for idx in tqdm(range(startpos, len(df), block), desc="Translating..."):
        for col_name, col_type in col_df_dict.items():
            if col_type == 'str':
                df[col_name].iloc[idx:idx + block] = translate_text_new(df[col_name].iloc[idx:idx + block], type_trans)
            elif col_type == 'list':
                print("Hi")
                df[col_name].iloc[idx:idx + block] = translate_list_new(df[col_name].iloc[idx:idx + block], type_trans)
            elif col_type == 'dict':
                df[col_name].iloc[idx:idx + block] = translate_dict_new(df[col_name].iloc[idx:idx + block], type_trans)
            else:
                continue

        if (idx - startpos + 1) % checkpoint == 0:
            df.iloc[old_idx:idx + block] = save_file(df, name).iloc[old_idx:idx + block]
            old_idx = idx

    df.iloc[old_idx:] = save_file(df, name).iloc[old_idx:]
    return df