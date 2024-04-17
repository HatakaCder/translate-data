from deep_translator import GoogleTranslator
from tqdm import tqdm
import translators as ts
import numpy as np
import os
import time

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

class Translate:
    def __init__(self, df, name, type_trans, startpos=0, checkpoint=None):
        self.df = df
        self.name = name
        self.type_trans = type_trans
        self.startpos = startpos
        if checkpoint == None:
            self.checkpoint = len(df)
        else:
            self.checkpoint = checkpoint

    def save_file(self):
        if os.path.isdir('translated') is False:
            os.mkdir('translated')
        self.df.to_json(f'translated/{self.name}_translated.json', orient='records', force_ascii=False)
        return self.df

    def translate_text(self, text):
        if text != None:
            if self.type_trans == 'deep_translator':
                text = GoogleTranslator(source='auto', target='vi').translate(text=text)
            elif self.type_trans == 'translator':
                text = ts.translate_text(text, to_language='vi', if_ignore_empty_query=True)
            else:
                pass
        return text

    def translate_list(self, lst):
        if len(lst) != 0:
            valid_data = [item for item in lst if item is not None]
            random_sample = valid_data[0]

            type_rnd = type(random_sample).__name__
            if type_rnd == 'str':
                lst = [self.translate_text(item) for item in lst]
            elif type_rnd == 'dict':
                lst = [self.translate_dict(item) for item in lst]
            else:
                pass
        return lst

    def translate_dict(self, dct):
        lst_keys = list(dct.keys())

        if len(lst_keys) != 0:
            lst_keys = np.setdiff1d(lst_keys, drop_keys)
            for item in lst_keys:
                type_keys_dict = type(dct[item]).__name__

                if type_keys_dict == 'list':
                    dct[item] = self.translate_list(dct[item])
                elif type_keys_dict == 'str':
                    dct[item] = self.translate_text(dct[item])
        return dct

    def translate_a(self, col_df_dict):
        print("Start pos:", self.startpos)
        old_sp = self.startpos
        old_idx = self.startpos
        for idx in tqdm(range(self.startpos, len(self.df)), desc="Translating..."):
            for col_name, col_type in col_df_dict.items():
                if col_type == 'str':
                    self.df.at[idx, col_name] = self.translate_text(self.df.at[idx, col_name])
                elif col_type == 'list':
                    self.df.at[idx, col_name] = self.translate_list(self.df.at[idx, col_name])
                elif col_type == 'dict':
                    self.df.at[idx, col_name] = self.translate_dict(self.df.at[idx, col_name])
                else:
                    continue

            if (idx - old_sp + 1) % self.checkpoint == 0:
                self.df.iloc[old_idx:idx] = self.save_file().iloc[old_idx:idx]
                old_idx = idx
                self.startpos = idx + 1
        self.df.iloc[old_idx:] = self.save_file().iloc[old_idx:]
        self.startpos = len(self.df)

    def translate_data(self):
        if os.path.exists(f'translated/{self.name}_translated.json'):
            ans = input(f"Do you want to overwrite the exist file: {self.name}_translated.json? (Y/N)")
            if ans.lower() != 'y': return self.df

        if self.checkpoint is None: checkpoint = len(self.df)

        col_df = np.setdiff1d(np.array(self.df.columns), drop_columns)
        col_types = self.df[col_df].apply(lambda col: type(col[0])).to_numpy()
        col_df_dict = dict(zip(col_df, [col_type.__name__ for col_type in col_types]))

        while self.startpos < len(self.df):
            try:
                self.translate_a(col_df_dict)
            except Exception as e:
                print("Loi bien dich:", e)
                time.sleep(3)

        return self.df