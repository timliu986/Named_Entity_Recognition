"""
負責對訓練資料進行預處理:
    1.統一標點符號，刪除空白及換行
    2.把關鍵字標記
"""
import pandas as pd
import pickle
import re
from transformers import BertTokenizer
from tqdm import tqdm
from opencc import OpenCC
import os

os.chdir('./version4')
import numpy as np

def preprocess(tokenizer):
    """
    input:
        tokenizer: 要用哪種tokenizer，目前都是用bert的
        max_len: 句子的最大長度， 若是為-1，則會用train data中的最長長度為max_len
        data_version: 訓練資料的版本
    output:
        將下方四項存為pickle檔
        tokenized_question: 斷詞後的問題
        labeled_key: 把關鍵字轉換成label("O":非關鍵字 "B":關鍵字開始 "I":關鍵字內容)
        max_len: 最長長度，最大為512，因為bert模型最大只能輸入512
    """

    cc = OpenCC('s2twp')
    #cc.convert('宏')  # =>巨集 ???

    ## 全形轉半形
    def strQ2B(uchar):
        """把字串全形轉半形"""
        if uchar == "巨集":
            return "宏"
        if uchar in ['，', '。', '、', '?', '！', '\.', '\?', '!']:
            return uchar
        u_code = ord(uchar)
        if u_code == 12288:  # 全形空格直接轉換
            u_code = 32
        elif 65281 <= u_code <= 65374:  # 全形字元（除空格）根據關係轉化
            u_code -= 65248
        return chr(u_code)

    def load_data(path, define_dict, min_len=50):
        X = []
        y = []
        sentence = []
        labels = []
        split_pattern = re.compile(r'[。？！\.\?,!]')
        print(path)
        with open(path, "r", encoding='utf-8') as file:
            for line in file.readlines():
                # 每行为一个字符和其tag，中间用tab或者空格隔开
                line = line.strip().split()
                if not line or len(line) < 2:
                    X.append(sentence.copy())
                    y.append(labels.copy())
                    sentence.clear()
                    labels.clear()
                    continue
                try:
                    word, tag = strQ2B(cc.convert(line[0])), define_dict[line[1]]
                    if word == '\ue13b':
                        print(line)
                except:
                    print('convert error!')
                    print("line : ", line)
                    print("word : ", cc.convert(line[0]))
                tag = tag if tag != 'o' else 'O'
                if split_pattern.match(word) and len(sentence) >= min_len:
                    sentence.append(word)
                    labels.append(tag)
                    X.append(sentence.copy())
                    y.append(labels.copy())
                    sentence.clear()
                    labels.clear()
                else:
                    sentence.append(word)
                    labels.append(tag)
            if len(sentence):
                X.append(sentence.copy())
                sentence.clear()
                y.append(labels.copy())
                labels.clear()
        print('success!')
        return X, y

    ''' QANER/train.txt '''
    labels_change_dict = {
        "B_disease": "B-DISEASE",
        "I_disease": "I-DISEASE",
        "B_body": "B-BODY",
        "I_body": "I-BODY",
        "B_treatment": "B-TREATMENT",
        "I_treatment": "I-TREATMENT",
        "B_test": "B-CHECK",
        "I_test": "I-CHECK",
        "B_physiology": "B-SIGNS",
        "I_physiology": "I-SIGNS",
        "B_symptom": "B-SIGNS",
        "I_symptom": "I-SIGNS",
        "B_drug": "B-TREATMENT",
        "I_drug": "I-TREATMENT",
        "B_department": "O",
        "I_department": "O",
        "B_crowd": "O",
        "I_crowd": "O",
        "B_time": "O",
        "I_time": "O",
        "B_feature": "O",
        "I_feature": "O",
        "O": "O"
    }
    min_len = 50
    X, y = load_data('./data/original_data/QANER/train.txt', labels_change_dict, min_len=min_len)
    new_x, new_y = load_data('./data/original_data/QANER/test.txt', labels_change_dict, min_len=min_len)
    X.extend(new_x)
    y.extend(new_y)
    new_x, new_y = load_data('./data/original_data/QANER/dev.txt', labels_change_dict, min_len=min_len)
    X.extend(new_x)
    y.extend(new_y)

    df_QA = pd.DataFrame({'sentence': X, 'encoding': y})
    df_QA = df_QA[df_QA['sentence'].apply(lambda x: x != [])]

    '''NER/train.txt'''

    labels_change_dict = {
        "B_疾病和诊断": "B-DISEASE",
        "I_疾病和诊断": "I-DISEASE",
        "B_解剖部位": "B-BODY",
        "I_解剖部位": "I-BODY",
        "B_手术": "B-TREATMENT",
        "I_手术": "I-TREATMENT",
        "B_实验室检验": "B-CHECK",
        "I_实验室检验": "I-CHECK",
        "B_症状": "B-SIGNS",
        "I_症状": "I-SIGNS",
        "B_影像检查": "B-CHECK",
        "I_影像检查": "I-CHECK",
        "B_药物": "B-TREATMENT",
        "I_药物": "I-TREATMENT",
        "O": "O"
    }
    min_len = 50
    X, y = load_data('./data/original_data/NER/train.txt', labels_change_dict, min_len=min_len)
    new_x, new_y = load_data('./data/original_data/NER/test.txt', labels_change_dict, min_len=min_len)
    X.extend(new_x)
    y.extend(new_y)
    new_x, new_y = load_data('./data/original_data/NER/dev.txt', labels_change_dict, min_len=min_len)
    X.extend(new_x)
    y.extend(new_y)

    df_NER = pd.DataFrame({'sentence': X, 'encoding': y})
    df_NER = df_NER[df_NER['sentence'].apply(lambda x: len(x) > 1)]

    ''' original/train.txt '''
    X = []
    y = []
    sentence = []
    labels = []
    split_pattern = re.compile(r'[。？！\.\?,!]')

    with open('./data/original_data/train.txt', "r", encoding='utf-8') as file:
        for line in file.readlines():
            # 每行为一个字符和其tag，中间用tab或者空格隔开
            line = line.strip().split('\t')
            if not line or len(line) < 2:
                X.append(sentence.copy())
                y.append(labels.copy())
                sentence.clear()
                labels.clear()
                continue
            try:
                word, tag = strQ2B(cc.convert(line[0])), line[1][-1] + '-' + line[1][:-2] if line[1] != 'O' else "O"
                if word == '\ue13b':
                    print(line)
            except:
                print('convert error!')
                print("line : ", line)
                print("word : ", cc.convert(line[0]))
            if split_pattern.match(word) and len(sentence) >= 50:
                sentence.append(word)
                labels.append(tag)
                X.append(sentence.copy())
                y.append(labels.copy())
                sentence.clear()
                labels.clear()
            else:
                sentence.append(word)
                labels.append(tag)
        if len(sentence):
            X.append(sentence.copy())
            sentence.clear()
            y.append(labels.copy())
            labels.clear()

    df = pd.DataFrame({'sentence': X, 'encoding': y})
    df = df[df['sentence'].apply(lambda x: x != [])]


    df = pd.concat([df, df_QA, df_NER], axis=0)
    df = df[df['sentence'].apply(lambda x: len(x) >= 5)]
    df = df.reset_index(drop=True)
    for i, (s, e) in enumerate(zip(df['sentence'], df['encoding'])):
        if len(s) != len(e):
            print('sentence-encoding error!')
            print(i)

    pd.Series([e for sube in df.encoding.tolist() for e in sube]).value_counts()




    def my_tokenizer(sentence,tokenizer,encoding):

        encoding = encoding.copy()
        if sentence.find("\ue13b") >0:
            i = 0
            t = 0
            while sentence.find("\ue13b", i) > 0:
                encoding.pop(sentence.find("\ue13b", i)-t)
                i = sentence.find("\ue13b", i)+1
                t += 1
        if sentence.find("\ue236") >0:
            i = 0
            t = 0
            while sentence.find("\ue236",i) > 0:
                encoding.pop(sentence.find("\ue236",i)-t)
                i = sentence.find("\ue236",i)+1
                t += 1
        sentence = re.sub("[a-zA-Z]", "[UNK]", sentence)
        sentence = re.sub("㎝", "[UNK]", sentence)
        sentence = re.sub("㎎", "[UNK]", sentence)


        sentence = tokenizer.tokenize(sentence) ## **cm ㎎
        for i,s in enumerate(sentence):
            if len(s)>1 and s not in ["[UNK]"]:
                delete_start = i + 1
                delete_times = len(s) - 1
                if "#" in s:## ##2...
                    t = sum([x == "#" for x in s])
                    delete_times -= t
                for t in range(delete_times):
                    encoding.pop(delete_start)
        encoding = ["[CLS]"] + encoding + ["[SEP]"] + (510 - len(encoding)) * ['O']
        sentence = ["[CLS]"] + sentence + ["[SEP]"] + (510 - len(sentence))*['[PAD]']

        return sentence ,encoding

    df['sentence'] = df['sentence'].apply(lambda col: ''.join(col))
    token = []
    my_encodings = []
    for sentence ,encoding in tqdm(zip(df['sentence'],df['encoding'])):
        t, e = my_tokenizer(sentence, tokenizer, encoding)
        assert t.index("[SEP]") == e.index("[SEP]")
        token.append(t)
        my_encodings.append(e)

    df['token'] = token
    df['encoding'] = my_encodings


    train, test = \
        np.split(df.sample(frac=1, random_state=42),
                 [int(.8 * len(df))])
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    print('max length :',len(max(list(df["encoding"]) ,key=len)))




    with open("./pickle/train_data.pkl", "wb") as file:
        pickle.dump((train['sentence'],train['token'], train['encoding']), file)

    with open("./pickle/test_data.pkl", "wb") as file:
        pickle.dump((test['sentence'],test['token'], test['encoding']), file)


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    preprocess(tokenizer)

