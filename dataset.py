"""
繼承pytorch的dataset
"""
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer  # bert斷詞，以字為單位
import pickle


class QADataset(Dataset):
    def __init__(self, tokenizer, mode):
        """
        ipnut:
          tokenizer: 使用何種tokenizer
        """
        assert mode in ['train' ,'test']
        (self.sentence, self.token, self.labeled_key) = pickle.load(
            open(f"./pickle/{mode}_data.pkl", "rb")
        )

        self.label2idx = {
                 'O':0,
                 'B-TREATMENT': 1,
                 'I-TREATMENT': 2,
                 'B-BODY': 3,
                 'I-BODY': 4,
                 'B-SIGNS': 5,
                 'I-SIGNS': 6,
                 'B-CHECK': 7,
                 'I-CHECK': 8,
                 'B-DISEASE': 9,
                 'I-DISEASE': 10,
                 '[CLS]':11,
                 '[SEP]':12
                }

        self.idx2label = {
                 0:'O',
                 1:'B-TREATMENT',
                 2:'I-TREATMENT',
                 3:'B-BODY',
                 4:'I-BODY',
                 5:'B-SIGNS',
                 6:'I-SIGNS',
                 7:'B-CHECK',
                 8:'I-CHECK',
                 9:'B-DISEASE',
                 10:'I-DISEASE',
                 11:'O',
                 12:'O'
                }

        self.tokenizer = tokenizer
        self.len = len(self.token)
        # print(self.key)

    def __getitem__(self, idx):
        """
        ipnut:
          idx:要的是第幾筆的資料
        output:
          question_ids: 將以斷詞的問題轉換成
          key_ids: 將標記後的結果轉換成對應的數字後輸出
        """
        token_ids = self.tokenizer.convert_tokens_to_ids(
            self.token[idx]
        )

        mask_ids = [float(i > 0) for i in token_ids]
        mask2_ids = [float(i not in [0,101,102]) for i in token_ids]

        key_ids = []
        for token in self.labeled_key[idx]:
            key_ids.append(self.label2idx[token])

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(mask_ids, dtype=torch.long),
            torch.tensor(mask2_ids, dtype=torch.long),
            torch.tensor(key_ids, dtype=torch.long),
            self.token[idx],
            self.sentence[idx],
        )

    def __len__(self):
        """
        output: data總共有多少筆
        """
        return self.len


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    dataset = QADataset(tokenizer=tokenizer,mode = 'train')
    for i in range(20):
        print("------------")
        print(i)
        print(dataset.__getitem__(i)[2])
        # print(dataset.__getitem__(0))

