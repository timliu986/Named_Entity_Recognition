"""
訓練bert模型

看看要不要調整loss計算方法
"""
from transformers import  BertTokenizer, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from dataset import QADataset
from model import FancyBertModelForTokenClassification
import torch
import numpy as np
import csv
from tqdm import tqdm
from sklearn.metrics import classification_report

# PRETRAINED_MODEL_NAME = "bert-base-chinese"
PRETRAINED_MODEL_NAME = "hfl/chinese-bert-wwm-ext"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class training_model(object):
    def __init__(self,
                 batch_size=12,
                 loss_weight = None,
                 frezze = False
                 ):
        """
        input:
            batch_size: 數據batch的大小
            learning_rate: 數據學習率
            max_norm: 梯度正規剪裁的數字
            epochs: 迭代數
            if_validation: 是否需要做validation，若需要，則會分 10%的訓練資料做validation
            save_validation: 是否需要將validation的結果儲存，若是True，
                            則會將validation的最後結果儲存在compare.csv
                            這個選項要跟if_validation一起打開才有效
        output:
            將model參數用torch.save儲存
        """
        number_class = 11
        if loss_weight == None:
            loss_weight = torch.tensor([1.0]*number_class, dtype=torch.float).to(device)
        else:
            assert len(loss_weight) == number_class
            loss_weight = torch.tensor(loss_weight, dtype=torch.float).to(device)

        self.model = FancyBertModelForTokenClassification.from_pretrained(
            PRETRAINED_MODEL_NAME, num_labels=number_class, loss_weight=loss_weight
        )  # 輸出的label最多到4

        if frezze:
            self.freeze_layers()

        if torch.cuda.is_available():
            self.model.cuda()


        self.train_data ,self.test_data = self.load_data(batch_size)

        self.build_model()
        '''
        self.train(max_norm = max_norm ,epochs = epochs)
        self.validation()
        if save_validation:
            self.save_validation()
        '''


    def load_data(self,batch_size,train_ratio = 0.9):

        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        train_dataset = QADataset(tokenizer=tokenizer)
        train_size = int(train_ratio * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size]
        )

        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


        return train_data,test_data

    def build_model(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        self.optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        # 用transformers的optimizer

    def freeze_layers(self ,unfreeze_layers = None):
        if unfreeze_layers == None:
            unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler', 'out.']
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

    def train(self ,max_norm ,epochs , learning_rate,warmup_steps):
        self.optimizer = AdamW(self.optimizer_grouped_parameters, lr=learning_rate)
        # 使用schedular調整learning rate
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps = warmup_steps,
            num_training_steps=epochs * len(self.train_data)
        )

        print("start training")
        max_grad_norm = max_norm
        stopping = 0
        last_loss = 1e20
        for epoch in range(epochs):

            # 把model變成train mode
            self.model.train()
            total_train_loss = 0
            train_steps_per_epoch = 0
            for batch in tqdm(self.train_data):
                questions_ids, mask_ids, key_ids, _, full_question = batch
                # 將data移到gpu
                questions_ids = questions_ids.to(device)
                mask_ids = mask_ids.to(device)
                key_ids = key_ids.to(device)

                # 將optimizer 歸零
                self.optimizer.zero_grad()
                outputs = self.model(
                    questions_ids,
                    token_type_ids=None,
                    attention_mask=mask_ids,
                    labels=key_ids,
                )

                # label_ids = key_ids.to("cpu").numpy()
                # tmp_eval_accuracy = flat_accuracy(outputs.logits.detach().cpu().numpy(), label_ids, mask_ids)
                # print(tmp_eval_accuracy)

                loss = outputs.loss
                # print(outputs.loss)
                loss.backward()

                total_train_loss += loss.item()
                train_steps_per_epoch += 1

                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

            print(
                f"Epoch:{epoch}\tTrain Loss: \
                    {total_train_loss / train_steps_per_epoch}"
            )

            validation_loss ,validation_acc = self.validation()

            if last_loss > validation_loss:
                last_loss = validation_loss
                stopping = 0
            else :
                stopping += 1

            if stopping == 2:
                print('early stopping!')
                break

    def validation(self):
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        eval_steps_per_epoch, eval_examples_per_epoch = 0, 0
        predictions, true_labels = [], []
        for batch in self.test_data:
            questions_ids, mask_ids, key_ids, _, full_question = batch
            questions_ids = questions_ids.to(device)
            mask_ids = mask_ids.to(device)
            key_ids = key_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = self.model(
                    questions_ids,
                    token_type_ids=None,
                    attention_mask=mask_ids,
                    labels=key_ids,
                )
                logits = self.model(
                    questions_ids, token_type_ids=None, attention_mask=mask_ids
                )

            logits = logits[0].detach().cpu().numpy()
            label_ids = key_ids.to("cpu").numpy()
            # print(logits)
            # print(label_ids)
            tmp_eval_accuracy, prediction, true_label = self.flat_accuracy(logits, label_ids, mask_ids)
            predictions.extend(prediction)
            true_labels.extend(true_label)

            eval_loss += tmp_eval_loss[0].mean().item()

            eval_accuracy += tmp_eval_accuracy



            eval_examples_per_epoch += questions_ids.size(0)
            eval_steps_per_epoch += 1
            validation_loss = eval_loss / eval_steps_per_epoch
            validation_acc = eval_accuracy / eval_steps_per_epoch

        print(f"Validation loss: {validation_loss}")
        print(
            f"Validation Accuracy: \
                        {validation_acc}"
        )
        print("-----------------------")
        print(classification_report(true_labels, predictions, target_names=['O',
                                                                             'TREATMENT-I',
                                                                             'TREATMENT-B',
                                                                             'BODY-B',
                                                                             'BODY-I',
                                                                             'SIGNS-I',
                                                                             'SIGNS-B',
                                                                             'CHECK-B',
                                                                             'CHECK-I',
                                                                             'DISEASE-I',
                                                                             'DISEASE-B'] ))
        print("-----------------------")
        return validation_loss,validation_acc

    def save(self,model_version):
        torch.save(self.model.state_dict(), f"./model/model_v{model_version}.pt")

    def save_validation(self):
        csv_file = open("comapre2.csv", "w", encoding="utf-8-sig", newline="")
        writer = csv.writer(csv_file)
        predictions = []
        self.model.eval()
        for batch in self.test_data:
            (
                questions_ids,
                mask_ids,
                key_ids,
                tokenized_question,
                full_question,
            ) = batch
            questions_ids = questions_ids.to(device)
            mask_ids = mask_ids.to(device)

            output = self.model(questions_ids, token_type_ids=None, attention_mask=mask_ids)
            output = output[0].detach().cpu().numpy()
            prediction = [list(p) for p in np.argmax(output, axis=2)]
            predictions.extend(prediction)

            for i in range(len(prediction)):
                key_word_pred = []
                key_word_true = []
                for j in range(len(prediction[i])):
                    if prediction[i][j] in [2,3,6,7,10]:#B
                        if len(key_word_pred) != 0:
                            key_word_pred.append("，")
                        key_word_pred.append(tokenized_question[j][i])

                    elif prediction[i][j]  in [1,4,5,8,9]:#I
                        key_word_pred.append(tokenized_question[j][i])


                    if key_ids[i][j] in [2,3,6,7,10]:#B
                        if j != 0:
                            key_word_true.append("，")
                        key_word_true.append(tokenized_question[j][i])

                    elif key_ids[i][j] in [1,4,5,8,9]:#I
                        key_word_true.append(tokenized_question[j][i])



                if len(key_word_pred) == 0:
                    writer.writerow([full_question[i], "".join(key_word_true), "None"])
                    continue


                key_word_pred = [x for x in key_word_pred if x != "[PAD]"]

                flag = False
                while True:
                    if len(key_word_pred) == 0:
                        writer.writerow(
                            [full_question[i], "".join(key_word_true), "None"]
                        )
                        flag = True
                        break
                    if key_word_pred[-1] == "，":
                        key_word_pred.pop()
                    else:
                        break
                if flag is True:
                    continue

                writer.writerow(
                    [full_question[i], "".join(key_word_true), "".join(key_word_pred)]
                )

    def flat_accuracy(self,preds, labels, mask):
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        mask = mask.to("cpu")
        mask_flat = np.array(mask).flatten()

        return (np.sum(pred_flat[mask_flat > 0] == labels_flat[mask_flat > 0]) / len(
            pred_flat[mask_flat > 0]
        ), pred_flat[mask_flat > 0], labels_flat[mask_flat > 0])



if __name__ == "__main__":
    model_version = input('model_version : ')

    hyperparameter = {"model_version": model_version,
                      'PRETRAINED_MODEL_NAME': PRETRAINED_MODEL_NAME,
                      "frezze": True,
                      "batch_size": 16,
                      "loss_weight": None,
                      "max_norm": 1.0,
                      "epochs": 6,
                      "learning_rate": 0.00002,
                      "warmup_steps": 3
                      }
    with open(f'./model/model_log/modelv{model_version}.txt', 'w') as file:
        for item, value in zip(hyperparameter.keys(),hyperparameter.values()):
            file.write('%s : %s \n ' %(item,value))
    
    print(hyperparameter)

    train_model = training_model(frezze = hyperparameter['frezze'],
                                 batch_size = hyperparameter['batch_size'],
                                 loss_weight = hyperparameter['loss_weight'])
    train_model.test_data
    train_model.train(max_norm = hyperparameter['max_norm'],
                      epochs = hyperparameter['epochs'],
                      learning_rate = hyperparameter['learning_rate'],
                      warmup_steps = hyperparameter['warmup_steps'],
                      )
    train_model.save(model_version)
    train_model.save_validation()

