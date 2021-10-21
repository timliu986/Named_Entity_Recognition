import sys
import os
import pandas as pd
from transformers import BertModel ,BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn
import torch
from torchcrf import CRF


class CRFBertModelForNER(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self ,config):
        self.num_labels = config.num_labels
        super(CRFBertModelForNER, self).__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(self.num_labels)

        self.init_weights()


    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None
        ):

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]

            sequence_output = self.dropout(sequence_output)
            emissions = self.classifier(sequence_output)
            emissions = emissions.permute(1, 0, 2)
            if attention_mask is not None:
                mask = attention_mask.permute(1, 0).type(torch.uint8)
                logits = self.crf.decode(emissions, mask)
                logits = [item + (512 - len(item)) * [0] for item in logits]
            else:
                logits = self.crf.decode(emissions)

            logits = torch.tensor(logits ,dtype = torch.int64)

            loss = None
            if labels is not None:
                # Only keep active parts of the loss
                tag = labels.permute(1, 0).long()
                if attention_mask is not None:

                    mask = attention_mask.permute(1, 0).type(torch.uint8)
                    loss = -self.crf(emissions, tag, mask=mask)

                else:
                    loss = -self.crf(emissions, tag)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


from transformers import  BertTokenizer, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from dataset import QADataset
import numpy as np
import csv
from tqdm import tqdm
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as clf_rep2
import os
import matplotlib.pyplot as plt
import re


PRETRAINED_MODEL_NAME = "hfl/chinese-bert-wwm-ext"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ner_model(object):
    def __init__(self ,number_class):
        """
        build model,set tokenizer and set random seed
        number_class : number of output class
        """
        self.tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        self.number_class = number_class

        self.model = CRFBertModelForNER.from_pretrained(
            PRETRAINED_MODEL_NAME, num_labels=self.number_class
        )

        if torch.cuda.is_available():
            self.model.cuda()

        torch.manual_seed(0)

    def load_model(self, trained_model_version, is_other_mission_model=False):
        """
        load trainded model
        :param trained_model_version:trained model name
        :param is_other_mission_model:if it is other mission model
        """

        state_dict = torch.load(f"D:/tim_work_dir/Auxiliary_answer/Named_Entity_Recognition/version4/model/model_v{trained_model_version}.pt", map_location='cpu')
        if is_other_mission_model:
            state_dict.pop('classifier.weight')
            state_dict.pop('classifier.bias')
        self.model.load_state_dict(
            state_dict, strict=False
        )

    def load_data(self,data_type,batch_size):
        """
        load data set and set dataloder
        :param data_type:load dataset type
        :param batch_size:batch size
        :return:
        dataset:dataset
        data:dataloder
        """
        dataset = QADataset(tokenizer=self.tokenizer, mode=data_type)
        data = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return dataset, data


    def train_validation_split(self, batch_size ,train_dataset , train_ratio):
        """

        :param batch_size: batch_szie
        :param train_dataset: train_dataset to split
        :param train_ratio: split ratio
        :param number_class: how much class to split
        :return: splited train and validation dataloader
        """
        train_size = int(train_ratio * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, validation_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size]
        )
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_data = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

        return train_data, validation_data

    def set_model(self, frezze, specific_lr,weight_decay):
        bert_param_optimizer = list(self.model.bert.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        self.optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params":
                    self.model.classifier.bias
                ,
                "lr": specific_lr,
                "weight_decay": 0.0,
            },
            {
                "params":
                    self.model.classifier.weight
                ,
                "lr": specific_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    self.model.crf
                ],
                "lr": specific_lr
            },
        ]

        if frezze:
            self.freeze_layers()
        # 用transformers的optimizer


    def freeze_layers(self, unfreeze_layers=None):
        """
        :param unfreeze_layers: set unfrezze layer
        :return:
        """
        if unfreeze_layers == None:
            unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler', 'out.']
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break



    def train_process(self, epochs,
                      warmup_steps,
                      max_norm,
                      model_version,
                      early_stop=True,
                      loss_log=None,
                      save_performance=True,
                      save_model=True):
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=epochs * len(self.train_data)
        )
        print("start training")
        max_grad_norm = max_norm
        stopping = 0
        last_loss = 1e20
        if loss_log == None:
            loss_log = {'train_loss': [],
                        'validation_loss': []}
        else:
            loss_log = loss_log
        for epoch in range(epochs):

            # 把model變成train mode
            self.model.train()
            total_train_loss = 0
            train_steps_per_epoch = 0
            for batch in tqdm(self.train_data):
                token_ids, mask_ids, _, key_ids, _, _ = batch
                # 將data移到gpu
                token_ids = token_ids.to(device)
                mask_ids = mask_ids.to(device)
                key_ids = key_ids.to(device)

                # 將optimizer 歸零
                self.optimizer.zero_grad()
                outputs = self.model(
                        token_ids,
                        token_type_ids=None,
                        attention_mask=mask_ids,
                        labels=key_ids
                    )


                loss = outputs.loss
                loss.backward()

                total_train_loss += loss.item()
                train_steps_per_epoch += 1

                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

            epoch_loss = total_train_loss / train_steps_per_epoch
            print(
                f"Epoch:{epoch + 1}/{epochs}\tTrain Loss: \
                                {epoch_loss}"
            )
            loss_log['train_loss'].append(epoch_loss)
            validation_loss, performance_single, performance_full = self.validation()
            loss_log['validation_loss'].append(validation_loss)
            if save_performance:
                self.save_performance(model_version = model_version,
                                      loss_log = loss_log,
                                      performance_single = performance_single,
                                      performance_full = performance_full)
            if last_loss > validation_loss:
                last_loss = validation_loss
                stopping = 0
                if save_model:
                    self.save(model_version = model_version + 'loss')
            else:
                stopping += 1

            if early_stop:
                if stopping == 2:
                    print('early stopping!')
                    break

        return loss_log, performance_single, performance_full


    def save_performance(self ,model_version ,loss_log ,performance_single ,performance_full):
        dirpath = f'./model/model_log/metrices/model_v{model_version}'
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
            print("Directory ", dirpath, " Created ")

        try :

            performance_single = pd.DataFrame(performance_single).applymap(lambda y:[y])
            self.performance_single_log = self.performance_single_log.apply(lambda x : x+performance_single[x.name])
            self.performance_single_log.to_csv(dirpath + '/performance_single.csv', encoding='utf-8-sig')

            performance_full = pd.DataFrame(performance_full).transpose().applymap(lambda y: [y])
            self.performance_full_log = self.performance_full_log.apply(lambda x: x + performance_full[x.name])
            self.performance_full_log.to_csv(dirpath + '/performance_full.csv', encoding='utf-8-sig')

            self.loss_log = self.loss_log.append({
                'train_loss' : [loss_log['train_loss'][-1]],
                'validation_loss' : [loss_log['validation_loss'][-1]]
            } ,ignore_index = True)
            self.loss_log.to_csv(dirpath+'/loss_log.csv',encoding='utf-8-sig')
            print('update ! ')
        except:
            self.performance_single_log = pd.DataFrame(performance_single).applymap(lambda y:[y])
            self.performance_single_log.to_csv(dirpath+'/performance_single.csv',encoding='utf-8-sig')

            self.performance_full_log = pd.DataFrame(performance_full).transpose().applymap(lambda y:[y])
            self.performance_full_log.to_csv(dirpath + '/performance_full.csv', encoding='utf-8-sig')

            self.loss_log = pd.DataFrame({
                'train_loss': loss_log['train_loss'],
                'validation_loss': loss_log['validation_loss']
            })
            self.loss_log.to_csv(dirpath + '/loss_log.csv', encoding='utf-8-sig')


    def train(self, batch_size, learning_rate,crf_lr,
              warmup_steps, frezze, epochs, max_norm,weight_decay,
              model_version, early_stop=True,loss_log=None):
        torch.manual_seed(0)
        self.train_dataset, _ = self.load_data(data_type='train', batch_size=batch_size)
        self.idx2label = self.train_dataset.idx2label
        self.set_model(frezze=frezze,
                       specific_lr = crf_lr,
                       weight_decay = weight_decay)
        self.optimizer = AdamW(self.optimizer_grouped_parameters, lr=learning_rate)
        self.train_data, self.validation_data = self.train_validation_split(batch_size, self.train_dataset,
                                                                            train_ratio=0.8)
        loss_log, performance_single, performance_full = self.train_process(epochs=epochs,
                                                                            warmup_steps=warmup_steps,
                                                                            max_norm=max_norm,
                                                                            model_version=model_version,
                                                                            early_stop=early_stop,
                                                                            loss_log=loss_log,
                                                                            save_performance=True)
        self.plot_loss(loss_log)

    def k_fold_train(self, k, batch_size, learning_rate, frezze, epochs, warmup_steps, max_norm, model_version,
                     early_stop=True, loss_log=None):
        torch.manual_seed(0)

        self.k_fold_single_performance = pd.DataFrame(columns=['weighted_precision',
                                                        'macro_precision',
                                                        'weighted_recall',
                                                        'macro_recall',
                                                        'weighted_f1',
                                                        'macro_f1'])
        self.k_fold_entity_performance = pd.DataFrame(columns=['weighted_precision',
                                                        'macro_precision',
                                                        'weighted_recall',
                                                        'macro_recall',
                                                        'weighted_f1',
                                                        'macro_f1'])

        self.train_dataset, _ = self.load_data(data_type='train', batch_size=batch_size)
        split_ratio = [int(len(self.train_dataset) / k)] * (k - 1) + [
            len(self.train_dataset) - (k - 1) * int(len(self.train_dataset) / k)]
        self.k_fold_train_set = torch.utils.data.random_split(
            self.train_dataset, split_ratio
        )

        for i in range(k):
            self.model = CRFBertModelForNER.from_pretrained(
                PRETRAINED_MODEL_NAME, num_labels=self.num_class
            )
            if torch.cuda.is_available():
                self.model.cuda()
            self.set_model(frezze=frezze)
            self.optimizer = AdamW(self.optimizer_grouped_parameters, lr=learning_rate)

            print(f'{i + 1}/{k} fold cross validation :')
            kth_train_fold = [fold for it, fold in enumerate(self.k_fold_train_set) if it != i]
            train_dataset = torch.utils.data.ConcatDataset(kth_train_fold)
            validation_dataset = self.k_fold_train_set[i]
            self.train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.validation_data = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

            loss_log, performance_single, performance_full = self.train_process(epochs=epochs,
                                                                      warmup_steps=warmup_steps,
                                                                      max_norm=max_norm,
                                                                      model_version=model_version,
                                                                      early_stop=early_stop,
                                                                      loss_log=loss_log,
                                                                      save_model=False,
                                                                      save_performance=False)
            k_fold_performance = pd.DataFrame({'weighted_precision':performance_single['weighted avg']['precision'],
                                                    'macro_precision':performance_single['macro avg']['precision'],
                                                    'weighted_recall':performance_single['weighted avg']['recall'],
                                                    'macro_recall':performance_single['macro avg']['recall'],
                                                    'weighted_f1':performance_single['weighted avg']['f1-score'],
                                                    'macro_f1':performance_single['macro avg']['f1-score']}, index=[f'{i + 1}_fold'])
            self.k_fold_single_performance = pd.concat([self.k_fold_single_performance, k_fold_performance], axis=0)

            k_fold_performance = pd.DataFrame({'weighted_precision':performance_full['weighted avg']['precision'],
                                                    'macro_precision':performance_full['macro avg']['precision'],
                                                    'weighted_recall':performance_full['weighted avg']['recall'],
                                                    'macro_recall':performance_full['macro avg']['recall'],
                                                    'weighted_f1':performance_full['weighted avg']['f1-score'],
                                                    'macro_f1':performance_full['macro avg']['f1-score']}, index=[f'{i + 1}_fold'])
            self.k_fold_entity_performance = pd.concat([self.k_fold_entity_performance, k_fold_performance], axis=0)

        self.k_fold_single_performance.loc['mean'] = self.k_fold_single_performance.mean(axis=0)
        self.k_fold_entity_performance.loc['mean'] = self.k_fold_entity_performance.mean(axis=0)

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(self.k_fold_single_performance)
            print('='*50)
            print(self.k_fold_entity_performance)
        self.k_fold_single_performance.to_csv(f'./result/model_v{model_version}_{k}_FoldTrain_TotalPerformance_single.csv')
        self.k_fold_entity_performance.to_csv(f'./result/model_v{model_version}_{k}_FoldTrain_TotalPerformance_entity.csv')

    def plot_loss(self, loss_log):
        plt.plot(loss_log['train_loss'], label='Train')
        plt.plot(loss_log['validation_loss'], label='validation')
        plt.legend()
        plt.ylabel('sigmoid_binary_cross_entropy')
        plt.xlabel('epochs')
        plt.title('Loss')
        plt.show()


    def validation(self):
        self.model.eval()
        predict_key = []
        predict = []
        true_label = []
        true_label_key = []
        total_loss = 0
        i = 0

        for test_batch in tqdm(self.validation_data):
            token_ids, mask_ids,mask2, key_ids, _, full_sentnece = test_batch
            token_ids = token_ids.to(device)
            mask_ids = mask_ids.to(device)
            key_ids = key_ids.to(device)
            with torch.no_grad():
                output = self.model(
                    token_ids, token_type_ids=None, attention_mask=mask_ids, labels=key_ids
                )
            loss = output[0].detach().cpu().numpy()
            output = output[1].detach().cpu().numpy()
            label = key_ids.detach().cpu().numpy()
            mask_flat = np.array(mask2).flatten()
            pred_key = output.flatten()[mask_flat > 0]
            pred_key = np.where(pred_key > 10 , 0 ,pred_key)
            label_key = label.flatten()[mask_flat > 0]
            tagging_pred = [self.idx2label[i] for i in pred_key]
            tagging_true = [self.idx2label[i] for i in label_key]
            predict_key.extend(pred_key)
            true_label_key.extend(label_key)
            predict.append(tagging_pred)
            true_label.append(tagging_true)
            total_loss += loss
            i += 1

        avg_loss = total_loss / i
        print(f"Validtaion loss : {avg_loss}")
        print('=' * 50)
        performance_single,performance_full = self.performance_report(true_label_key = true_label_key,
                                                                      true_label = true_label,
                                                                      predict_key = predict_key,
                                                                      predict = predict)

        return avg_loss, performance_single,performance_full


    def performance_report(self,true_label_key,true_label, predict_key,predict):
        print(classification_report(true_label_key, predict_key , target_names=[ 'O',
                                                                         'B-TREATMENT',
                                                                         'I-TREATMENT',
                                                                         'B-BODY',
                                                                         'I-BODY',
                                                                         'B-SIGNS',
                                                                         'I-SIGNS',
                                                                         'B-CHECK',
                                                                         'I-CHECK',
                                                                         'B-DISEASE',
                                                                         'I-DISEASE' ]))
        performance_single = classification_report(true_label_key, predict_key , target_names=[ 'O',
                                                                         'B-TREATMENT',
                                                                         'I-TREATMENT',
                                                                         'B-BODY',
                                                                         'I-BODY',
                                                                         'B-SIGNS',
                                                                         'I-SIGNS',
                                                                         'B-CHECK',
                                                                         'I-CHECK',
                                                                         'B-DISEASE',
                                                                         'I-DISEASE'],output_dict=True)
        print('='*50)

        print(clf_rep2(true_label, predict))
        performance_full = clf_rep2(true_label, predict, output_dict=True)

        return performance_single,performance_full

    def test(self,batch_size ,save ,result_data_name = 'result'):
        test_dataset,test_data = self.load_data(data_type = 'test' ,batch_size=batch_size)
        self.idx2label = test_dataset.idx2label
        self.model.eval()
        predict_key = []
        predict = []
        true_label = []
        true_label_key = []
        total_loss = 0
        i = 0
        total_data = pd.DataFrame(columns=['Sentence', 'True_entity','Pred_entity'])
        for test_batch in tqdm(test_data):
            token_ids, mask_ids,_, key_ids, _, sentence = test_batch
            token_ids = token_ids.to(device)
            mask_ids = mask_ids.to(device)
            key_ids = key_ids.to(device)
            with torch.no_grad():
                output = self.model(
                    token_ids, token_type_ids=None, attention_mask=mask_ids, labels=key_ids
                )
            loss = output[0].detach().cpu().numpy()
            output = output[1].detach().cpu().numpy()
            label = key_ids.detach().cpu().numpy()

            entity_pred = [self.entity_tagging(sentence = s, pred = [self.idx2label[i] for i in p]) for s, p in zip(sentence ,output)]
            entity_true = [self.entity_tagging(sentence = s, pred = [self.idx2label[i] for i in t]) for s, t in zip(sentence ,label)]
            total_data = total_data.append(pd.DataFrame({'Sentence': list(sentence),
                                                         'True_entity': entity_pred,
                                                         'Pred_entity': entity_true}), ignore_index=True)
            mask = mask_ids.to("cpu")
            mask_flat = np.array(mask).flatten()
            pred_key = output.flatten()[mask_flat > 0]
            pred_key = np.where(pred_key > 10 , 0 ,pred_key)
            label_key = label.flatten()[mask_flat > 0]
            label_key = np.where(label_key > 10 , 0 ,label_key)
            tagging_pred = [self.idx2label[i] for i in pred_key]
            tagging_true = [self.idx2label[i] for i in label_key]

            predict_key.extend(pred_key)
            true_label_key.extend(label_key)
            predict.append(tagging_pred)
            true_label.append(tagging_true)
            total_loss += loss
            i += 1

        avg_loss = total_loss / i
        print(f"Validtaion loss : {avg_loss}")
        print('=' * 50)
        performance_single, performance_full = self.performance_report(true_label_key=true_label_key,
                                                                       true_label=true_label,
                                                                       predict_key=predict_key,
                                                                       predict=predict)

        if save:
            total_data.to_csv(f'./result/{result_data_name}.csv', encoding='utf-8-sig')

    def batch_cut(self, sentence, cut_length):
        out = []
        split_word = ['。', '？', '?']
        if set(split_word).intersection(sentence):
            sub_sentence = []
            for s in sentence:
                sub_sentence.append(s)
                if len(sub_sentence) == cut_length or (len(sub_sentence) > 100 and s in split_word):
                    out.append(sub_sentence)
                    sub_sentence = []
            return out
        else:
            l = len(sentence)
            for ndx in range(0, l, cut_length):
                out.append(sentence[ndx:min(ndx + cut_length, l)])
            return out

    def test_for_input(self ,sentence ,return_sequence):
        self.model.cuda()
        self.model.eval()
        sentence_tokenizer = re.sub("[a-zA-Z]", "[UNK]", sentence)
        sentence_tokenizer = re.sub(r'[0-9]',"[UNK]" ,sentence_tokenizer)
        sentence_tokenizer = re.sub("\\s+", "", sentence_tokenizer)
        sentence_tokenizer = re.sub(" ", "", sentence_tokenizer)
        sentence_tokenizer = sentence_tokenizer.replace(".", "[UNK]")
        # 刪除多餘換行
        sentence_tokenizer = re.sub("\r*\n*", "", sentence_tokenizer)
        # 刪除括號內英文

        # 將輸入加入cls並padding到max len
        sentence_tokenizer = self.tokenizer.tokenize(sentence_tokenizer)
        if len(sentence_tokenizer) > 510 :

            print('length cut')
            sentence_tokenizer = self.batch_cut(sentence_tokenizer,cut_length = 510)
            sentence_tokenizer = [["[CLS]"] + s + ["[SEP]"] + ["[PAD]"] * (510 - len(s) if len(s) < 510 else 0) for s in sentence_tokenizer]
            sentence_ids = [self.tokenizer.convert_tokens_to_ids(s) for s in sentence_tokenizer]
            mask = [[float(i > 0) for i in sub_sentence_ids] for sub_sentence_ids in sentence_ids ]
            sentence_ids = torch.tensor(
                sentence_ids, dtype=torch.long).to(device)
            mask = torch.tensor(mask, dtype=torch.long).to(device)
        else:

        # 將輸入斷詞並產生mask

            sentence_tokenizer = ["[CLS]"] + sentence_tokenizer + ["[SEP]"] + ["[PAD]"] * (510 - len(sentence_tokenizer))
            sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokenizer)
            mask = [float(i > 0) for i in sentence_ids]
            sentence_ids = torch.tensor(
                [sentence_ids], dtype=torch.long).to(device)
            mask = torch.tensor([mask], dtype=torch.long).to(device)
            with torch.no_grad():
                output = self.model(
                    sentence_ids, token_type_ids=None, attention_mask=mask)
            pred = output.logits.cpu().numpy().flatten()
            pred = [self.idx2label[i] for i in pred]
            entity = self.entity_tagging(sentence = sentence ,pred = pred)
            if return_sequence:
                return entity,pred
            else:
                return entity

    def entity_tagging(self,sentence,pred):
        sentence = re.sub("\\s+", "", sentence)
        sentence = re.sub("\r*\n*", "", sentence)
        sentence = re.sub(" ", "", sentence)
        english_sentence = re.findall('[a-zA-Z]+',sentence)
        sentence = ["[CLS]"] + [s for s in sentence] + ['[SEP]'] + ['[PAD]']*(510-len(sentence))
        entity_dic = {
            'TREATMENT': [],
            'BODY': [],
            'SIGNS': [],
            'CHECK': [],
            'DISEASE': []
        }
        entity = ''
        Type = ''
        for sen , p in zip(sentence,pred):
            if p[0] == "B":
                if Type != '':
                    for s in english_sentence:
                        if entity in s:
                            entity = s
                    entity_dic[Type].append(entity)
                entity = sen
                Type = p[2:]
            elif p[0] == "I" and p[2:] == Type:
                entity += sen
            else:
                if Type != '':
                    for s in english_sentence:
                        if entity in s:
                            entity = s
                    entity_dic[Type].append(entity)
                    entity = ''
                    Type = ''

        '''
        pair_df = pd.DataFrame({
            'token':sentence,
            'pred' : pred
        })
        '''
        return entity_dic


    def save(self,model_version):
        torch.save(self.model.state_dict(), f"./model/model_v{model_version}.pt")


if __name__ == '__main__':
    model = CRFBertModelForNER.from_pretrained(PRETRAINED_MODEL_NAME ,num_labels=13)
    model.classifier.bias
    for n,param in model.named_parameters():
        print(n,param.size())