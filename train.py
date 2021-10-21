from model import ner_model
import torch
import os


PRETRAINED_MODEL_NAME = "hfl/chinese-bert-wwm-ext"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model_version = input('model_version : ')
    model_name = f'model_v{model_version}.pt'

    hyperparameter = {"model_version": model_version,
                      'PRETRAINED_MODEL_NAME': PRETRAINED_MODEL_NAME,
                      "number_class" : 13,
                      "frezze": False,
                      "batch_size": 15,
                      "loss_weight": None,#torch.tensor([0.02]+[1.0]*10, dtype=torch.float).to(device),
                      "max_norm": 1.0,
                      "epochs": 15,
                      "learning_rate": 0.00002,
                      "crf_learning_rate":0.00002*10,
                      "warmup_steps": 3,
                      "multiple_label" : True
                      }

    model = ner_model(
                           number_class = hyperparameter['number_class']
                           )

    if model_name in os.listdir('./model'):
        print('='*10)
        print(f'retrain model {model_name}')
        print('=' * 10)
        model.load_model(model_version,is_other_mission_model = True)
        try:
            with open(f'./model/model_log/modelv{model_version}.txt') as file:
                for lines in file:
                    line = lines.split(' : ')
                    if line == [' ']:
                        break
                    key = line[0]
                    val = line[1][:-1]
                    if key == 'loss_log' :
                        loss_log = val
                model.train(batch_size=hyperparameter['batch_size'],
                            max_norm=hyperparameter['max_norm'],
                            epochs=hyperparameter['epochs'],
                            frezze=hyperparameter['frezze'],
                            learning_rate=hyperparameter['learning_rate'],
                            warmup_steps=hyperparameter['warmup_steps'],
                            model_version=hyperparameter['model_version'],
                            loss_log=loss_log,
                            crf_lr=hyperparameter['crf_learning_rate']
                            )
        except:
            print('no model log!')
            model.train(    batch_size=hyperparameter['batch_size'],
                            max_norm=hyperparameter['max_norm'],
                            epochs=hyperparameter['epochs'],
                            frezze=hyperparameter['frezze'],
                            learning_rate=hyperparameter['learning_rate'],
                            warmup_steps=hyperparameter['warmup_steps'],
                            model_version=hyperparameter['model_version'],
                            crf_lr=hyperparameter['crf_learning_rate']
                            )

    else:
        '''
        model.set_model(frezze = True)
        model.validation(batch_size_v = hyperparameter['batch_size'])
        '''
        model.train(batch_size=hyperparameter['batch_size'],
                    max_norm=hyperparameter['max_norm'],
                    epochs=hyperparameter['epochs'],
                    frezze=hyperparameter['frezze'],
                    learning_rate=hyperparameter['learning_rate'],
                    warmup_steps=hyperparameter['warmup_steps'],
                    model_version=hyperparameter['model_version'],
                    crf_lr=hyperparameter['crf_learning_rate']
                    )

    hyperparameter['loss_log'] = model.loss_log

    with open(f'./model/model_log/modelv{model_version}.txt', 'w') as file:
        for item, value in zip(hyperparameter.keys(),hyperparameter.values()):
            file.write('%s : %s \n ' %(item,value))
        