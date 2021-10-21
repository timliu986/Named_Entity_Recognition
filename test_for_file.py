import pandas as pd
from model import ner_model
from tqdm import tqdm
def test_for_file(df ,model ,result_data_name):

    output = pd.DataFrame(columns=['Sentence' , 'Entity'])
    for sentence in tqdm(df.iloc[:,0]):
        entity = model.test_for_input(sentence)
        output = output.append({'Sentence': sentence,
                                'Rntity': entity}, ignore_index=True)


    output.to_csv(f'/result/{result_data_name}.csv' ,encoding='utf-8-sig')

if __name__ == '__main__':
    model_version = input('model version :　')
    data_name = input('data name : ')
    result_data_name = input ('result data name :')
    model = ner_model(number_class = 13)
    model.load_model(model_version)
    model.idx2label = {
        0: 'O',
        1: 'B-TREATMENT',
        2: 'I-TREATMENT',
        3: 'B-BODY',
        4: 'I-BODY',
        5: 'B-SIGNS',
        6: 'I-SIGNS',
        7: 'B-CHECK',
        8: 'I-CHECK',
        9: 'B-DISEASE',
        10: 'I-DISEASE',
        11: 'O',
        12: 'O'
    }
    data_path = f'./data/{data_name}.csv'
    df = pd.read_csv(data_path).dropna().reset_index(drop=True)
    df = df['Question_Title'] + '，' + df['Question_Body']
    test_for_file(df = df,model = model ,result_data_name = result_data_name)


