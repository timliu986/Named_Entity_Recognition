from model import ner_model

if __name__ == "__main__":
    version = input("model version : ")
    model = ner_model(number_class = 13)
    model.load_model(version)
    model.test(batch_size = 128,
               save = False)