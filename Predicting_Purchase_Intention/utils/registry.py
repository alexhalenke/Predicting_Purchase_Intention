from Predicting_Purchase_Intention.utils.params import LOCAL_REGISTRY_PATH
import pickle

def save_model(model_name: str,
               model):

    pickle.dump(model, open(LOCAL_REGISTRY_PATH + model_name + '.pkl','wb'))

    return None



def load_model(model_name: str):

    model = pickle.load(open(LOCAL_REGISTRY_PATH + model_name + '.pkl', "rb"))

    return model
