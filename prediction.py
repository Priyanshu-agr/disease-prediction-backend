import pandas as pd
import pickle
# import sklearn

def predict_disease(config):
    df=pd.read_csv('../Final_Train_Data.csv')

    pkl_filename1="../model.pkl"
    pkl_filename2="../encoder.pkl"

    with open(pkl_filename1,'rb') as f_in:
        model=pickle.load(f_in)

    with open(pkl_filename2,'rb') as e_in:
        encoder=pickle.load(e_in)            

    all_symptoms = [col for col in df.drop(['prognosis','Unnamed: 0','Unnamed: 145','\xa0'], axis=1).columns]
    new_names=[]
    for name in all_symptoms:
        name=name.replace('\xa0',' ')
        new_names.append(name)

    all_symptoms=new_names

    inp = {col : 0 for col in all_symptoms }

    for dis in config:
        inp.update({dis : 1})

    inp = pd.DataFrame(inp , index=[0])
    ans = encoder.inverse_transform(model.predict(inp))[0]

    return ans
    
def similar(dis):
    data=pd.read_csv('../data_for_sim_dos.csv')
    similarity=pd.read_csv('../similarity.csv')

    idx = data.index.to_list().index(dis)
    distances = similarity[idx]
    dis_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:4]
    final_list=[]
    for i in dis_list:
      final_list.append(data.index[i[0]])

    return final_list
