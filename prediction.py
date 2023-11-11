import pandas as pd
import pickle

def predict_disease(config):
    df=pd.read_csv('./Final_Train_data.csv')
    df.drop(axis=1,columns=['Unnamed: 0','fluid_overload'],inplace=True)

    pkl_filename1="Project_model.pkl"
    pkl_filename2="encoder.pkl"

    with open(pkl_filename1,'rb') as f_in:
        model=pickle.load(f_in)

    with open(pkl_filename2,'rb') as e_in:
        encoder=pickle.load(e_in)   

    inp = pd.DataFrame({i:0 for i in df.drop('prognosis',axis=1).columns},index=[0])
    for symptoms in config:
        inp[symptoms]=1     
    
    ans=encoder.inverse_transform(model.predict(inp))
    return ans

def similar(dis):
    data=pd.read_csv('./data_for_sim_dos.csv')
    similarity=pd.read_csv('./similarity.csv')

    idx = data.index.to_list().index(dis)
    distances = similarity[idx]
    dis_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:4]
    final_list=[]
    for i in dis_list:
      final_list.append(data.index[i[0]])

    return final_list
