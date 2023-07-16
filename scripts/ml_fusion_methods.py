import pickle 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import argparse 
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 


class MLModel():
  def __init__(self, model_type='logistic'): 
    self.model = None 
    self.model_type = model_type
    if model_type == 'logistic' : 
      self.model = LogisticRegression() 
    elif model_type == 'SVC': 
      self.model = SVC(kernel='linear')
    elif model_type=='decision_tree': 
      self.model = DecisionTreeClassifier() 
    elif model_type == 'random_forest': 
      self.model = RandomForestClassifier(n_estimators=20, random_state=42)


  def train_model(self, X_train,y_train): 
    if self.model_type=='logistic': 
      self.model.fit(X_train, y_train) 
    elif self.model_type == 'SVC': 
      self.model.fit(X_train, y_train)
    elif self.model_type == 'decision_tree': 
      self.model.fit(X_train, y_train)
    elif self.model_type == 'random_forest': 
      self.model.fit(X_train, y_train)

  
  def eval_model(self, X_test, result_type = 'classes'): 
    if result_type=='classes': 
      return self.model.predict(X_test) 
    elif result_type == 'probabilities': 
      return self.model.predict_proba(X_test) 

  def cross_validate(self, X_train, y_train): 
    scores = cross_val_score(self.model, X_train, y_train, cv=5)
    print('Cross-Validation Accuracy Scores', scores) 
    series_scores = pd.Series(scores) 
    print("Minimum Accuracy: ", round(scores.min(), 3))
    print("Average Accuracy: ", round(scores.mean(), 3))
    print("Maximum Accuracy: ", round(scores.max(), 3))
  

  def get_classification_report(self, X_test, y_test): 
    y_pred = self.eval_model(X_test, result_type='classes')

    print('y_pred = ', list(y_pred))
    print('y_test =', y_test.to_list())
    print(classification_report(y_test, y_pred, digits=3))
    


def get_pickled_file(path_to_file): 
  data = None 

  with open(path_to_file, 'rb') as file:
    data = pickle.load(file)
  return data 

def get_tfidf_embedding(path_to_file): 
  data = get_pickled_file(path_to_file) 
  embeddings = [] 
  for i in range(data.shape[0]): 
    embeddings.append(data[i].toarray().tolist()[0])
  return embeddings 

def get_vad(dataset): 
  vad = [] 
  for i,j in dataset.iterrows(): 
    vad.append([j['arousal'], j['valence'], j['dominance']]) 
  return vad 

def concat_tfidf(embeddings, vad):
  concat_array = [] 
  for i in range(len(embeddings)): 
    concat_array.append(embeddings[i]+vad[i])
  return np.array(concat_array)  

def append_tfidf(embeddings, vad):
  append_vectors = [] 
  for i in range(len(embeddings)): 
    vector = [ embeddings[i], vad[i]+[0]*(len(embeddings[i])-3)] 
    append_vectors.append(vector) 
  
  append_vectors = np.array(append_vectors) 
  x,y,z = append_vectors.shape
  X = np.reshape(append_vectors, (x, z*y)) 
  return X


def driver_code(train_dataset_path='./data/curate_misogyny_dataset.csv', 
                test_dataset_path='./data/eval_data.csv', 
                train_embeddings_path = './data/embeddings/tfidf_train_cleaned.pkl' , 
                test_embeddings_path ='./data/embeddings/tfidf_eval_cleaned.pkl' , 
                x_label='transcript', 
                y_label='label', 
                embedding = 'tfidf', 
                method=None, 
                model_type='logistic'):
  train_dataset = pd.read_csv(train_dataset_path) 
  test_dataset = pd.read_csv(test_dataset_path)
  embeddings = None 
  vad = None 
  X_train, y_train, X_test, y_test = None, None, None, None 
  if embedding == 'tfidf': 
    train_embeddings = get_tfidf_embedding(train_embeddings_path)
    test_embeddings = get_tfidf_embedding(test_embeddings_path) 
    print(len(train_embeddings[0])) 
    print(len(test_embeddings[0]))
    if method == 'concat': 
      train_vad = get_vad(train_dataset)
      test_vad = get_vad(test_dataset)
      X_train = concat_tfidf(train_embeddings, train_vad) 
      X_test = concat_tfidf(test_embeddings, test_vad) 
    elif method == 'append': 
      train_vad = get_vad(train_dataset)
      test_vad = get_vad(test_dataset)
      X_train = append_tfidf(train_embeddings, train_vad) 
      X_test = append_tfidf(test_embeddings, test_vad) 

    elif method == 'text-only': 
    
      X_train = train_embeddings
      X_test = test_embeddings 

    y_train = train_dataset[y_label]
    y_test = test_dataset[y_label]
    model = MLModel(model_type=model_type) 
    
    model.train_model(X_train, y_train) 
    print('Statistic for ', model_type, ' classifier')
    model.cross_validate(X_train, y_train)
    model.get_classification_report(X_test, y_test) 
  

if __name__=='__main__':
  arg= argparse.ArgumentParser() 
  arg.add_argument('--train_dataset', type=str, default='./data/train_set.csv')
  arg.add_argument('--test_dataset', type=str, default='./data/eval_set.csv')
  arg.add_argument('--train_embeddings_path', type=str, default='./data/embeddings/tfidf_train_cleaned.pkl' ) 
  arg.add_argument('--test_embeddings_path', type=str, default='./data/embeddings/eval_tfidf_cleaned.pkl' ) 

  arg.add_argument('--x_label', type=str, default='transcript') 
  arg.add_argument('--y_label', type=str, default='label') 
  arg.add_argument('--embedding', type=str, default='tfidf') 
  arg.add_argument('--method', type=str, default='concat' )
  arg.add_argument('--model_type', type=str, default='logistic' )
  args = arg.parse_args()
  driver_code(
    train_dataset_path = args.train_dataset, 
    test_dataset_path = args.test_dataset, 
    train_embeddings_path = args.train_embeddings_path, 
    test_embeddings_path = args.test_embeddings_path,
    x_label=args.x_label, 
    y_label=args.y_label, 
    embedding = args.embedding, 
    method=args.method, 
    model_type= args.model_type
  )





