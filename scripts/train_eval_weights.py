import pandas as pd 
import numpy as np 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoConfig
import torch 
from torch import cuda
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
import argparse 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MisogynyDataset(Dataset):
    def __init__(self, file_path, text_attribute, target_attribute):
        self.tokenizer = AutoTokenizer.from_pretrained("MilaNLProc/bert-base-uncased-ear-misogyny")
        self.data = pd.read_csv(file_path)
        #self.data['text'] = self.data['text'].applymap(str)
        self.text_attr = text_attribute
        self.label_attr = target_attribute 

        self.max_len = 512
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        text = self.data.loc[index, self.text_attr]
        label = self.data.loc[index, self.label_attr ]
        

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        
        return {'input_ids': encoding['input_ids'][0], 'attention_mask': encoding['attention_mask'][0], 'label': label, 'text': text}

class AutoModelClass(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = super().forward(input_ids, attention_mask=attention_mask)
        logits = self.linear(outputs[0])
        return logits



def train_model(train_loader, 
                epochs=3, 
                model_name="MilaNLProc/bert-base-uncased-ear-misogyny", 
                path_to_save='./models/ami-reddit.pt',
                num_classes=2, 
                model_lr=1e-05, 
                save_model=True, 
                class_weights=None, 
                local_model = False): 
  config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
  if not local_model: 
    model = AutoModelClass.from_pretrained(model_name, config=config)
  else: 
    model = AutoModelClass.from_pretrained(model_name, local_files_only=True)
  model.to(device)
  loss_function = None 
  optimizer = torch.optim.Adam(params =  model.parameters(), lr=model_lr)
  if num_classes==2 and class_weights is not None:
    #pos_weights = torch.tensor(class_weights[1]/class_weights[0]).to(device)
    #pos_weight = torch.tensor(class_weights[1]/class_weights[0]).to(device)
    class_weights = torch.tensor(class_weights).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss(weight=class_weights)
  elif num_classes==2 and class_weights is None: 
    loss_function = torch.nn.BCEWithLogitsLoss() 
  elif num_classes>2 and class_weights is not None: 
    weights = torch.tensor(class_weights).to(device) 
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
  for epoch in range(epochs):
      print(f"Epoch : {epoch+1}")
      model.train()
      epoch_loss = 0
      epoch_acc = 0
      for steps, batch in tqdm(enumerate(train_loader, 0)):
          # Move the data to the GPU
          #print(batch['text'])
          input_ids = batch['input_ids'].to(device, dtype=torch.long)
          attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
          labels = batch['label'].to(device, dtype=torch.long)

          # Zero out the gradients
          optimizer.zero_grad()
          #print(outputs.logits)
          #print(labels)
          # Forward pass
          outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
          #print(outputs)
          #print(outputs.logits.dtype)
          # Compute the loss
          predicted_labels = torch.argmax(outputs.logits, dim=1)
          #print(predicted_labels)
          #print(labels)
          #print(loss_function(predicted_labels, labels).dtype)
          one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
          #print(one_hot_labels.dtype)
          loss = loss_function(outputs.logits, one_hot_labels.to(device, torch.float32))
          #print(loss)
          epoch_loss +=loss.detach() 
          _, max_indices = torch.max(outputs.logits, dim=1) 
          bath_acc = (max_indices==labels).sum().item()/labels.size(0)
          epoch_acc += bath_acc
          #loss.requires_grad = True
          # Backward pass
          loss.backward()

          # Update the weights
          optimizer.step()
      print(f"Train Loss: {epoch_loss/steps}")
      print(f"Train Accuracy: {epoch_acc/steps}")

  if save_model==True and not local_model: 
    torch.save(model, path_to_save)
  else: 
    model.save_pretrained(path_to_save)

  return model 

def plot_confusion_matrix(cm): 
  sns.heatmap(cm, annot=True, cmap='Blues')

  # add labels to the plot
  plt.xlabel('Predicted labels')
  plt.ylabel('True labels')
  plt.title('Confusion Matrix')
  plt.show()

 



def evaluation(y_test, y_pred): 

  print("Classification Report: ")

  print("Accuracy: " ,accuracy_score(y_test, y_pred)) 
  print("Precision: ", precision_score(y_test, y_pred)) 
  print("Recall: ", recall_score(y_test, y_pred)) 
  print("F1 Score: ", f1_score(y_test, y_pred)) 
  print(classification_report(y_test, y_pred, digits=3))
  cm = confusion_matrix(y_test,y_pred)
  print("Confusion Matrix:" , cm)
  #plot_confusion_matrix(cm)
  


def eval_model(model_path, test_loader, complete_evaluation=False, local_model=False, return_logits=False): 
  if not local_model: 
    model = torch.load(model_path)
  else: 
    model = AutoModelClass.from_pretrained(model_path)
  model.to(device)
  model.eval()
  num_correct = 0
  num_total = 0
  preds, true_labels = [], []
  
  logits_output = None 
  if return_logits == True: 
    logits_output = [] 

  for steps, batch in tqdm(enumerate(test_loader,0))  :
          input_ids = batch['input_ids'].to(device, dtype = torch.long)
          attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
          labels = batch['label'].to(device, dtype = torch.long)

          outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
          logits = outputs.logits 
          if return_logits == True: 
            logits_output.append(logits) 

          predictions = torch.argmax(logits, dim=-1)
          preds+=predictions.tolist()
          true_labels+=labels.tolist()
         
          num_correct += torch.sum(predictions == labels)
          num_total += len(labels)

  accuracy = float(num_correct) / num_total
  print(f" accuracy of the model:{accuracy}")

  if complete_evaluation and not return_logits: 
    evaluation(true_labels, preds) 
    return (true_labels, preds)
  
  else: 
    return logits_output 
  

def calculate_weights(labels): 
  class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), 
                y=labels)
  return [1.0, 9.0] 

def fusion_eval(MODEL_PATH_1, MODEL_PATH_2, ): 
  None 

def find_false_neg(input_data_path, true_labels, pred_labels): 
  dataset = pd.read_csv(input_data_path) 
  note_index = [] 
  for i in range(len(true_labels)): 
    if true_labels[i] == 1 and pred_labels == 0: 
      note_index.append(i) 
  print(note_index)
  print("False Negatives are:........" )
  for i in note_index: 
    print(dataset['text'][i]) 


def driver(MODEL_PATH, input_file_path, BATCH_SIZE=8, 
           epochs = 3, text_attr = 'text', 
           label_attr='label', 
           mode='train', 
           model_name="MilaNLProc/bert-base-uncased-ear-misogyny",
           model_lr=1e-05, save_model=True, 
           complete_eval=True, 
           weights=False, 
           local_model=False, 
           fusion=True):
  if mode == 'train': 
    print("Loading Train Data: ...............")
    train_dataset = MisogynyDataset(input_file_path, text_attr, label_attr)
    class_weights = None 

    if weights == True: 
      data = pd.read_csv(input_file_path)[label_attr]
      class_weights = calculate_weights(data) 
      data = None 


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Training model: ............ ")
    train_model(train_loader, 
                epochs=epochs, 
                model_name=model_name, 
                path_to_save=MODEL_PATH, 
                num_classes=2, 
                model_lr=model_lr, 
                save_model=save_model,
                class_weights = class_weights,
                local_model = local_model
                )
  if mode == 'test': 
    print("Loading Test Data: ...............")
    test_dataset = MisogynyDataset(input_file_path, text_attr, label_attr)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print("Testing Model: ....................")
    true_labels, preds_labels = eval_model(MODEL_PATH, test_loader, complete_evaluation=complete_eval, local_model=local_model)
    #cm = confusion_matrix(true_labels, preds_labels) 
    #plot_confusion_matrix(cm)
    #find_false_neg(input_file_path, true_labels, preds_labels) 
    print(true_labels)
    print(preds_labels)

if __name__=='__main__': 
  arg= argparse.ArgumentParser() 
  arg.add_argument('--model_path', type=str, required=True)
  arg.add_argument('--model_name', type=str, default="MilaNLProc/bert-base-uncased-ear-misogyny") 
  arg.add_argument('--input_file_path', type=str, required=True) 
  arg.add_argument('--batch_size', type=int, default=8) 
  arg.add_argument('--epochs', type=int, default=3) 
  arg.add_argument('--text_attr', type=str, default='text') 
  arg.add_argument('--label_attr', type=str, default='label')
  arg.add_argument('--mode', type=str, required=True) 
  arg.add_argument('--weights', type=bool, default=False)
  arg.add_argument('--learning_rate', type=float, default=1e-05)
  arg.add_argument('--save_model', type=bool, default=True) 
  arg.add_argument('--complete_eval', type=bool, default=True) 
  arg.add_argument('--local_model', type=bool, default=False) 

  args = arg.parse_args()

  driver(args.model_path, 
         args.input_file_path, 
         BATCH_SIZE = args.batch_size, 
         epochs = args.epochs, 
         text_attr = args.text_attr, 
         label_attr = args.label_attr, 
         mode = args.mode, 
         model_name = args.model_name, 
         model_lr = args.learning_rate, 
         save_model = args.save_model, 
         complete_eval = args.complete_eval, 
         weights = args.weights,
         local_model = args.local_model 
         
         )
  













