import os
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
import urllib.parse
import re
import numpy as np
import string
from tqdm import tqdm
import pandas as pd

def initialize_llm(cache_dir = "/scratch/user/syedwali/Python_env/myGNN/LLM/llama/"):

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    print(f"Using device: {device}")  
    tokenizer = LlamaTokenizer.from_pretrained(cache_dir, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(cache_dir, local_files_only=True).to(device)
    print("LLM Initialized Successfully......")
    return model,tokenizer


def initialize_finetuned_llm(base_model_id = "/scratch/user/syedwali/Python_env/myGNN/LLM/llama",
                            fine_tuned_directory=  "/scratch/user/syedwali/llama-train-finetune/"):


    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  # Llama , same as before
        quantization_config=bnb_config,  # Same quantization config as before
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, local_files_only=True)
    ft_model = PeftModel.from_pretrained(base_model,fine_tuned_directory)
    inputs = tokenizer("remember your job is to generate explanation for predicted outcomes by GNN", return_tensors="pt").to('cuda')
    inputs.pop('token_type_ids', None)
    ft_model.eval()
    ft_model.to(device)
    # Generate the response
    with torch.no_grad():
        _ = ft_model.generate(inputs["input_ids"], max_length=4096,early_stopping=True)

    print("LLM Initialized Successfully......")
    return ft_model, tokenizer

def test_with_explanation(loader, model,llm_model, tokenizer, fine_tune= False,device='cuda'):
    from tqdm import tqdm
    model.eval()
    correct = 0
    num_graphs = 0
    all_preds = []
    all_labels = []
    explanations=[]
    #i=0
    for batch in (tqdm(loader)):
        batch.to('cuda')
        with torch.no_grad():
            pred = model(batch.x_dict,batch.edge_index_dict,batch).max(dim=1)[1]
            label = batch.y

        # Calculate Integrated Gradients of the Predicted Outcome
        flow_imp,packet_imp,flow_input,packet_input=integrated_explainer(model,batch)
        # Explain predicted outcome using LLM
        
        predicted_class=label.item();
        #print("Processing Data Instance: ", i)
        response=generate_explanation(llm_model, tokenizer, predicted_class, flow_imp,packet_imp, flow_input, packet_input,fine_tune,top_features=5,prompt_lengt=1500,stopping_criteria=True)
        
        all_preds.append(pred.cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())
        explanations.append(response)
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs
        #i+=1
        # if predicted_class==7:
        #     return correct / num_graphs, all_preds, all_labels, explanations
        #print('explain')
        #explain_prediction(model, batch, device)
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels)

    return correct / num_graphs, all_preds, all_labels, explanations



def integrated_explainer(model,batch,device='cuda'):
    
    batch.to(device)
    flow_imp=[];packet_imp=[];
    flow_input=[];packet_input=[];
    explainer = Explainer(
              model=model,
              algorithm=CaptumExplainer('IntegratedGradients'),
              explanation_type='model',
              node_mask_type='attributes',
              edge_mask_type='object',
              model_config=dict(
                  mode='multiclass_classification',
                  task_level='graph',
                  return_type='probs',  # Model returns probabilities.
              ),
              )
    hetero_explanation = explainer(
              batch.x_dict,
              batch.edge_index_dict,
              batch=batch
              )
    for node_type, tensor in hetero_explanation.node_mask_dict.items():
          tensor_list = tensor.tolist()
          if node_type == 'flow':
              flow_imp.extend(tensor_list)
          elif node_type == 'packet':
              packet_imp.extend(tensor_list)
    
    for node_type, tensor in hetero_explanation.x_dict.items():
          tensor_list = tensor.tolist()
          if node_type == 'flow':
              flow_input.extend(tensor_list)
          elif node_type == 'packet':
              packet_input.extend(tensor_list)
    return flow_imp,packet_imp,flow_input,packet_input



def generate_explanation(model, tokenizer, predicted_class, flow_imp,packet_imp, flow_input, packet_input,fine_tune=False,
    top_features=5,prompt_lengt=1500,stopping_criteria=True):
    query,payload_analysis,top_three_payloads=prompt_generator(predicted_class, flow_imp,packet_imp, flow_input, packet_input,top_features)
    if payload_analysis:
        response1=(analyze_payload(top_three_payloads,model,tokenizer,fine_tune))
        response2=llm_explanation_without_payload(model,tokenizer,query,prompt_lengt,stopping_criteria,fine_tune)
        response=response1 + "\n" + response2
        
    else:
        response=llm_explanation_without_payload(model,tokenizer,query,prompt_lengt,stopping_criteria,fine_tune)
    return response





def prompt_generator(predicted_class, flow_imp,packet_imp, flow_input, packet_input,top_features=5,
                    features_info='/scratch/user/syedwali/Python_env/myGNN/8class_data_iot/features_description.csv'):
        df = pd.read_csv(features_info)
        feature_names = df.iloc[:, 0].tolist()
        payload_analysis=False;top_three_payloads=False;
        scaled_flow=(np.array(flow_imp)[0]/np.array(flow_imp)[0].max())
        sorted_indices = np.argsort(scaled_flow)
        # Reverse the indices for descending order
        flow_value=np.array(flow_input)[0]
        sorted_indices = sorted_indices[::-1]
        # Sort the scaled importance and feature names
        sorted_scaled_importance = scaled_flow[sorted_indices]
        sorted_flow_values=flow_value[sorted_indices]
        sorted_feature_names = np.array(feature_names)[sorted_indices]
        if (predicted_class==1):
            result_str = "The predicted attack from Graph Neural Network is Denial of Service attack, Top features contributing in this prediction are \n"
        elif (predicted_class==2) :
            result_str = "The predicted attack from Graph Neural Network is Distributed Denial of Service attack, Top features contributing in this prediction are \n"
        elif (predicted_class==3):
            result_str = "The predicted attack from Graph Neural Network is Mirai attack, Top  features contributing in this prediction are \n"
        elif (predicted_class==4):
            result_str = "The predicted attack from Graph Neural Network is Reconnaisance attack, Top features contributing in this prediction are \n"
        elif (predicted_class==5):
            result_str = "The predicted attack from Graph Neural Network is Spoofing attack, Top features contributing in this prediction are \n"
        elif (predicted_class==6):
            result_str = "The predicted attack from Graph Neural Network is Bruteforce attack, Top features contributing in this prediction are \n"
        elif (predicted_class==7):
            result_str = "The predicted attack from Graph Neural Network is Web-based attack, Top features contributing in this prediction are \n"
            payload_analysis=True
            packet_imp=np.array(packet_imp);
            packet_input=np.array(packet_input);
            # Step 1: Calculate the average of each row
            averages = np.mean(packet_imp, axis=1)
            # Step 2: Identify the indices of the top three rows with the highest averages
            top_three_indices = np.argsort(averages)[-3:]
            # Step 3: Extract those rows and concatenate them to form a new vector of shape (1, 1500*3)
            top_three_payloads = packet_input[top_three_indices].flatten().reshape(1, -1)
            
            #print(analyze_payload(top_three_payloads,model,tokenizer))
            
        elif (predicted_class==0):
            result_str = "The prediction from Graph Neural Network is benign, Top features contributing in this prediction are \n"
        else:
            raise ValueError("No Predicted Class Value found, terminating explanation and inference")
            return "Error"
            
    
        for i in range(top_features):
            result_str += f"Feature: {sorted_feature_names[i]},  Actual Value: {sorted_flow_values[i]:.3f} \n"
        if (predicted_class!=0):
            result_str +="Dont expect any threshold value on your own, Now You explain prediction, and each feature relevance to the predicted attack, for this type of attack and also what can we do? Your answer must start with these words The predicted attack is."
        else:
            result_str +="Dont expect any threshold value on your own, Now You explain prediction briefly, and each feature relevance to the predicted benign. Your answer must start with these words The predicted Class is ."
            
        return result_str, payload_analysis, top_three_payloads




def llm_explanation_without_payload(model,tokenizer,query,prompt_lengt=1500,stopping_criteria=True,fine_tune=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(query, return_tensors="pt").to(device)
    if fine_tune:
        inputs.pop('token_type_ids', None); model.eval();
    model.to('cuda')
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=prompt_lengt,early_stopping=stopping_criteria,num_beams = 5)
    
    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(query):].strip()
    # start_phrase = "The predicted attack from Graph Neural Network is"
    # # Find the start of the response
    # start_index = response.find(start_phrase)
    # if start_index != -1:
    #     response = response[start_index:]
    #print(f"Response: {response}")
    return response


def analyze_payload(payload_decimal,model,tokenizer,fine_tune=False):
    
    # Convert each decimal number to its corresponding hexadecimal value and concatenate
    cleaned_str=clean_payload(payload_decimal)
    
    #print(cleaned_str)
    #print("*"*50)
    query= "System Prompt: Is this payload malicious/ web-based attack or normal and why.The payload string is. " +cleaned_str + ".Provide one sentence answer. Start your answer with The payload is. " 
    inputs = tokenizer(query, return_tensors="pt").to('cuda')
    if fine_tune:
        inputs.pop('token_type_ids', None)
        model.eval()
    model.to('cuda')
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=4096,early_stopping=True,num_beams = 5)
    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(query):].strip()
    # start_phrase = "The payload is"
    # # Find the start of the response
    # start_index = response.find(start_phrase)
    # if start_index != -1:
    #     response = response[start_index:]
    #print(response)
    return response




def clean_payload(payload_decimal):
    # Convert each decimal number to its corresponding hexadecimal value and concatenate
    hex_payload = ''.join(format(byte, '02x') for byte in np.array(payload_decimal[0]).astype(int))
    
    # Convert the hexadecimal string to an ASCII string
    ascii_str = ''.join([chr(int(hex_payload[i:i+2], 16)) for i in range(0, len(hex_payload), 2)])
    
    # Decode any URL encoded parts
    filtered_str = urllib.parse.unquote(ascii_str)
    
    # Remove non-ASCII characters
    filtered_str = re.sub(r'[^\x00-\x7F]+', ' ', filtered_str)
    
    # Extract only words, characters, or known ASCII symbols
    filtered_str = re.sub(r'[^a-zA-Z0-9\s' + re.escape(string.punctuation) + ']+', '', filtered_str)
    
    # Clean up excessive whitespace and newlines
    cleaned_str = re.sub(r'[ \t]+', ' ', filtered_str)  # Replace multiple spaces or tabs with a single space
    cleaned_str = re.sub(r'\n+', '\n', cleaned_str)  # Replace multiple newlines with a single newline
    cleaned_str = re.sub(r' *\n *', '\n', cleaned_str)  # Remove spaces before and after newlines
    cleaned_str = cleaned_str.strip()  # Strip leading and trailing whitespace
    
    return cleaned_str



