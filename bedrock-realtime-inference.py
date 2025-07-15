#load libraries
import boto3
import ast
from anthropic import Anthropic
from botocore.config import Config
import os
import time
import json
import base64
import io
from io import StringIO
import re
from botocore.exceptions import ClientError
import fitz
from datetime import datetime    
import argparse
import streamlit as st
from boto3.dynamodb.conditions import Key 

# # parse the input arguments
# parser = argparse.ArgumentParser(description='Script so useful.')
# parser.add_argument("--opt1")
# parser.add_argument("--opt2")


# args = parser.parse_args()

# model = args.opt1 #['nova-pro','nova-lite','nova-micro','claude-3-5-sonnet','claude-3-sonnet','claude-3-haiku','claude-instant-v1','claude-v2:1', 'claude-v2']
# pdf_filename = args.opt2 # pdf file path


config = Config(
    read_timeout=600, # Read timeout parameter
    retries = dict(
        max_attempts = 10 ## Handle retries
    )
)

# st.set_page_config(initial_sidebar_state="auto")
# Read credentials
with open('config.json','r',encoding='utf-8') as f:
    config_file = json.load(f)
    
BUCKET=config_file["Bucket_Name"]
OUTPUT_TOKEN=config_file["max-output-token"]
REGION=config_file["bedrock-region"]
DYNAMODB_TABLE=config_file["DynamodbTable"]
DYNAMODB  = boto3.resource('dynamodb')
LOAD_DOC_IN_ALL_CHAT_CONVO=config_file["load-doc-in-chat-history"]
CHAT_HISTORY_LENGTH=config_file["chat-history-loaded-length"]

# pricing info
with open('pricing.json','r',encoding='utf-8') as f:
    pricing_file = json.load(f)

#define the clients
S3 = boto3.client('s3')
bedrock_runtime = boto3.client(service_name='bedrock-runtime',region_name=REGION,config=config)

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'input_token' not in st.session_state:
    st.session_state['input_token'] = 0
if 'output_token' not in st.session_state:
    st.session_state['output_token'] = 0
if 'chat_hist' not in st.session_state:
    st.session_state['chat_hist'] = []
if 'user_sess' not in st.session_state:
    st.session_state['user_sess'] =str(time.time())
if 'chat_session_list' not in st.session_state:
    st.session_state['chat_session_list'] = []
if 'count' not in st.session_state:
    st.session_state['count'] = 0
if 'userid' not in st.session_state:
    st.session_state['userid']= config_file["UserId"]
if 'cost' not in st.session_state:
    st.session_state['cost'] = 0
if 'time' not in st.session_state:
    st.session_state['time'] = 0


def get_session_ids_by_user(table_name, user_id):
    # """
    # Get Session Ids and corresponding top message for a user to populate the chat history drop down on the front end
    # """
    if DYNAMODB_TABLE:
        table = DYNAMODB.Table(table_name)
        message_list={}
        session_ids = []
        args = {
            'KeyConditionExpression': Key('UserId').eq(user_id)
        }
        while True:
            response = table.query(**args)
            session_ids.extend([item['SessionId'] for item in response['Items']])
            if 'LastEvaluatedKey' not in response:
                break
            args['ExclusiveStartKey'] = response['LastEvaluatedKey']

        for session_id in session_ids:
            try:
                message_list[session_id]=DYNAMODB.Table(table_name).get_item(Key={"UserId": user_id, "SessionId":session_id})['Item']['messages'][0]['user']
            except Exception as e:
                print(e)
                pass
    else:
        try:
            message_list={}
            # Read the existing JSON data from the file
            with open(LOCAL_CHAT_FILE_NAME, "r", encoding='utf-8') as file:
                existing_data = json.load(file)
            for session_id in existing_data:
                message_list[session_id]=existing_data[session_id][0]['user']

        except FileNotFoundError:
            # If the file doesn't exist, initialize an empty list
            message_list = {}
    return message_list

def put_db(params,messages):
    """Store long term chat history in DynamoDB"""    
    chat_item = {
        "UserId": st.session_state['userid'], # user id
        "SessionId": params["session_id"], # User session id
        "messages": [messages],  # 'messages' is a list of dictionaries
        "time":messages['time']
    }

    existing_item = DYNAMODB.Table(DYNAMODB_TABLE).get_item(Key={"UserId": st.session_state['userid'], "SessionId":params["session_id"]})
    if "Item" in existing_item:
        existing_messages = existing_item["Item"]["messages"]
        chat_item["messages"] = existing_messages + [messages]
    response = DYNAMODB.Table(DYNAMODB_TABLE).put_item(
        Item=chat_item
    )

def get_chat_history_db(params,cutoff,claude3):
    current_chat, chat_hist=[],[]
    if params['chat_histories']: 
        chat_hist=params['chat_histories'][-cutoff:]              
        for d in chat_hist:
            if d['image'] and claude3 and LOAD_DOC_IN_ALL_CHAT_CONVO:
                content=[]
                for img in d['image']:
                    s3 = boto3.client('s3')
                    match = re.match("s3://(.+?)/(.+)", img)
                    image_name=os.path.basename(img)
                    _,ext=os.path.splitext(image_name)
                    if "jpg" in ext: ext=".jpeg"                        
                    if match:
                        bucket_name = match.group(1)
                        key = match.group(2)    
                        obj = s3.get_object(Bucket=bucket_name, Key=key)
                        base_64_encoded_data = base64.b64encode(obj['Body'].read())
                        base64_string = base_64_encoded_data.decode('utf-8')                        
                    content.extend([{"type":"text","text":image_name},{
                      "type": "image",
                      "source": {
                        "type": "base64",
                        "media_type": f"image/{ext.lower().replace('.','')}",
                        "data": base64_string
                      }
                    }])
                content.extend([{"type":"text","text":d['user']}])
                current_chat.append({'role': 'user', 'content': content})
            elif d['document'] and LOAD_DOC_IN_ALL_CHAT_CONVO:
                doc='Here are the documents:\n'
                for docs in d['document']:
                    uploads=handle_doc_upload_or_s3(docs)
                    doc_name=os.path.basename(docs)
                    doc+=f"<{doc_name}>\n{uploads}\n</{doc_name}>\n"
                if not claude3 and d["image"]:
                    for docs in d['image']:
                        uploads=handle_doc_upload_or_s3(docs)
                        doc_name=os.path.basename(docs)
                        doc+=f"<{doc_name}>\n{uploads}\n</{doc_name}>\n"
                current_chat.append({'role': 'user', 'content': [{"type":"text","text":doc+d['user']}]})
            else:
                current_chat.append({'role': 'user', 'content': [{"type":"text","text":d['user']}]})
            current_chat.append({'role': 'assistant', 'content': d['assistant']})  
    else:
        chat_hist=[]
    return current_chat, chat_hist
    
# function to read model output 
def bedrock_streemer(model_id,response, cost, handler):
    stream = response.get('body')
    answer = ""    
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if  chunk:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                
                if "contentBlockDelta" in chunk_obj: 
                    content_block_delta = chunk_obj.get("contentBlockDelta")

                    if "delta" in content_block_delta:                    
                        delta = content_block_delta['delta']
                        if "text" in delta:
                            text=delta['text'] 
                            # st.write(text, end="")                        
                            answer+=str(text)       
                            # handler.markdown(answer.replace("$","USD ").replace("%", " percent"))

                elif "delta" in chunk_obj:                    
                    delta = chunk_obj['delta']
                    if "text" in delta:
                        text=delta['text'] 
                        # st.write(text, end="")                        
                        answer+=str(text)       
                        # handler.markdown(answer.replace("$","USD ").replace("%", " percent"))
                        
                if "amazon-bedrock-invocationMetrics" in chunk_obj:
                    st.session_state['input_token'] = chunk_obj['amazon-bedrock-invocationMetrics']['inputTokenCount']
                    st.session_state['output_token'] =chunk_obj['amazon-bedrock-invocationMetrics']['outputTokenCount']
                    
                    # if 'claude' in model_id:
                    #     pricing=st.session_state['input_token']*pricing_file[f"anthropic.{model}"]["input"]+ st.session_state['output_token'] *pricing_file[f"anthropic.{model}"]["output"]
                    # elif 'nova' in model_id:
                    #     pricing= st.session_state['input_token'] *pricing_file[f"amazon.{model}"]["input"]+ st.session_state['output_token'] *pricing_file[f"amazon.{model}"]["output"]
                    # cost+=pricing          
    return answer,cost , st.session_state['input_token'], st.session_state['output_token']
    
# nova model invoke function
def bedrock_nova_(system_message, prompt,model_id,image_path,cost, handler=None):
    chat_history = []
    content=[]
    if image_path:       
        if not isinstance(image_path, list):
            image_path=[image_path]      
        for img in image_path:
            s3 = boto3.client('s3')
            match = re.match("s3://(.+?)/(.+)", img)
            image_name=os.path.basename(img)
            _,ext=os.path.splitext(image_name)
            if "jpg" in ext.lower(): ext=".jpeg"                        
            if match:
                bucket_name = match.group(1)
                key = match.group(2)    
                obj = s3.get_object(Bucket=bucket_name, Key=key)
                base_64_encoded_data = base64.b64encode(obj['Body'].read())
                base64_string = base_64_encoded_data.decode('utf-8')
            content.extend([
                # {"text":image_name},
                {
              "image": {
                "format": "jpeg",
                "source": {
                    "bytes": base64_string,
                }
              }
            }])
    content.append({
        # "type": "text",
        "text": prompt
            })
    chat_history.append({"role": "user",
            "content": content})
    # print(system_message)
    inf_params = {"max_new_tokens": 4000,"temperature": 0.2}
    prompt = {
        "inferenceConfig": inf_params,
        "system":[{
            "text": system_message
            }
            ],
        "messages": chat_history
    }
    prompt = json.dumps(prompt)
    # print(prompt)
    print(model_id)
    response = bedrock_runtime.invoke_model_with_response_stream(body=prompt, modelId=model_id, accept="application/json", contentType="application/json")
    # print(response)
    
    answer,cost,input_token, output_token =bedrock_streemer(model_id,response,cost, handler) #,input_token, output_token
    
    return answer,cost,input_token, output_token


#claude model invoke
def bedrock_claude_(system_message, prompt,model_id,image_path, cost, handler=None):
    content=[]
    chat_history = []
    if image_path:       
        if not isinstance(image_path, list):
            image_path=[image_path]      
        for img in image_path:
            s3 = boto3.client('s3')
            match = re.match("s3://(.+?)/(.+)", img)
            image_name=os.path.basename(img)
            _,ext=os.path.splitext(image_name)
            if "jpg" in ext.lower(): ext=".jpeg"                        
            if match:
                bucket_name = match.group(1)
                key = match.group(2)    
                obj = s3.get_object(Bucket=bucket_name, Key=key)
                base_64_encoded_data = base64.b64encode(obj['Body'].read())
                base64_string = base_64_encoded_data.decode('utf-8')
            
            content.extend([{"type":"text","text":image_name},{
              "type": "image",
              "source": {
                "type": "base64",
                "media_type": f"image/{ext.lower().replace('.','')}",
                "data": base64_string
              }
            }])
    content.append({
        "type": "text",
        "text": prompt
            })
    chat_history.append({"role": "user",
            "content": content})
    # print(system_message)
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "temperature": 0.2,
        "system":system_message,
        "messages": chat_history
    }
    
    prompt = json.dumps(prompt)
    # print(prompt)
    # print(model_id)
    response = bedrock_runtime.invoke_model_with_response_stream(body=prompt, modelId=model_id, accept="application/json", contentType="application/json")
    answer,cost,input_token, output_token =bedrock_streemer(model_id,response,cost, handler) 
    return answer, cost,input_token, output_token

# invoke bedrock model with retries, in case of throttling
def _invoke_bedrock_with_retries(chat_template, prompt, model_id, image_path,cost, handler=None):
    max_retries = 10
    backoff_base = 2
    max_backoff = 3  # Maximum backoff time in seconds
    retries = 0

    while True:
        try:
            if 'nova' in model_id:
                response,cost,input_token, output_token = bedrock_nova_(chat_template, prompt, model_id, image_path,cost, handler)
            elif 'claude' in model_id:
                response, cost,input_token, output_token = bedrock_claude_(chat_template, prompt, model_id, image_path,cost, handler)
            return response, cost, input_token, output_token
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            elif e.response['Error']['Code'] == 'ModelStreamErrorException':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            elif e.response['Error']['Code'] == 'EventStreamError':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            else:
                # Some other API error, rethrow
                raise


# combine multiple ojects in json
def combine_json_objects(json_objects):
    combined = {}

    for json_obj in json_objects:
        json_obj = json_obj.replace("```json","").replace("```","")
        json_object = ast.literal_eval(json_obj)

        for section, section_data in json_object.items():
            if section not in combined:
                combined[section] = {}

            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if isinstance(value, dict):
                        if key not in combined[section]:
                            combined[section][key] = {}
                    
                        if len(value.keys()) != 2:
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, dict):
                                    if sub_key not in combined[section][key]:
                                        combined[section][key][sub_key] = {}
                                    if sub_key in combined[section][key] and "confidence" in combined[section][key][sub_key]:
                                        if sub_value["confidence"] > combined[section][key][sub_key]["confidence"]:
                                            combined[section][key][sub_key] = sub_value

                                    else:
                                        combined[section][key][sub_key] = sub_value
                            
                        elif len(value.keys()) == 2: 
                            if key in combined[section] and "confidence" in combined[section][key]:
                                if value["confidence"] > combined[section][key]["confidence"]:
                                    combined[section][key] = value
                            else:
                                combined[section][key] = value
                        
    return combined

# function to write/append to json
def write_to_json(response,file_name, model):
    # Load existing JSON data (or create an empty list if it's a new file)
    file_name = file_name.split('.')[0]
    # datestamp = str(int(datetime.now().timestamp()))
    
    # Write the updated data back to the JSON file
    with open('data/output/' + file_name+'-'+ model + '.json', 'w') as f:
        json.dump(json.loads(response), f, indent=4)


# convert pdf to multiple image files of each page
def convert_pdf_to_images(pdf_path, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    """Converts a PDF file to multiple image files."""
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  
        pix = page.get_pixmap()  
        output_file = f"{output_folder}/page_{page_num + 1}.jpeg"
        pix.save(output_file)
    doc.close()

def convert_pdf_to_markdown(pdf_path):
    doc = fitz.open(pdf_path)
    markdown_text = ""
    for page in doc:
        markdown_text += page.get_text("markdown")
    return markdown_text

# Function takes a user query and a uploaded document and geenrtes a prompt
def query_llm(params):
    """
    Function takes a user query and a uploaded document. Caches documents in S3 is optional
    """  
    if 'nova' in params['model']:
        model_id='us.amazon.'+params['model']+'-v1:0'
    else:
        model_id='anthropic.'+ params['model']
        if "3-5-sonnet" in params['model']:
            model_id+="-20240620-v1:0"
        elif "sonnet" in params['model'] or "haiku" in params['model']:
            model_id+="-20241022-v2:0" if "sonnet" in params['model'] else "-20240307-v1:0"

    print("upload_doc:",params['upload_doc'].name)   
    file_name = params['upload_doc'].name #"CWDoc.pdf"#doc_path[0]  # Replace with your PDF file path
    if 'pdf' in file_name:
        print("I am in pdf")
        context_text = convert_pdf_to_markdown(file_name)
    elif 'xlsx' in file_name:
        context_text = convert_excel_to_markdown(file_name)
        
    with open("prompt/doc_chat.txt","r", encoding="utf-8") as f:
        chat_template=f.read()  

    # entities template 
    with open("prompt/entities_sinclair.txt","r",encoding="utf-8") as f:
        entity_template=f.read()

    inputTokens = 0
    outputTokens = 0
    cost = 0

    prompt = f"""You are an AI assistant specialized in Intelligent Document Processsing (IDP) to classify text (given in the image) and entity extraction. I will provide you with a piece of image which contains the text, and your task is to classify it into one appropriate category and extract relevant entities from the context based on the classification category and output them in a structured JSON format.  
                
        <text>
            {context_text}
        </text>
        
        <!-- This will tell you how to extract the entities depending upon the content class. -->
        <entity_template>
            {entity_template}        
        </entity_template>
        
        <!-- This will tell you how to do your tasks well. -->
        <task_guidance>
                1. Analyze the provided image content, and generate the output as <entity_template></entity_template> format
                2. Do NOT make up answers, Only extract entities based on the content in the image.
                3. Most important if the classification category is "Other" or "Procedural", DO NOT extract any entities. leave the value as empty like, "entitites" : "".
                4. When providing your response:
                    - Do not include any preamble or XML tags before and after the json response output.
                    - It should not be obvious you are referencing the result.
                    - Provide your answer in JSON form. Reply with only the answer in JSON format.
                    - json objects should be key-value pair. Do not include type, value in the json object.
                5. Ensure that each entity has a "confidence" field with a value between 0 and 100, representing the certainty of the extracted entity.
                6. Stricktly do not include any preamble or XML tags before and after the json response output.
        </task_guidance>
   
        <!-- These are your tasks that you must complete -->
        <tasks>
                Profile the content provided to you in the image and double check your work.
        </tasks>"""
        
    print("###################################")
    print(prompt)
    print("###################################")     
    
    # # Retrieve past chat history   
    # current_chat,chat_hist=get_chat_history_db(params, CHAT_HISTORY_LENGTH,claude3)
    image_path = ''
    response, cost,input_token, output_token =_invoke_bedrock_with_retries(chat_template, prompt, model_id, image_path, cost, handler= None)
    
    print(response)
    print("###############################################################")
    pdf_file = params['upload_doc'].name
    write_to_json(response,pdf_file, params['model'])

    # log the following items to dynamodb
    chat_history={"user":params['question'],
    "assistant":response,
    "image":image_path,
    # "document":doc_path,
    "modelID":model_id,
    "time":str(time.time()),
    "input_token":round(st.session_state['input_token']) ,
    "output_token":round(st.session_state['output_token'])} 
    #store convsation memory and user other items in DynamoDB table
    if DYNAMODB_TABLE:
        put_db(params,chat_history)
    # use local memory for storage
    else:
        save_chat_local(LOCAL_CHAT_FILE_NAME,[chat_history], params["session_id"])  

    return response

def get_chat_historie_for_streamlit(params):
    """
    This function retrieves chat history stored in a dynamoDB table partitioned by a userID and sorted by a SessionID
    """
    if DYNAMODB_TABLE:
        chat_histories = DYNAMODB.Table(DYNAMODB_TABLE).get_item(Key={"UserId": st.session_state['userid'], "SessionId":params["session_id"]})
        if "Item" in chat_histories:
            chat_histories=chat_histories['Item']['messages'] 
        else:
            chat_histories=[]
    else:
        chat_histories=load_chat_local(LOCAL_CHAT_FILE_NAME,params["session_id"])         

# Constructing the desired list of dictionaries
    formatted_data = []   
    if chat_histories:
        for entry in chat_histories:           
            image_files=[os.path.basename(x) for x in entry.get('image', [])]
            doc_files=[os.path.basename(x) for x in entry.get('document', [])]
            assistant_attachment = '\n\n'.join(image_files+doc_files)
            
            formatted_data.append({
                "role": "user",
                "content": entry["user"],
            })
            formatted_data.append({
                "role": "assistant",
                "content": entry["assistant"],
                "attachment": assistant_attachment
            })
    else:
        chat_histories=[]            
    return formatted_data,chat_histories


def chat_bedrock_(params):
    st.title('Chatty AI Assitant 🙂')
    params['chat_histories']=[]
    if params["session_id"].strip():
        st.session_state.messages, params['chat_histories']=get_chat_historie_for_streamlit(params)
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):   
            if "```" in message["content"]:
                st.markdown(message["content"],unsafe_allow_html=True )
            else:
                st.markdown(message["content"].replace("$", "\$"),unsafe_allow_html=True )
            if message["role"]=="assistant":
                if message["attachment"]:
                    with st.expander(label="**attachments**"):
                        st.markdown( message["attachment"])
    if prompt := st.chat_input("Whats up?"):  
        start = time.time()
        st.session_state.messages.append({"role": "user", "content": prompt})        
        with st.chat_message("user"):             
            st.markdown(prompt.replace("$","\$"),unsafe_allow_html=True )
        with st.chat_message("assistant"): 
            message_placeholder = st.empty()
            time_now=time.time()            
            params["question"]=prompt
            answer=query_llm(params)#, message_placeholder
            # md = st.list(answer)
            message_placeholder.markdown(answer.replace("$", "\$"),unsafe_allow_html=True ) #(answer),unsafe_allow_html=True )#.replace("$", "\$")
            st.session_state.messages.append({"role": "assistant", "content": str(answer)}) 
            
            st.session_state['time'] =  time.time() - start
            print("Timetaken:", st.session_state['time'])
        st.rerun()

def save_uploaded_file(uploaded_file):
    """Saves the uploaded file to the specified directory."""
    try:
        path = '../sinclair'
        print(uploaded_file.name)
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False
        
def get_key_from_value(dictionary, value):
    return next((key for key, val in dictionary.items() if val == value), None)

def app_sidebar():
    with st.sidebar: 
        uploaded_file = ''
        st.metric(label="Bedrock Session Cost", value=f"${round(st.session_state['cost'],2)}") 
        st.metric(label="Bedrock Session time(Min)", value=f"{round(st.session_state['time']/60,0)}") 
        st.write("-----")
        button=st.button("New Chat", type ="primary")
        models=['nova-pro','nova-lite','nova-micro','claude-3-5-sonnet','claude-3-sonnet','claude-3-haiku','claude-instant-v1','claude-v2:1', 'claude-v2']
        model=st.selectbox('**Model**', models)
        params={"model":model} 
        user_sess_id=get_session_ids_by_user(DYNAMODB_TABLE, st.session_state['userid'])
        float_keys = {float(key): value for key, value in user_sess_id.items()}
        sorted_messages = sorted(float_keys.items(), reverse=True)      
        sorted_messages.insert(0, (float(st.session_state['user_sess']),"New Chat"))        
        if button:
            st.session_state['user_sess'] = str(time.time())
            sorted_messages.insert(0, (float(st.session_state['user_sess']),"New Chat"))      
        st.session_state['chat_session_list'] = dict(sorted_messages)
        chat_items=st.selectbox("**Chat Sessions**",st.session_state['chat_session_list'].values(),key="chat_sessions")
        session_id=get_key_from_value(st.session_state['chat_session_list'], chat_items)
        uploaded_file = st.file_uploader('Upload a document', accept_multiple_files=False, help="pdf,csv,txt,png,jpg,xlsx,json,py doc format supported") 
        if uploaded_file: #is not None
            print(uploaded_file)
            print("uploaded file is not none")
            file_upload_status = save_uploaded_file(uploaded_file)
            if file_upload_status:
                st.success(f"File '{uploaded_file.name}' saved successfully!")
            # else:
            #     st.error(f"Failed to save '{uploaded_file.name}'.")
            
        if uploaded_file and LOAD_DOC_IN_ALL_CHAT_CONVO:
            st.warning('You have set **load-doc-in-chat-history** to true. For better performance, remove upload files (by clicking **X**) **AFTER** first query **RESPONSE** on uploaded files. See the README for more info', icon="⚠️")
        params={"model":model, "session_id":str(session_id), "chat_item":chat_items, "upload_doc":uploaded_file }    
        st.session_state['count']=1

        # st.write("-----")
        # with st.expander("Architecture"):
        #     st.write("In this demo, we demonstrate how to Classify and Extract entities from documents using Amazon Bedrock. We use the following architecture:")            
        #     st.image("images/camp-arch.png")
        return params
def main():
    params=app_sidebar()
    chat_bedrock_(params)

if __name__ == '__main__':
    main()
