import socket,os
import threading
import leveled_agent_with_feedback as lv
import time
import unicodedata
import multiprocessing

if __name__ == "__main__":
    global sock, port, threads, buffer_size, cefr_levels, phase_times, intro_level, user_began_chatting

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 12345  
    sock.bind(('0.0.0.0', port))  
    sock.listen(5)

    threads = []
    buffer_size = 32768
    cefr_levels = ["A1","A2","B1","B2", "C1", "C2"]

    phase_times = [10, 11, 12]
    intro_level = 'A1'

    user_began_chatting = False

def recv_msg(connection, agent, stop_event):
    global user_began_chatting
    while True:
        buf = connection.recv(buffer_size).decode('utf-8').replace('\n',' ').replace('\r','')
        print(f"Received Payload: '{buf}'")
        if (buf[0:3] == 'END'):
            print(f"Closing connection from {address}")
            connection.close()
            agent.ended = True
            break
        msg = buf[5:-1]
        print(f"Chatting to agent: '{buf}'")
        response = agent.chat_to_agent(msg)
        print(f"Received response: '{response}'")
        if stop_event.is_set():
            break
        response = unicodedata.normalize('NFKD', response).encode('ascii', 'ignore').decode('ascii')
        connection.send((response + "\n").encode('utf-8'))
        user_began_chatting = True

def serve(connection, address, agent):
    global user_began_chatting

    print(f"Connection received from {address}")
    cefr_level = -1
    phase_levels = []
    timer = 0

    buf = connection.recv(buffer_size).decode('utf-8').replace('\n',' ').replace('\r','')
    if (buf[0:3] == 'END'):
        print(f"Closing connection from {address}")
        connection.close()
        return
    msg = buf[5:-1]
    # TODO: response handling should be in different thread, user can spam or end during level iteration
    response = ""
    if cefr_level < 0:
        try:
            words = msg.split()
            cefr_level = cefr_levels.index(words[0])
            permutation = int(words[1])
            if permutation == 0:
                phase_levels = [intro_level, words[0], 'unfiltered']
            elif permutation == 1:
                phase_levels = [intro_level, 'unfiltered', words[0]]
            else:
                raise ValueError(f"'{permutation}' is an invalid permutation value.")
        except:
            response = f"'{msg}' is in an invalid format. Try again."
            cefr_level = -1
    connection.send((response + "\n").encode('utf-8'))
    
    for phase in range(len(phase_levels)):
        user_began_chatting = False
        stop_event = threading.Event()
        recv_thread = threading.Thread(target=recv_msg, args=(connection, agent, stop_event))
        recv_thread.start()
        while True:
            if not user_began_chatting:
                continue
            print(f'Beginning of phase {phase}. Sleeping for {phase_times[phase]} seconds.')
            recv_thread.join(timeout=phase_times[phase]) #time.sleep(phase_times[phase])
            break
        stop_event.set()
        print(f'{phase_times[phase]} seconds elapsed. End of phase {phase}.')
        recv_thread.join()
        if agent.ended:
            return
        if (phase == len(phase_levels) - 1):
            # ending transition
            connection.send('Got to go. Sorry. Bye!\n'.encode('utf-8'))
        else:
            connection.send('**BREAK**\n'.encode('utf-8'))
            phase += 1   
            agent.set_level(phase_levels[phase])
            user_began_chatting = False

    connection.close()

def sock_accept():
    while True:    
        print("Initializing new agent.")
        agent = lv.LeveledAgentWithFeedback(max_iter=5, min_lv_ratio=0.66, user_cefr_level=intro_level, verbosity=True, lrfxn=custom_level_ratio)
        agent.chat_to_agent("...")
        custom_level_ratio("testing 1 2 3", 0)
        print("Agent initialized. Ready to accept new connections.")
        connection,address = sock.accept()
        print(f"Connection received from {address}.")
        thread = threading.Thread(target=serve, args=(connection, address, agent))
        threads.append(thread)
        thread.start()
    sock.close()

def custom_level_ratio(text, lvl):
    global level_req
    global level_ansq

    while True:
        # global level_req
        # global level_ansq
        if len(level_req) > 0:
            time.sleep(0.05)
            continue
        break
    
    if True:
        # global level_req
        level_req.insert(0, (text, lvl))

    while True:
        # global level_req
        # global level_ansq
        if len(level_ansq) > 0:
            ret = level_ansq[0]
            level_req.pop(0)
            return ret
        time.sleep(0.05)



import nltk
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup 

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            logits = self.fc(x)
            return logits

def level_ratio_process(level_requests, level_answers):
    level_model_pth = "acecefr/bert_classifier_v3.pth"
    bert_model_name = 'bert-base-uncased'
    num_classes = 6
    max_length = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    level_model = BERTClassifier(bert_model_name, num_classes).to(device)
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    level_model.load_state_dict(torch.load(level_model_pth, weights_only=True), strict=False)
    level_model.eval()

    while True:
        if len(level_requests) == 0:
            time.sleep(0.05)
            continue
        text, level = level_requests[0]
        sentences = nltk.sent_tokenize(text)
        print("split into sentences")
        preds = []
        at_level = 0
        for s in sentences: 
            print(f"predicting sentence: {s}")
            encoding = tokenizer(s, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            with torch.no_grad():
                outputs = level_model(input_ids=input_ids, attention_mask=attention_mask)
                _, prs = torch.max(outputs, dim=1)
            predicted = prs.item()
            print(f"level {predicted}")
            preds.append(predicted)
            if predicted <= level:
                at_level += 1
            elif predicted == level+1:
                at_level += 0.5
        level_answers.insert(0, (at_level / len(sentences), preds))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    global level_ansq
    global level_req
    level_ansq = multiprocessing.Manager().list()
    level_req = multiprocessing.Manager().list()
    thread = threading.Thread(target=sock_accept)
    threads.append(thread)
    thread.start()
    lrp = multiprocessing.Process(target=level_ratio_process, args=(level_req, level_ansq))
    lrp.start()
    thread.join()
