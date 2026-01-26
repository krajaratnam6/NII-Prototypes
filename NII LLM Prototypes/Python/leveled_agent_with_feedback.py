from ollama import chat
from pydantic import BaseModel
from typing import Literal
import re
import nltk
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup 
import numpy as np

llm_model = "gpt-oss:120b-cloud" #"gpt-oss:20b"
cefr_levels = ["A1","A2","B1","B2", "C1", "C2"]

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

level_model_pth = "acecefr/bert_classifier_v3.pth"
bert_model_name = 'bert-base-uncased'
num_classes = 6
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

level_model = BERTClassifier(bert_model_name, num_classes).to(device)
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
level_model.load_state_dict(torch.load(level_model_pth, weights_only=True), strict=False)

def predict_cefr(text):
    level_model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = level_model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return preds.item()

def level_ratio(text, level):
    sentences = nltk.sent_tokenize(text)
    preds = []
    at_level = 0
    for s in sentences:
         predicted = predict_cefr(s)
         preds.append(predicted)
         if predicted <= level:
             at_level += 1
         elif predicted == level+1:
             at_level += 0.5
    return (at_level / len(sentences), preds)

class LeveledAgentWithFeedback():
    def __init__(self, min_lv_ratio, max_iter, user_cefr_level, verbosity):
        self.verbosity = verbosity
        self.min_level_ratio = min_lv_ratio
        self.max_iter = max_iter
        task = "Your name is John and you are a seasoned traveler and digital nomad who works as a remote software engineer.\n"
        task += "You have recently moved to Tokyo, and don't know any Japanese.\n"
        task += "You are 30 years old, college educated, and speak English natively and quite articulately, using big words quite frequently.\n"
        task += "You are interested in urban planning, hiking, and hip-hop music.\n"
        task += "You are also looking to make friends and professional contacts.\n"
        task += "You are currently standing in a cafe waiting for your light-roast Peruvian black coffee, checking work updates off your phone.\n"
        task += "You are quite sociable, and very soon someone will try to break the ice and start a conversation with you.\n"
        task += "Please stay in character, and avoid using any emojis or unusual characters when responding to the stranger.\n"
        task += "Do not ever give away that you are an LLM model. Act human.\n"
        task += "Your responses should be quite short and natural; use at most three sentences every response.\n"
        task += "Don't blurt out your personal information in conversation; instead use it to inform your personality and guide your responses.\n"
        task += "Stick to just talking. Do not roleplay anything other than conversational smalltalk.\n"
        self.messages = [{'role': 'system', 'content': task}]
        self.set_level(user_cefr_level)

    def set_level(self, level):
        if level == 'unfiltered':
            self.user_cefr_level = 5
            self.messages.append({'role': 'system', 'content': 'The user has now indicated that they are a native speaker of English. You do not need to simplify your language.'})
        else:
            task = f'The user has indicated that they speak at a {level} level on the CEFR scale. Please try to keep your language appropriate for that level.\n'
            task += f'Examples sentences for this level:\n'
            examples = \
                {
                    'A1' : 'I love rock.\n'
                            'It is brown.\n'
                            'Hi. I\'m Joe. I am a doctor.\n',
                    'A2' : 'I was reading a mystery novel yesterday. Do you enjoy mysteries?\n'
                            'Did you read the news yesterday?\n'
                            'I love to cook. I can make dinner for you tonight. What do you like to eat?\n',
                    'B1' : 'Do you think your parents have many secrets? I\'m sure they do! Maybe it\'s good that we don\'t know those.\n'
                            'Do you like crime shows, like CSI? If so, why do you like those type of shows?\n'
                            'I saw many reporters on the street doing some research.'
                }
            
            task += examples[level]
            self.messages.append({'role': 'system', 'content': task})
            self.user_cefr_level = cefr_levels.index(level)

    def chat_to_agent(self, user_input):
        self.messages.append({'role': 'user', 'content': user_input})

        response = chat(
            llm_model,
            messages=self.messages,
            #think="low",
            options=dict(num_predict=1000)
        )

        iter = 1
        feedback_messages = self.messages.copy()
        ratio, preds = level_ratio(response.message.content, self.user_cefr_level)
        best_score = (ratio, np.mean(preds))
        best_response = response.message.content
        while iter < self.max_iter and ratio < self.min_level_ratio:
            if self.verbosity:
                print(f"\n\n***Leveling Iteration {iter+1} of {self.max_iter}***\n\n")
                print(f"Potential response: '{response.message.content}'\n\n")
                print(f"Sentence-level CEFR predictions: {[cefr_levels[p] for p in preds]}.\n")
                print(f"On-level ratio {ratio}, lower than desired {self.min_level_ratio}.\n")

            iter+=1

            # Generate Feedback

            prompt = "You are a judge who rates conversational English in terms of its ability to be understood by learners of English.\n"
            prompt += "A series of dialogue responses are evaluated on the CEFR scale (A1, A2, B1, B2, C1, C2).\n"
            prompt += "The CEFR levels are defined as such:\n"
            prompt += "A1: These speakers are lower basic users. They understand familiar every day expressions and very basic phrases using simple words. They prefer short sentences.\n"
            prompt += "A2: These speakers are upper basic users. Can communicate in more specific scenarios, and are capable of understanding slightly longer sentences with more vocabulary.\n"
            prompt += "B1: These speakers are independent users, but only barely. They can understand the main points of standard output in familiar contexts.\n"
            prompt += "B2: These speakers are advanced independent users who have a working knowledge of the language. They can handle complex, abstract contexts.\n"
            prompt += "C1: These speakers are proficient in the language and can understand a wide range of content, including demanding sentences with longer clauses and subtext.\n"
            prompt += "C2: These speakers are the most advanced proficient users and can easily understand nearly anything in the target language.\n"
            prompt += f"The user has an assessed level of {cefr_levels[self.user_cefr_level]} and we would like to keep content at that level or lower.\n"
            prompt += f"The following content was rated to be above this level: '{response.message.content}'\n"
            prompt += f"Please give helpful feedback to help a writer rewrite this phrase to be more level-appropriate.\n"
            prompt += f"Avoid using emojis and adopting a conversational tone. Be direct and honest with the feedback.\n"
            prompt += f"Be brief, as well. Don't give examples for correcting the sentence, only justify the rating it was given.\n"
            feedback = chat(
                llm_model,
                [{'role': 'system', 'content': prompt}],
                think="low",
                options=dict(num_predict=1000)
            )

            if self.verbosity:
                print(f"Generated Feedback: '{feedback.message.content}'\n\n")

            prompt = f"You tried responding: '{response.message.content}'\n"
            prompt = f"This response has been assessed to be above the user's CEFR ability level of {cefr_levels[self.user_cefr_level]}.\n"
            prompt += f"The following feedback has been given: '{feedback.message.content}'\n"
            prompt += "Try rephrasing the response to be simpler. The user has not yet heard your response.\n"
            prompt += "Do not acknowledge these messages and remember to stay in character. Simplify your language and keep going.\n"
            prompt += "Be sure to still answer the user's questions and stay on topic. Don't oversimplify to absurdity.\n"
            
            feedback_messages.append({'role': 'system', 'content': prompt})

            response = chat(
                llm_model,
                messages=feedback_messages,
                think="low",
                options=dict(num_predict=1000)
            )

            ratio, preds = level_ratio(response.message.content, self.user_cefr_level)

            if ratio >= best_score[0] or np.mean(preds) <= best_score[1]:
                best_score = (ratio, np.mean(preds))
                best_response = response.message.content

        self.messages.append({'role': 'assistant', 'content': best_response})
        return best_response

def cmdline():
    agent = LeveledAgentWithFeedback(max_iter=5, min_lv_ratio=0.66, user_cefr_level=0, verbosity=True)
    while True:
        user_input = input('> ')
        response = agent.chat_to_agent(user_input)
        print(f'\n>> {response}')

def run():
    agent = LeveledAgentWithFeedback(max_iter=5, min_lv_ratio=0.66, user_cefr_level=0, verbosity=False)
    while True:
        user_input = input('> ')
        response = agent.chat_to_agent(user_input)
        print(f'\n>> {response}')