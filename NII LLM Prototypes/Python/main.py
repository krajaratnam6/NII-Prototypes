from ollama import chat
from pydantic import BaseModel
from typing import Literal
import re

class CEFRLevel(BaseModel):
    justification: str
    level: Literal['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

def ToCEFR(level: Literal['A1', 'A2', 'B1', 'B2', 'C1', 'C2']) -> str:
    return level

model = "gpt-oss:120b-cloud" #"gpt-oss:20b"
verbose = True
vocab_paths = {"A1":"vocab/A1.txt", "A2":"vocab/A2.txt", "B1":"vocab/B1.txt", "B2":"vocab/B2.txt"}

learner_model_summary = \
        {
            "goals":"I would like to learn about colors.",
            "interests":"I am interested in sports and science.",
            "CEFR_self":"A1",
            "CEFR_predict":"A1",
            "strengths":"This speaker is good at inferring from context.",
            "weaknesses":"This speaker has a weak vocabulary.",
            "num_conv": 0,
        }

def GetVocab():
    vocab = set()
    with open(vocab_paths[learner_model_summary["CEFR_self"]], 'r') as file:
        for line in file:
            vocab.add(line.strip().lower())
    return vocab

def main():
    vocab = GetVocab()
    ci_ratio = 0.9
    max_iter = 10
    VocabConstrainedAgent(vocab, ci_ratio, max_iter)


def VocabConstrainedAgent(vocab, ci_ratio, max_iter):
    task = "You are tasked with helping an English language-learner practice English in a believable scenario.\n"
    task += "You are acting as a clerk in the produce section of a grocery store. Try to preserve this role as much as possible.\n"
    task += "The self-reported CEFR level of the student is: " + learner_model_summary["CEFR_self"] + "\n"
    task += "The modeled CEFR level of the student is: " + learner_model_summary["CEFR_predict"] + "\n"
    task += "The learner has demonstrated the following strenghts: " + learner_model_summary["strengths"] + "\n"
    task += "The learner has demonstrated the following weaknesses: " + learner_model_summary["weaknesses"] + "\n"
    task += "The learner has been in " + str(learner_model_summary["num_conv"]) + " situated conversations so far."
    if learner_model_summary["num_conv"] > 0:
        task += "The learner's progress so far can be summarized as follows: " + "\n"
        i = 1
        for conv in learner_model_summary["conv"]:
            task += "\tConversation " + i + ": " + conv + "\n"
    messages = [{'role': 'system', 'content': task}]

    opener = chat(
            model,
            messages,
            #think="low",
            options=dict(num_predict=1000)
        )
    
    print(opener.message.content + "\n")

    while True:
        user_input = input('> ')
        response = chat(
            model,
            messages=[*messages, {'role': 'user', 'content': user_input}],
            #think="low",
            options=dict(num_predict=1000)
        )

        messages += [
            {'role': 'user', 'content': user_input},
            {'role': 'assistant', 'content': response.message.content},
        ]

        iter = 1

        ratio, out_vocab = vocab_ratio(vocab, response.message.content)
        while iter < max_iter and ratio < ci_ratio:
            iter+=1
            notice = "This response lies above the user's CEFR level of " + learner_model_summary["CEFR_self"]
            notice += f".\nOnly {ratio*100}% of those words fall within the users vocabulary. Aim for {ci_ratio*100}% or higher"
            notice += f".\nThese words were too advanced: {out_vocab}"
            notice += ".\n Please rephrase the previous response using simpler vocabulary and/or more standard forms. Avoid casual shortenings. Do not acknowledge this message."
            messages += [{'role':'system', 'content':notice}]
            if verbose:
                print(f"Generated response with ci ratio of {ratio}, which is lower than {ci_ratio}. Regenerating, attempt {iter} of {max_iter}...\n")
            response = chat(
                model,
                messages,
                options=dict(num_predict=1000)
            )
            ratio, out_vocab = vocab_ratio(vocab, response.message.content)
        
        print(">> " + response.message.content + '\n')

def vocab_ratio(vocab, content):
    word_list = re.sub(r'[^a-z ]', '', content.lower()).split()
    in_vocab = 0; total_count = 0
    out_vocab = []
    for word in word_list:
        total_count+=1
        if word in vocab:
            in_vocab += 1
        else:
            out_vocab.append(word)
    return (in_vocab / total_count, out_vocab)

def PredictCEFR(messages, self_report):
    request = "Given the previous chat history, map the user's English proficiency onto a CEFR level.\ " \
    "Consider that the user self-reported their level to be " + self_report + ". " +\
    "Just output A1, A2, B1, B2, C1, or C2. No justification or formatting is necessary. Please keep your response within 2 characters."

    print(CEFRLevel.model_json_schema())

    response = chat(
        model, #'deepseek-v3.1:671b-cloud',
        messages=[*messages, {'role': 'system', 'content': request}],
        # format=CEFRLevel.model_json_schema(),
        tools=[ToCEFR],
        options=dict(temperature=0)
    )

    return response.message.content
    #return CEFRLevel.model_validate_json(response.message.content).level

def vocab_check(level):
    return

main()