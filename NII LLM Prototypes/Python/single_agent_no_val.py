from ollama import chat

model = "gpt-oss:120b-cloud" #"gpt-oss:20b"
verbose = True

def main():
    learner_model_summary = \
        {
            "goals":"I would like to be able to go to university in English.",
            "interests":"I am interested in sports and science.",
            "CEFR_self":"B1",
            "CEFR_predict":"B1",
            "strengths":"This speaker is good at understanding complex sentences.",
            "weaknesses":"This speaker has a weak vocabulary.",
            "num_conv": 0,
        }
    
    if verbose:
        print("\n\nLearner Model Summary:\n" + str(learner_model_summary) + "\nGenerating Conversation...")

    prompt = PromptGeneration(learner_model_summary)
    full_prompt = prompt["prompt"] + "\n"
    full_prompt += "You are in a setting represented by this image description: " + prompt["background_desc"] + "\n"
    full_prompt += "The high-level design of this conversation is as follows: " + prompt["general_desc"] + "\n"

    if verbose:
        print("\n\nPrompt Generation Output:\n" + str(prompt) +\
              "\n\nFinal Prompt:\n" + str(prompt) + "\n\n")

    print(">> " + prompt["opener"])
    SingleAgent(full_prompt)

def PromptGeneration(learner_model_summary):
    task = "Loosely design a situated conversation in a grounded location for a learner trying to improve their English. \
    Give the learner a well-defined conversational goal in the scenario.\n"
    task += "The goals of the student are as follows:\n" + learner_model_summary["goals"] + "\n"
    task += "The personal (not necessarily goal-related) interests of the student are as follows:\n" + learner_model_summary["interests"] + "\n"
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
    response = chat(
        model,
        messages=messages,
        options=dict(num_predict=1000)
    )
    
    general_desc = response.message.content
    messages += [
        {'role': 'assistant', 'content': general_desc},
        {'role': 'system', 'content': "Considering the above, write a short description for a background image that can convey the setting of the conversation."}
    ]

    response = chat(
        model,
        messages=messages,
        options=dict(num_predict=1000)
    )

    background_desc = response.message.content
    messages += [
        {'role': 'assistant', 'content': background_desc},
        {'role': 'system', 'content': "Now write a prompt for an LLM agent that will take on the role of the tutor in this situated conversation.\
          Again, consider the goal of the scenario, as well as the learners' long-term goal."}
    ]

    response = chat(
        model,
        messages=messages,
        options=dict(num_predict=1000)
    )

    prompt = response.message.content
    messages += [
        {'role': 'assistant', 'content':prompt},
        {'role': 'system', 'content': "Now write a short opening line from the perspective of this agent, explaining the context of the situation and the diegetic goal. Stay in character."}
    ]
    opener = chat(model, messages=messages, options=dict(num_predict=1000)).message.content
    return {'general_desc': general_desc, 'background_desc':background_desc, 'prompt':prompt, 'opener':opener}

def SingleAgent(prompt):
    messages = [
    {
        'role': 'system',
        'content': prompt,
    },
    ]

    while True:
        user_input = input('> ')
        response = chat(
            model,
            messages=[*messages, {'role': 'user', 'content': user_input}],
            think="low",
            options=dict(num_predict=1000)
        )

        messages += [
            {'role': 'user', 'content': user_input},
            {'role': 'assistant', 'content': response.message.content},
        ]
        print(">> " + response.message.content + '\n')

main()