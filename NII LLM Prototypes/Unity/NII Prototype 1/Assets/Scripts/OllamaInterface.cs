using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using OllamaSharp;
public class OllamaInterface : MonoBehaviour
{
    OllamaApiClient ollamaApiClient;
    Chat chat;

    [TextArea(15, 20)]
    public string prompt;
    public string model = "gpt-oss:20b";
    public string uri = "http://localhost:11434";

    public Text output;

    // Start is called before the first frame update
    void Start()
    {
        ollamaApiClient = new OllamaApiClient(uri, model);
        chat = new Chat(ollamaApiClient, prompt);
    }

    void Reinit()
    {
        ollamaApiClient.Dispose();
        ollamaApiClient = new OllamaApiClient(uri, model);
        chat = new Chat(ollamaApiClient, prompt);
    }

    public async void Chat(string input)
    {
        output.text = "";
        await foreach (var answerToken in chat.SendAsync(input))
            output.text += answerToken;
    }
}
