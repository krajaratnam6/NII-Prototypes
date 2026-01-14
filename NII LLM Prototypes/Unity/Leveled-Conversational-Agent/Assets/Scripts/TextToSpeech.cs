using System;
using System.Collections;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class TextToSpeech : MonoBehaviour
{
    [SerializeField] LLMDialogueManager dialogueManager;
    [SerializeField] private AvatarState avatarState;

    public AudioSource audioSource;

    private string apiKey;
    private string ttsApiUrl = "https://api.openai.com/v1/audio/speech";

    // Available voices for the TTS, there could be more, check OpenAI documentation for the latest list https://platform.openai.com/docs/api-reference/audio/createSpeech
    public enum Voice
    {
        alloy,
        ash,
        coral,
        echo,
        fable,
        onyx,
        nova,
        sage,
        shimmer
    }

    [SerializeField]
    private Voice selectedVoice = Voice.alloy;

    // List of supported emotions, this should match the JSON schema defined in LLMDialogueManager
    public enum Emotion
    {
        neutral,
        happy,
        sad,
        angry,
        surprised,
        confused
    }

    string emotionCache;

    public delegate void AudioPlaybackHandler(bool isPlaying);
    public static event AudioPlaybackHandler OnAudioPlayback;

    void Start()
    {
        apiKey = APIKeys.APIKey;
    }

    public void CreateSpeech(string message, string emotion)
    {
        Debug.Log("CreateSpeech\n");
        Debug.Log(emotion);
        dialogueManager = GetComponent<LLMDialogueManager>();
        Debug.Log(dialogueManager);
        dialogueManager.HandleEmotion(emotion);
        StartCoroutine(SendTTSRequest(message));
    }

    IEnumerator SendTTSRequest(string textToConvert)
    {
        TTSRequest ttsRequest = new TTSRequest
        {
            model = "tts-1",
            input = textToConvert,
            voice = selectedVoice.ToString(),
            response_format = "mp3"
        };

        string jsonPayload = JsonUtility.ToJson(ttsRequest);

        UnityWebRequest request = new UnityWebRequest(ttsApiUrl, "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");
        request.SetRequestHeader("Authorization", "Bearer " + apiKey);

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            OnAudioPlayback?.Invoke(false);
            Debug.LogError("TTS Error: " + request.error);
        }
        else
        {
            Debug.Log("TTS Request Successful");

            byte[] audioData = request.downloadHandler.data;

            // Save the audio data to a local file
            string filePath = Path.Combine(Application.persistentDataPath, "speech.mp3");
            File.WriteAllBytes(filePath, audioData);
            Debug.Log("Audio saved to: " + filePath);

            // Play the saved audio file
            StartCoroutine(PlayAudioClip("file://" + filePath));
        }
    }

    IEnumerator PlayAudioClip(string filePath)
    {
        // Load the audio file to an audio clip with UnityWebRequestMultimedia
        using (UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip(filePath, AudioType.MPEG))
        {
            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                OnAudioPlayback?.Invoke(false);
                Debug.LogError("Audio Playback Error: " + www.error);
            }
            else
            {
                AudioClip clip = DownloadHandlerAudioClip.GetContent(www);
                audioSource.clip = clip;
                audioSource.Play();
                Debug.Log("Playing Audio");
                avatarState.isTalking = true;

                // Start audio playback notification
                OnAudioPlayback?.Invoke(true);

                // Play animation based on emotion
                dialogueManager.StartTalkingAnimation();

                yield return new WaitForSeconds(clip.length);

                // Clean up the audio file
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
                // Stop animation
                dialogueManager.StopTalkingAnimation();

                // Restore recording
                OnAudioPlayback?.Invoke(false);
                avatarState.isTalking = false;
            }
        }
    }
}

[System.Serializable]
public class TTSRequest
{
    public string model;
    public string input;
    public string voice;
    public string response_format;
}
