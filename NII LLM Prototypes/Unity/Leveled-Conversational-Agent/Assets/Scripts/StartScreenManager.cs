using UnityEngine;
using TMPro;

public class StartScreenManager : MonoBehaviour
{
    public Transcribe tr;
    public ClientManager cm;
    public GameObject waitingAnim, returnKeyMsg, initSettings;
    public FlatCameraController fcm;
    bool listeningForReturn = true;
    public int state = 0;

    public TMP_InputField addr, port;
    public TMP_Dropdown cefr, perm;

    string[] cefrLevels = {"A1","A2","B1","B2","C1","C2"};

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    public void BadAck()
    {
        initSettings.SetActive(true);
        returnKeyMsg.SetActive(true);
        waitingAnim.SetActive(false);
        listeningForReturn = true;
    }

    public void GoodAck()
    {
        state = 1;
        listeningForReturn = true;
        waitingAnim.SetActive(false);
        returnKeyMsg.SetActive(true);
    }

    // Update is called once per frame
    void Update()
    {
        if (listeningForReturn)
        {
            if (Input.GetKeyUp(KeyCode.Return))
            {
                listeningForReturn = false;
                if (state == 0)
                {
                    initSettings.SetActive(false);
                    returnKeyMsg.SetActive(false);
                    waitingAnim.SetActive(true);
                    int portNumber = 0;
                    int.TryParse(port.text, out portNumber);
                    cm.Connect(addr.text, portNumber, cefrLevels[cefr.value], perm.value.ToString());
                }
                else if (state == 1)
                {
                    fcm.enabled = true;
                    tr.canTranscribe = true;
                    gameObject.SetActive(false);
                }
            }
        }
    }
}
