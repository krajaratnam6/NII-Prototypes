using UnityEngine;

public class PhaseManager : MonoBehaviour
{
    public FlatCameraController fcm;
    public Transcribe tr;
    public ClientManager cm;
    public GameObject intermission, intermissionTxt, fla, endScreen, startScreen, pauseScreen;
    public float[] phaseTimes;
    public int currPhase = 0;
    public bool paused = true, waitingForResponse = false;
    public float timer = 0;
    bool waitingForFla = false, waitingForReturn = false, waitingForEnd = false, firstResponse = true;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    public void ResponseFinished()
    {
        firstResponse = false;

        if (!waitingForResponse)
            return;

        if (currPhase > 0)
        {
            fcm.enabled = false;
            tr.canTranscribe = false;
        }

        if (waitingForEnd)
        {
            intermissionTxt.SetActive(false);
            intermission.SetActive(true);
            waitingForFla = true;
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
            fla.SetActive(true);
            return;
        }

        timer = 0;
        currPhase++;
        cm.SendRawMessageToServer("PHASE");
        firstResponse = true;
        if (currPhase < phaseTimes.Length)
        {
            waitingForResponse = false;
            if (currPhase > 1)
            {
                intermissionTxt.SetActive(false);
                intermission.SetActive(true);
                waitingForFla = true;
                Cursor.lockState = CursorLockMode.None;
                Cursor.visible = true;
                fla.SetActive(true);
            }
        }
        else
        {
            waitingForEnd = true;
        }
    }

    public void EndScreen()
    {
        endScreen.SetActive(true);
        cm.SendRawMessageToServer("END");
        waitingForReturn = true;
    }

    // Update is called once per frame
    void Update()
    {
        if (waitingForReturn)
        {
            if (Input.GetKeyDown(KeyCode.Return))
            {
                waitingForReturn = false;
                if (endScreen.activeInHierarchy)
                {
                    Application.Quit();
                }
                else
                {
                    intermission.SetActive(false);
                    fcm.enabled = true;
                    tr.canTranscribe = true;
                    Cursor.lockState = CursorLockMode.Locked;
                    Cursor.visible = false;
                }
            }
            return;
        }

        if (waitingForFla)
        {
            if (!fla.activeInHierarchy)
            {
                waitingForFla = false;

                if (waitingForEnd)
                {
                    EndScreen();
                    return;
                }

                intermissionTxt.SetActive(true);
                waitingForReturn = true;
            }
            return;
        }

        if (!firstResponse && !waitingForResponse && !startScreen.activeInHierarchy && !pauseScreen.activeInHierarchy)
        {
            timer += Time.deltaTime;
            if (timer >= phaseTimes[currPhase])
            {
                waitingForResponse = true;
            }
        }
    }
}
