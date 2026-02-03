using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

public class ClientManager : MonoBehaviour
{
    LLMDialogueManager llmdialoguemgr;
    public string ipAddress = "192.168.86.33";
    public int port = 12345;
    public string cefrLevel = "A1";
    public bool canSend = true;
    public GameObject waitingAnimation, spacebarMsg;
    public Transcribe transcribe;
    public StartScreenManager ssm;

    private TcpClient client = null;
    private NetworkStream stream;
    private byte[] receiveBuffer = new byte[1024];
    private Thread clientReceiveThread;
    private string incoming = "";

    private void Start()
    {
        llmdialoguemgr = GetComponent<LLMDialogueManager>();
        ConnectToServer();
    }

    // Call this method to connect to the server
    public void ConnectToServer()
    {
        try
        {
            clientReceiveThread = new Thread(new ThreadStart(ListenForData));
            clientReceiveThread.IsBackground = true;
            clientReceiveThread.Start();
            Debug.Log("Client started listening for data...");
        }
        catch (System.Exception e)
        {
            Debug.Log("On client connect exception: " + e);
        }
    }

    public void Connect(string ipAddress, int port, string cefrLevel, string permutation)
    {
        try
        {
            client = new TcpClient(ipAddress, port);
            stream = client.GetStream();
            Debug.Log("Connecting to server...");
            SendServerMessage($"{cefrLevel} {permutation}");
        }
        catch (SocketException socketException)
        {
            Debug.Log("Socket exception: " + socketException);
            ssm.BadAck();
        }

    }
    // Runs in a separate thread to listen for incoming data
    private void ListenForData()
    {
        while (true)
        {
            try
            {
                if (client != null && stream != null && client.Connected)
                {
                    int bytesRead = stream.Read(receiveBuffer, 0, receiveBuffer.Length);
                    if (bytesRead > 0)
                    {
                        string receivedData = Encoding.ASCII.GetString(receiveBuffer, 0, bytesRead);
                        // Must use Unity's main thread for UI or game object interaction
                        // You can use a queue or similar mechanism to pass data safely
                        Debug.Log("Received from server: " + receivedData);
                        incoming = receivedData;
                    }
                }
            }
            catch (SocketException socketException)
            {
                Debug.Log("Socket exception: " + socketException);
            }
        }
    }

    private void ReceiveMessage(string str)
    {
        if (ssm.state == 0)
        {
            canSend = true;
            if (str == "Success\n")
            {
                ssm.GoodAck();
            }
            else
            {
                ssm.BadAck();
            }
        }
        else
        {
            canSend = true;
            llmdialoguemgr.ReceiveMessage(str);
            waitingAnimation.SetActive(false);
        }
    }

    public void SendServerMessage(string message, bool locking = true)
    {
        if (client != null && client.Connected && canSend)
        {
            try
            {
                byte[] messageAsByteArray = Encoding.ASCII.GetBytes("MSG: " + message + " ");
                stream.Write(messageAsByteArray, 0, messageAsByteArray.Length);
                Debug.Log("Sent to server: " + message);
                if (ssm.state != 0)
                {
                    waitingAnimation.SetActive(true);
                    transcribe.canTranscribe = false;
                }
                if (locking)
                    canSend = false;
            }
            catch (System.Exception e)
            {
                Debug.Log("Socket write exception: " + e);
                // bring up error screen
                throw e;
            }
        }
    }

    void OnDestroy()
    {
        Disconnect();
    }

    private void OnApplicationQuit()
    {
        Disconnect();
    }

    void Disconnect()
    {
        if (client != null && client.Connected)
        {
            try
            {
                byte[] messageAsByteArray = Encoding.ASCII.GetBytes("END");
                stream.Write(messageAsByteArray, 0, messageAsByteArray.Length);
                Debug.Log("Sent to server: END");
            }
            catch (System.Exception e)
            {
                Debug.Log("Socket write exception: " + e);
            }

            client.Close();
            client = null;
        }
        if (clientReceiveThread != null)
        {
            clientReceiveThread.Abort();
        }
    }

    void Update()
    {
        if (incoming.Length > 0)
        {
            ReceiveMessage(incoming);
            incoming = "";
        }

        if (transcribe.canTranscribe && !spacebarMsg.activeInHierarchy)
        {
            spacebarMsg.SetActive(true);
        }
        else if (!transcribe.canTranscribe && spacebarMsg.activeInHierarchy)
        {
            spacebarMsg.SetActive(false);
        }
    }
}
