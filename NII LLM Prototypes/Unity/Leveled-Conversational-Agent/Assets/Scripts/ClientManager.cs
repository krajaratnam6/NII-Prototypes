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

    private TcpClient client;
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

    // Runs in a separate thread to listen for incoming data
    private void ListenForData()
    {
        try
        {
            client = new TcpClient(ipAddress, port);
            stream = client.GetStream();
            Debug.Log("Connected to server!");
            SendServerMessage(cefrLevel);

            while (client.Connected)
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

    private void ReceiveMessage(string str)
    {
        canSend = true;
        llmdialoguemgr.ReceiveMessage(str);
        // output.text = "> " + str;
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
                // output.text = "waiting for server response...";
                if (locking)
                    canSend = false;
            }
            catch (System.Exception e)
            {
                Debug.Log("Socket write exception: " + e);
            }
        }
    }

    void OnDestroy()
    {
        Disconnect();
    }

    void Disconnect()
    {
        if (client != null)
        {
            SendServerMessage("END");
            client.Close();
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
    }
}
