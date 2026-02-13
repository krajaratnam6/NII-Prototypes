using UnityEngine;
using TMPro;

public class StartScreenManager : MonoBehaviour
{
    public Transcribe tr;
    public ClientManager cm;
    public GameObject waitingAnim, initSettings, canDoQuestionnaire;
    public TMP_Text returnKeyMsg, surveyQuestion, welcomeText;
    public FlatCameraController fcm;
    bool listeningForReturn = true;
    public int state = 0;
    public bool finishedCanDo = false;

    public TMP_InputField addr, port;
    public TMP_Dropdown perm;

    int canDoStage = 0, canDoLevel = 0, canDoQuestion = 0, disagreeCount = 0;
    int cefrSpeaking = -1, cefrListening = -1;

    public string cefrLevel = "A0";

    string listeningAnswers = "", speakingAnswers = "";

    string[] cefrLevels = {"A1","A2","B1","B2","C1","C2"};

    string[][] canDoListening =
        {
        new string[] { "数字、値段、時間を聞いて、理解することができる。", // A1 Questions (11 items)
                       "ゆっくりはっきりと話され、考える間があれば、言ったことを理解することができる。",
                       "意味が分かるように長いポーズを置いて、当人に向かって非常にゆっくりと話され、はっきりと発音されれば、言っていることを理解することができる。",
                       "ゆっくりはっきりと話されれば、自分についての簡単な質問を理解することができる。",
                       "時間や日付を聞いて、理解することができる。",
                       "基本的な挨拶や決まり文句を聞いて、理解することができる。（例：please, thank you)",
                       "ゆっくり丁寧に話されれば、簡単な質問を理解することができる。",
                       "すぐ周りの日用品の名前を聞いて、理解することができる。",
                       "数字や値段を聞いて、理解することができる。",
                       "曜日や月を聞いて、理解することができる。",
                       "教室内の日用品の名前を聞いて、理解することができる。" },
        new string[] { "人々、その家族、家、仕事、趣味などに関する基本的な情報を理解することができる。", // A2 Questions (9 items)
                       "自分自身や自分の家族に関するごく簡単な文を理解することができる。",
                       "日常の個人的ニーズ（買い物、外食、医者に行くなど）に関する簡単な語句、質問や情報を聞いて、理解することができる。",
                       "自分自身、自分の家族、買い物、勉強に関する聞き慣れた語や表現を理解することができる。",
                       "自分が関心がある分野（趣味、社会生活、休日、音楽、テレビ、映画、旅行など）に関してよく使われる語句を聞いて、理解することができる。",
                       "ゆっくりはっきりと話されれば、自分自身や自分の家族に関する基礎的な語句を聞いて、理解することができる。",
                       "自分自身や人々、家・学校・友人・ペット・趣味などの身の回りの事柄に関する句や表現を聞いて、理解することができる。",
                       "直接自分につながりのある領域（家族、学校、地域）に関してよく使われる語句を聞いて、理解することができる。",
                       "最優先事項（ごく基本的な個人や家族の情報、買い物、地域、仕事など）に関する語句や表現を聞いて、理解することができる。" },
        new string[] { "当人に向かってはっきりと標準語で話されれば、日常会話の要点を理解することができる。", // B1 Questions (5 items)
                       "短い物語を聞いて、次に何が起こるかを推測できる程度にその内容を理解することができる。",
                       "はっきりと標準語で話されれば、短い物語や身近なトピック（現代の文化など）についての長い話を理解することができる。",
                       "なじみのある発音ではっきりと話されれば、日常の勉強や仕事に関するトピックについての明確な事実情報を理解することができ、メッセージの概要とともに詳細も聞き取ることができる。",
                       "短い物語を聞いて、次に何が起こるかについて仮定することができる。" }
        };

    string[][] canDoSpeaking =
    {
        new string[] { "相手の状態を尋ねたり、言われたことに反応することができる",
                        "他人を紹介したり、挨拶やお別れの基本的な表現を使うことができる",
                        "「お願いします」「ありがとう」といった基本的な挨拶や決まり文句を言うことができ、相手の状態を尋ねたり、自分の状態を言うことができる",
                        "相手の状態を尋ねたり、ニュースへ反応することができる",
                        "相手の状態を尋ねたり、似たような質問に答えることができる" },
    new string[] { "普段の状況で興味がある話題であれば短い会話に参加することができる",
                    "誘ったり、相手からの誘いに反応することができる",
                    "謝ったり、謝りを受け入れることができる",
                    "誰かに会ったり、誰かと別れるときに友好的な関係を築けるような言葉を伝えることができる",
                    "簡単な言葉で自分の感情を表現し、感謝の意を表明することができる" ,
                    "丁寧に相手に話しかけることができる" },
    new string[] { "会話や議論についていくことができるが、言いたいことを正確に言おうとするとついていけなくなる可能性がある",
                    "話者と興味が一致していることに関してかなり長い間会話ができる",
                    "会話に参加することができ、お互いに共通の仕事上の話題に関する質問をしたり、質問に答えることができる",
                    "語の使用域がおおむね適切であれば、身近なことや興味があるほとんどの話題に関する会話に十分対応することができる",
                    "長い会話や議論を維持することができるが、ときに自分の考えを伝えるときに少し助けが必要な可能性がある",
                    "驚き、幸せ、悲しみ、関心、無関心といった感情を適切に表現したり、そういった感情に適切に反応することができる" },
    };

    float canDoThreshold = 0.79f; // If they pass 75% of the questions on each level, they can continue

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    public void CanDoAnswer(bool agree)
    {
        if (canDoStage == 0)
        {
            listeningAnswers += (agree ? "1" : "0");
        }
        else
        {
            speakingAnswers += (agree ? "1" : "0");
        }

        disagreeCount += (agree ? 0 : 1);
        canDoQuestion++;
        ManageCanDo();
    }

    void ManageCanDo()
    {
        welcomeText.gameObject.SetActive(false);

        string[][] questionBank = (canDoStage == 0) ? canDoListening : canDoSpeaking;
        float disagreeRatio = ((float) disagreeCount) / (questionBank[canDoLevel].Length);
        float agreeRatio = ((float) canDoQuestion - disagreeCount) / (questionBank[canDoLevel].Length);
        bool belowThreshold = (disagreeRatio > ((1.0f - canDoThreshold) + 0.00001f));
        bool aboveThreshold = (agreeRatio >= canDoThreshold);

        print($"Disagree count: {disagreeCount}, Disagree ratio: {disagreeRatio}\n");

        if (canDoQuestion >= questionBank[canDoLevel].Length || belowThreshold || aboveThreshold)
        {
            if (!belowThreshold)
            {
                if (aboveThreshold)
                {
                    print("Above can-do threshold... skipping rest of level");

                    while (canDoQuestion < questionBank[canDoLevel].Length)
                    {
                        if (canDoStage == 0)
                        {
                            listeningAnswers += "-";
                        }
                        else
                        {
                            speakingAnswers += "-";
                        }
                    }
                }

                canDoLevel++;

                if (canDoStage == 0)
                {
                    cefrListening++;
                }
                else
                {
                    cefrSpeaking++;
                }
            }
            else
            {
                print("Below can-do threshold");
            }

            if (canDoLevel >= questionBank.Length || belowThreshold)
            {
                canDoLevel = 0;
                if (canDoStage == 0)
                {
                    canDoStage = 1;
                }
                else
                {
                    welcomeText.text = "ありがとうございました";
                    canDoQuestionnaire.SetActive(false);
                    print($"CEFR Listening Answers: {listeningAnswers}, CEFR Speaking Answers: {speakingAnswers}\n");
                    print($"CEFR Listening: {cefrListening}, CEFR Speaking: {cefrSpeaking}\n");
                    int lvl = Mathf.Min(cefrListening, cefrSpeaking);
                    if (lvl >= 0)
                    {
                        cefrLevel = cefrLevels[lvl];
                    }
                    else
                    {
                        cefrLevel = "A0";
                    }
                    canDoQuestionnaire.SetActive(false);
                    welcomeText.gameObject.SetActive(true);

                    finishedCanDo = true;

                    SendRequest();

                    return;
                }
            }

            canDoQuestion = 0;
            disagreeCount = 0;

        }

        questionBank = (canDoStage == 0) ? canDoListening : canDoSpeaking;
        surveyQuestion.text = questionBank[canDoLevel][canDoQuestion];
        canDoQuestionnaire.SetActive(true);
    }

    void SendRequest()
    {
        waitingAnim.SetActive(true);
        int portNumber = 0;
        int.TryParse(port.text, out portNumber);
        StartCoroutine(cm.Connect(addr.text, portNumber, cefrLevel, perm.value.ToString()));
    }

    public void BadAck()
    {
        initSettings.SetActive(true);
        returnKeyMsg.gameObject.SetActive(true);
        returnKeyMsg.text = "Could not connect to the server.";
        waitingAnim.SetActive(false);
        listeningForReturn = true;
    }

    public void GoodAck()
    {
        state = 1;
        listeningForReturn = true;
        waitingAnim.SetActive(false);
        returnKeyMsg.text = "リターンキーを押して開始します";
        returnKeyMsg.gameObject.SetActive(true);
    }

    // Update is called once per frame
    void Update()
    {
        if (listeningForReturn)
        {
            if (Input.GetKeyUp(KeyCode.Return) || Input.GetKeyUp(KeyCode.KeypadEnter))
            {
                listeningForReturn = false;
                if (state == 0)
                {
                    initSettings.SetActive(false);
                    returnKeyMsg.gameObject.SetActive(false);
                    welcomeText.gameObject.SetActive(false);

                    if (finishedCanDo)
                    {
                        SendRequest();
                    }
                    else
                    {
                        ManageCanDo();
                    }
                }
                else if (state == 1)
                {
                    fcm.enabled = true;
                    tr.canTranscribe = true;
                    gameObject.SetActive(false);
                }
            }
        }

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }
    }
}
