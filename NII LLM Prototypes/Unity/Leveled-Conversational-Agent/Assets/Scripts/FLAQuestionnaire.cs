using UnityEngine;
using System.Linq;
using TMPro;

public class FLAQuestionnaire : MonoBehaviour
{
    public ClientManager cm;
    public int phase = 0;
    public int likertPoints = 4;
    public TMP_Text questionText;

    string[] preQuestions =
    {
        "英語を話すとき、自信が持てません。",
        "英語で話すよう求められると、強い緊張を感じます。",
        "英語を話さなければならないとき、動悸や強い緊張を感じます。",
        "日本語で話すときよりも、英語を話すときの方が緊張や不安を感じます。",
        "英語を話すとき、不安を感じます。",
        "英語を話すとき、自信をもって話すことができます。",
        "人前で英語を話さなければならないとき、不安を感じます。",
        "英語を話すとき、他の人に否定的に評価されるのではないかと不安になります。",
        "英語で間違いをすることについて、あまり心配していません。",
        "英語のネイティブスピーカーと一緒にいても、比較的リラックスして過ごせると思います。",
        "準備なしに英語を話さなければならないとき、強い不安を感じます。",
        "英語を話す場面では、緊張のため本来知っていることがうまく出てこないことがあります。",
        "英語を話す準備をしていても、不安を感じます。"
    };

    int[] preReverseCoding = { 5,8,9 };

    string[] postQuestions =
    {
        "先ほどの英語会話では、内容を比較的理解することができました。",
        "先ほどの英語会話では、自信をもって話すことができませんでした。",
        "英語を話し始めたとき、強い緊張を感じました。",
        "先ほどの英語会話では、自信をもって話すことができました。",
        "英語を話している間、動悸や強い緊張を感じました。",
        "英語を話している間、不安を感じました。",
        "日本語ではなく英語で話さなければならなかったため、より緊張や不安を感じました。",
        "英語で話す必要があると分かったとき、緊張しました。",
        "先ほどの英語会話では、間違いをすることについてあまり心配していませんでした。",
        "先ほどの英語会話では、自分の英語力について否定的に評価されるのではないかと不安でした。",
        "準備なしで英語を話さなければならなかったとき、強い不安を感じました。",
        "とても緊張して、本来知っていることがうまく出てこないと感じました。",
        "英語を話す準備ができていると感じていても、不安を感じました。"
    };

    int[] postReverseCoding = { 0, 3, 8 };

    int[] postIgnoreScoring = { 0 };

    string responses = "";

    public int iQuestion = 0, score = 0;

    public void HandleResponse(int i)
    {
        responses += i.ToString();

        if (phase == 0)
        {
            if (preReverseCoding.Contains(iQuestion))
            {
                score += (likertPoints - 1 - i);
            }
            else
            {
                score += i;
            }

            iQuestion++;

            if (iQuestion >= preQuestions.Length)
            {
                iQuestion = 0;
                cm.SendLogToServer($"Pre-intervention FLA score: {score}, raw responses: {responses}");
                score = 0;
                responses = "";
                phase++;
                Refresh();
                gameObject.SetActive(false);
            }
        }
        else
        {
            if (!postIgnoreScoring.Contains(iQuestion))
            {
                if (postReverseCoding.Contains(iQuestion))
                {
                    score += (likertPoints - 1 - i);
                }
                else
                {
                    score += i;
                }
            }

            iQuestion++;

            if (iQuestion >= postQuestions.Length)
            {
                iQuestion = 0;
                cm.SendLogToServer($"Post phase {phase} FLA score: {score}, raw responses: {responses}");
                score = 0;
                responses = "";
                phase++;
                Refresh();
                gameObject.SetActive(false);
            }
        }
        Refresh();
    }

    void Refresh()
    {
        questionText.text = ((phase == 0) ? preQuestions : postQuestions)[iQuestion];
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Refresh();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
