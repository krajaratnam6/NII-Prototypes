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
        "英語を話すとき、私は十分に自信が持てない。",
        "英語で間違えても、心配しない。",
        "英語を話すことになると分かると、体が震えるほど緊張する。",
        "準備なしで英語を話さなければならないと、不安になる。",
        "英語の授業では、緊張のあまり本来知っていることでも忘れてしまう。",
        "英語で話す準備ができていても、不安を感じる。",
        "英語を話すとき、自分に自信がある。",
        "英語を話さなければならないとき、心臓がドキドキする。",
        "人前で英語を話さなければならないと、不安で心もとない気持ちになる。",
        "日本語ではなく英語で話すと、より緊張して不安になる。",
        "英語を話すとき、緊張し不安になる。",
        "これから英語を話すと分かっているとき、私は自信がありリラックスしている。",
        "英語で学ぶべき文法規則の多さに圧倒される。",
        "英語を話すとき、他の人に笑われるのではないかと不安になる。",
        "英語母語話者の前でも、安心して話すことができる。"
    };

    int[] preReverseCoding = { 1, 6, 11, 14 };

    string[] postQuestions =
    {
        "今の会話は理解しやすいと感じた。",
        "先ほど英語を話したとき、私は十分に自信を持てなかった。",
        "英語で間違えることについて心配しなかった。",
        "英語を話し始めるとき、私は体が震えるほど緊張した。",
        "準備なしで英語を話さなければならず、とても不安になった。",
        "緊張のあまり、本当は知っていることでも忘れてしまった。",
        "英語で話す準備はできていると感じていたが、それでも不安だった。",
        "英語を話すとき、自分に自信があった。",
        "英語を話さなければならないとき、心臓がドキドキした。",
        "英語を話さなければならないとき、不安で心もとない気持ちになった。",
        "日本語ではなく英語で話さなければならなかったため、より緊張し不安になったと思う。",
        "英語を話すことになっているとき、緊張し不安を感じた。",
        "英語を話さなければならないと分かっていたが、とても自信がありリラックスしていた。",
        "英語で話す際、文法規則の多さに圧倒された。",
        "自分の英語力について笑われるのではないかと不安だった。",
        "このバーチャルの英語話者と話すとき、安心して会話することができた。"
    };

    int[] postReverseCoding = { 2, 7, 12, 15 };

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
