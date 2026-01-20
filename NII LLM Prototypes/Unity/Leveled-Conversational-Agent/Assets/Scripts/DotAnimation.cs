using UnityEngine;

public class DotAnimation : MonoBehaviour
{
    public RectTransform[] dot;
    public float rise = 80, riseTime = 0.2f, timeBetween = 0.3f;
    int i = 0;
    float origY;
    bool rising = true;
    float timer = 0;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        origY = dot[0].position.y;
    }

    // Update is called once per frame
    void Update()
    {
        bool end = false;
        float maxTime = ((i == dot.Length) ? timeBetween : riseTime);
        timer += Time.deltaTime;
        if (timer >= maxTime)
        {
            timer = maxTime;
            end = true;
        }

        if (i < dot.Length)
        {
            float srcY = rising ? origY : (origY + rise);
            float dstY = rising ? (origY + rise) : origY;
            float y = Mathf.Lerp(srcY, dstY, timer / maxTime);
            dot[i].transform.position = new Vector3(dot[i].transform.position.x, y, dot[i].transform.position.z);
        }

        if (end)
        {
            timer = 0;
            if (i == dot.Length)
            {
                rising = !rising;
                i = 0;
            }
            else
            {
                ++i;
            }
        }
    }
}
