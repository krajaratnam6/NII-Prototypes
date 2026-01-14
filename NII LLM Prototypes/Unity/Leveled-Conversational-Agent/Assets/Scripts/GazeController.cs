using UnityEngine;
using System.Collections;

/// <summary>
/// This is a gaze model based on the paper Eyes Alive https://doi.org/10.1145/566654.566629
/// This gaze model is still pretty much working in progress and may create a lot of unnatural eye movements, consider replasing it with your own implementation or disabling it entirely
/// I would also appreciate any contributions to improve this model
/// </summary>

public class GazeController : MonoBehaviour
{
    [Header("References")]
    public Transform leftEye;
    public Transform rightEye;
    public Transform head;
    public Transform eyeCenter;     //眼球中心参考位置
    public Transform body;

    public Transform gazeTarget;    //注视目标
    public Transform eyeTargetTemp;     //眼球临时目标
    public Transform targetOrigin;
    public Transform headTarget;
    public Transform mainPOI;
    public Transform[] POIs;

    public AvatarState avatarState;

    [Header("Parameters")]
    // 扫视幅度参数
    public float maxSaccadeMagnitudeTalking = 27.5f;
    public float maxSaccadeMagnitudeListening = 22.7f;

    public float saccadeMagnitudeLimitY = 12f;
    public float saccadeMagnitudeLimitX = 27.5f;

    public float headFollowThresholdTalking = 0.9f;     // 说话模式头部跟随阈值百分比，眼动随机值超过最大值*此比例则头部跟随移动
    public float headFollowThresholdListening = 0.9f;       // 聆听模式头部跟随阈值百分比

    // 扫视方向参数
    public float[] directionProbabilities = { 15.50f, 6.50f, 17.70f, 7.40f, 16.80f, 7.89f, 20.40f, 7.80f };
    public float[] directionAngles = { 0f, 45f, 90f, 135f, 180f, 225f, 270f, 315f };

    // 扫视持续时间参数
    public float saccadeIntervalMeanAtTalking = 1.8f;
    public float saccadeIntervalMeanAwayTalking = 2.1f;
    public float saccadeIntervalMeanAtListening = 2.5f;
    public float saccadeIntervalMeanAwayListening = 1.6f;

    // 头部参数
    public float maxHeadHorizontalAngle = 40f;
    public float maxHeadVerticalAngle = 20f;

    public float HeadAngleOffset = -8f;
    public float EyeAngleOffset = -10f;
    private Quaternion _headInitialLocalRotation;

    // 身体旋转相关
    [SerializeField]
    private float _bodyRotateSpeed = 3f; // 身体旋转速度

    //Private Parameters
    // 计时器相关
    //private float saccadeTimer = 0f;
    //private float currentInterval = 0f;

    // 扫视相关
    private float _saccadeMagnitude;
    private float _saccadeDuration;
    //private float _saccadeTimer = 0f;
    //private Quaternion _eyeStartRotLeft, _eyeTargetRotLeft;
    //private Quaternion _eyeStartRotRight, _eyeTargetRotRight;
    private Coroutine _saccadeCoroutine;
    private Coroutine _saccadeTimer;

    [SerializeField] private float _eyeCenterDistance;
    [SerializeField] private float[] _saccadeLimit;

    private bool _isSaccading = false;

    // 身体旋转相关
    private Quaternion _bodyTargetRotation;
    [SerializeField]
    private bool _isBodyRotating = false;

    public float headRotateSpeed = 120f; // 头部旋转速度（度/秒）

    //SuddenMove
    private Vector3 _targetLastPos;
    [SerializeField] private float _suddenMoveThreshold = 0.5f; // 阈值，单位为米

    void Start()
    {
        Initialize();
        _saccadeTimer = StartCoroutine(SaccadeRoutine());
    }

    void Initialize()
    {
        _headInitialLocalRotation = head.localRotation; // 记录头部相对于身体的初始本地旋转
        //初始化目标
        SetPrimaryGazeTarget(mainPOI);  //设置主注视目标
        SetHeadTarget(mainPOI);         //设置头部目标
        CalculateSaccadeLimit();
    }

    public void SetPrimaryGazeTarget(Transform target)
    {
        gazeTarget.position = target.position;
        targetOrigin.transform.LookAt(target);
        eyeTargetTemp.position = target.position;
    }

    void SetHeadTarget(Transform target)
    {
        headTarget.position = target.position;
    }

    void CalculateSaccadeLimit()    //暂时无用
    {
        // 计算eyeCenter到leftEye和rightEye连线的垂线距离
        Vector3 leftToRight = rightEye.position - leftEye.position;
        Vector3 leftToCenter = eyeCenter.position - leftEye.position;
        float projectionLength = Vector3.Dot(leftToCenter, leftToRight.normalized);  //点积计算投影长度
        Vector3 projectedPoint = leftEye.position + leftToRight.normalized * projectionLength;
        _eyeCenterDistance = Vector3.Distance(eyeCenter.position, projectedPoint);
    }

    IEnumerator SaccadeRoutine()    //循环计时
    {
        float meanInterval;
        if (avatarState.isTalking)
        {
            meanInterval = avatarState.isGazeAt ? saccadeIntervalMeanAtTalking : saccadeIntervalMeanAwayTalking;
        }
        else
        {
            meanInterval = avatarState.isGazeAt ? saccadeIntervalMeanAtListening : saccadeIntervalMeanAwayListening;
        }

        // 生成随机时间间隔
        float interval = GenerateRandomSaccadeInterval(meanInterval);

        // 等待指定时间
        yield return new WaitForSeconds(interval);

        //后续任务————————————————————
        if (avatarState.isGazeAt)   //注视模式则开始随机扫视
        {
            AwaySaccade();

        }
        else
        {
            ReturnSaccade();
        }
    }

    float GenerateRandomSaccadeInterval(float mean)
    {
        // 生成0到1之间的随机数
        float U = Mathf.Clamp(UnityEngine.Random.value, float.Epsilon, 1f - float.Epsilon); //float.Epsilon 是浮点数最小值

        // 使用指数分布公式：-mean * ln(1-U)
        float interval = -mean * Mathf.Log(1 - U);

        return interval;
    }

    void AwaySaccade()
    {
        _saccadeMagnitude = GenerateRandomSaccadeMagnitude();    //随机扫视幅度
        float directionAngle = ChooseRandomDirection();
        _saccadeMagnitude = Mathf.Min(_saccadeMagnitude, GetMaxSaccadeMagnitude(directionAngle)); // 限制在椭圆范围内
        _saccadeDuration = 25f + 2.4f * _saccadeMagnitude;  //扫视持续时间，单位为毫秒

        // 获取当前注视方向
        Quaternion baseRot = targetOrigin.rotation;

        // 计算扫视旋转
        float theta = directionAngle * Mathf.Deg2Rad;
        float horizontal = Mathf.Cos(theta) * _saccadeMagnitude; // 左右
        float vertical = Mathf.Sin(theta) * _saccadeMagnitude;   // 上下
        Quaternion saccadeRot = Quaternion.Euler(vertical, horizontal, 0);

        // 目标旋转
        Quaternion endRot = baseRot * saccadeRot;

        //Debug.Log("directionAngle: " + directionAngle);
        //Debug.Log("saccadeMagnitude: " + _saccadeMagnitude);
        //Debug.Log("RotationX: " + endRot.eulerAngles.x);
        //Debug.Log("RotationY: " + endRot.eulerAngles.y);

        if (_saccadeCoroutine != null)
            StopCoroutine(_saccadeCoroutine);
        _saccadeCoroutine = StartCoroutine(SaccadeProcess(baseRot, endRot, _saccadeDuration));
    }

    void ReturnSaccade()
    {
        // 计算当前扫视点到主注视点的方向向量（在eyeCenter本地坐标系下）
        Vector3 localFrom = eyeCenter.InverseTransformPoint(eyeTargetTemp.position);
        Vector3 localTo = eyeCenter.InverseTransformPoint(gazeTarget.position);

        Vector3 delta = localTo - localFrom;

        // 计算极坐标下的幅度和方向
        float x = delta.x;
        float y = delta.y;
        float returnMagnitude = new Vector2(x, y).magnitude; // 幅度
        float returnAngle = Mathf.Atan2(y, x) * Mathf.Rad2Deg; // 方向角

        // 限制幅度在椭圆范围内
        //returnMagnitude = Mathf.Min(returnMagnitude, GetMaxSaccadeMagnitude(returnAngle));
        _saccadeMagnitude = returnMagnitude;
        _saccadeDuration = 25f + 2.4f * _saccadeMagnitude;

        // 获取当前注视方向
        Quaternion baseRot = targetOrigin.rotation;

        // 计算扫视旋转
        float theta = returnAngle * Mathf.Deg2Rad;
        float horizontal = Mathf.Cos(theta) * _saccadeMagnitude; // 左右
        float vertical = Mathf.Sin(theta) * _saccadeMagnitude;   // 上下
        Quaternion saccadeRot = Quaternion.Euler(vertical, horizontal, 0);

        // 目标旋转
        Quaternion endRot = baseRot * saccadeRot;

        if (_saccadeCoroutine != null)
            StopCoroutine(_saccadeCoroutine);
        _saccadeCoroutine = StartCoroutine(SaccadeProcess(baseRot, endRot, _saccadeDuration));
    }

    IEnumerator SaccadeProcess(Quaternion startRot, Quaternion endRot, float duration)
    {
        float timer = 0f;
        while (timer < duration)
        {
            timer += Time.deltaTime * 1000f; // 毫秒
            float t = Mathf.Clamp01(timer / duration);
            float easeT = EaseInOutSine(t);
            targetOrigin.rotation = Quaternion.Slerp(startRot, endRot, easeT);
            yield return null;
        }
        targetOrigin.rotation = endRot;

        // 切换凝视状态
        avatarState.isGazeAt = !avatarState.isGazeAt;

        //开始新的扫视循环
        StartCoroutine(SaccadeRoutine());
    }

    float GenerateRandomSaccadeMagnitude()
    {
        // 生成0到15之间的随机数
        float rand = UnityEngine.Random.Range(float.Epsilon, 15f);

        // 计算magnitude：−6.9 ∗ ln(Rand/15.7)
        float magnitude = -6.9f * Mathf.Log(rand / 15.7f);

        return magnitude;
    }

    float ChooseRandomDirection()
    {
        float rand = UnityEngine.Random.value;

        float total = 0f;
        foreach (float p in directionProbabilities)
            total += p;

        // 累加概率，找到rand落在哪个区间
        float cumulative = 0f;
        for (int i = 0; i < directionProbabilities.Length; i++)
        {
            cumulative += directionProbabilities[i] / total; // 归一化
            if (rand <= cumulative)
                return directionAngles[i];
        }

        // 防止浮点误差，返回最后一个
        return directionAngles[directionAngles.Length - 1];
    }


    // 根据方向角度计算最大扫视幅度
    public float GetMaxSaccadeMagnitude(float directionAngle)
    {
        float a = saccadeMagnitudeLimitX;
        float b = saccadeMagnitudeLimitY;
        float theta = directionAngle * Mathf.Deg2Rad; // 角度转弧度

        float numerator = a * b;
        float denominator = Mathf.Sqrt(Mathf.Pow(b * Mathf.Cos(theta), 2) + Mathf.Pow(a * Mathf.Sin(theta), 2));    //椭圆极坐标公式
        return numerator / denominator;
    }

    void VoR()
    {
        //随机小角度转动头部保持眼球不动
    }

    float EaseInOutSine(float t)
    {
        return -(Mathf.Cos(Mathf.PI * t) - 1f) / 2f;
    }

    void LateUpdate()
    {
        targetOrigin.position = eyeCenter.position;  //将眼球原点放置到头部参考点位置
        //判断状态若在注视则持续跟踪
        if (avatarState.isGazeAt && !_isSaccading)
        {
            SetPrimaryGazeTarget(mainPOI);  //当注视且不在扫视的时候持续注视主目标
            SetHeadTarget(mainPOI);
            _targetLastPos = mainPOI.position;  //记录主目标位置
        }
        if (SuddenMove(mainPOI.position, _targetLastPos))   //若主目标突然移动则立刻切换注视
        {
            if (_saccadeTimer != null)
                StopCoroutine(_saccadeTimer);
            if (_saccadeCoroutine != null)
                StopCoroutine(_saccadeCoroutine);
            _isSaccading = false;
            avatarState.isGazeAt = true;
            SetPrimaryGazeTarget(mainPOI);
            SetHeadTarget(mainPOI);
            Debug.Log("Sudden Move Detected");
        }
        HeadLookAt();
        RotateBody();
        EyeLookAt();
    }

    void EyeLookAt()
    {
        if (eyeTargetTemp != null)
        {
            Vector3 toTargetLeft = (eyeTargetTemp.position - leftEye.position).normalized;
            Quaternion rotLeft = Quaternion.LookRotation(Vector3.Cross(Vector3.up, toTargetLeft), -toTargetLeft);
            rotLeft *= Quaternion.Euler(0, -90, 0);
            leftEye.rotation = rotLeft;
            //Quaternion localRotLeft = Quaternion.Inverse(leftEye.parent.rotation) * rotLeft;
            //Vector3 leftEuler = localRotLeft.eulerAngles;
            //leftEuler.x = (leftEuler.x - 180f ) / 2f - 180 + EyeAngleOffset;
            //leftEuler.z = leftEuler.z /2f;
            //leftEye.localEulerAngles = leftEuler;
            //leftEye.localRotation *= Quaternion.Euler(0, 0, 90);
            //leftEye.localRotation *= Quaternion.Euler(90, 0, 0);

            Vector3 toTargetRight = (eyeTargetTemp.position - rightEye.position).normalized;
            Quaternion rotRight = Quaternion.LookRotation(Vector3.Cross(Vector3.up, toTargetRight), -toTargetRight);
            rotRight *= Quaternion.Euler(0, -90, 0);
            rightEye.rotation = rotRight;
            //Quaternion localRotRight = Quaternion.Inverse(rightEye.parent.rotation) * rotRight;
            //Vector3 rightEuler = localRotRight.eulerAngles;
            //rightEuler.x = (rightEuler.x - 180f) / 2f - 180f + EyeAngleOffset;
            //rightEuler.z = rightEuler.z / 2f;
            //rightEye.localEulerAngles = rightEuler;
            //rightEye.localRotation *= Quaternion.Euler(0, 0, 90);
            //rightEye.localRotation *= Quaternion.Euler(90, 0, 0);
        }
    }

    void HeadLookAt()
    {
        if (headTarget != null)
        {
            // 计算目标在身体坐标系下的方向
            Vector3 localHeadPos = body.InverseTransformPoint(head.position);   //将头部位置数据从世界坐标转换到身体坐标系下
            Vector3 localTargetPos = body.InverseTransformPoint(headTarget.position);
            Vector3 toTargetLocal = (localTargetPos - localHeadPos).normalized;

            // 计算本地目标旋转
            Quaternion desiredLocalRot = Quaternion.LookRotation(toTargetLocal, Vector3.up);
            desiredLocalRot *= Quaternion.Euler(HeadAngleOffset, 0, 0);

            // 计算与初始本地旋转的相对旋转
            Quaternion relativeRot = Quaternion.Inverse(_headInitialLocalRotation) * desiredLocalRot;
            Vector3 relativeEuler = relativeRot.eulerAngles;
            relativeEuler.x = (relativeEuler.x > 180) ? relativeEuler.x - 360 : relativeEuler.x;
            relativeEuler.y = (relativeEuler.y > 180) ? relativeEuler.y - 360 : relativeEuler.y;

            // 检查是否超出限制
            bool overHorizontal = Mathf.Abs(relativeEuler.y) > maxHeadHorizontalAngle;
            //bool overVertical = Mathf.Abs(relativeEuler.x) > maxHeadVerticalAngle;

            //if (overHorizontal || overVertical && !_isBodyRotating)
            if (overHorizontal && !_isBodyRotating)
            {
                CalculateBodyRotation(toTargetLocal);
            }
            else
            {
                head.localRotation = desiredLocalRot;
                // 平滑旋转
                head.localRotation = Quaternion.RotateTowards(
                    head.localRotation,
                    desiredLocalRot,
                    headRotateSpeed * Time.deltaTime
                );
            }
        }
    }

    void CalculateBodyRotation(Vector3 toTargetLocal)
    {
        // 记录目标在世界坐标系下的方向
        Vector3 worldTargetPos = body.TransformPoint(toTargetLocal);    //将目标点转换到身体坐标系下
        Vector3 bodyPos = body.position;
        Vector3 flatDir = worldTargetPos - bodyPos;     //计算目标点与身体原点之间的方向
        flatDir.y = 0; // 压平y轴，只在水平方向旋转

        if (flatDir.sqrMagnitude > 0.0001f) //避免距离过近出现异常
        {
            _bodyTargetRotation = Quaternion.LookRotation(flatDir, Vector3.up);
            _isBodyRotating = true;
        }
    }

    void RotateBody()
    {
        if (_isBodyRotating)
        {
            body.rotation = Quaternion.Slerp(body.rotation, _bodyTargetRotation, Time.deltaTime * _bodyRotateSpeed);
            // 判断是否已接近目标朝向
            if (Quaternion.Angle(body.rotation, _bodyTargetRotation) < 0.5f)
            {
                body.rotation = _bodyTargetRotation;
                _isBodyRotating = false;
            }
        }
    }

    bool SuddenMove(Vector3 currentPos, Vector3 previousPos)
    {
        return Vector3.Distance(currentPos, previousPos) > _suddenMoveThreshold;
    }
}
