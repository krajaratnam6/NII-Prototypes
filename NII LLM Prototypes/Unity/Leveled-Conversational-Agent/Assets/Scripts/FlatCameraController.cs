using UnityEngine;

public class FlatCameraController : MonoBehaviour
{
    public float sensX, sensY;

    public float xMax = 45f, xMin = -45f, yMax = 45f, yMin = -45; 

    public Transform orientation;

    float xRotation, yRotation;

    public GameObject paused;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
    }

    // Update is called once per frame
    void Update()
    {
        if (paused.activeInHierarchy)
            return;

        float mouseX = Input.GetAxisRaw("Mouse X") * Time.deltaTime * sensX;
        float mouseY = Input.GetAxisRaw("Mouse Y") * Time.deltaTime * sensY;

        yRotation += mouseX;
        xRotation -= mouseY;
        xRotation = Mathf.Clamp(xRotation, xMin, xMax);
        yRotation = Mathf.Clamp(yRotation, yMin, yMax);

        transform.rotation = Quaternion.Euler(xRotation, yRotation, 0);
        orientation.rotation = Quaternion.Euler(0, yRotation, 0);
    }
}
