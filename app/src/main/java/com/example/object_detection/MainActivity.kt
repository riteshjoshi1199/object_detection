package com.example.object_detection

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Paint.Style.FILL
import android.graphics.Paint.Style.STROKE
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.hardware.camera2.params.SessionConfiguration.SESSION_REGULAR
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.object_detection.databinding.ActivityMainBinding
import com.example.object_detection.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.util.concurrent.Executors


class MainActivity: AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    private val colors: List<Int> by lazy {
        listOf(
            android.graphics.Color.BLUE,
            android.graphics.Color.GREEN,
            android.graphics.Color.RED,
            android.graphics.Color.CYAN,
            android.graphics.Color.GRAY,
            android.graphics.Color.BLACK,
            android.graphics.Color.DKGRAY,
            android.graphics.Color.MAGENTA,
            android.graphics.Color.YELLOW,
            android.graphics.Color.RED,
        )
    }

    private val paint by lazy { Paint() }
    private lateinit var bitmap: Bitmap
    private lateinit var model: SsdMobilenetV11Metadata1


    @SuppressLint("MissingPermission")
    private fun openCamera() {
        val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        cameraManager.openCamera(cameraManager.cameraIdList[0], object: CameraDevice.StateCallback() {
            @RequiresApi(Build.VERSION_CODES.P)
            override fun onOpened(camera: CameraDevice) {
                val surfaceTexture = binding.textureView.surfaceTexture
                val surface = Surface(surfaceTexture)

                val captureRequest = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                camera.createCaptureSession(SessionConfiguration(SESSION_REGULAR, listOf(OutputConfiguration(surface)),
                    Executors.newSingleThreadScheduledExecutor(), object: CameraCaptureSession.StateCallback() {
                        override fun onConfigured(session: CameraCaptureSession) {
                            session.setRepeatingRequest(captureRequest.build(), null, null)
                        }

                        override fun onConfigureFailed(session: CameraCaptureSession) {
                            Log.e(TAG, "onConfigureFailed:")
                        }
                    }))
            }

            override fun onDisconnected(camera: CameraDevice) {
                Log.e(TAG, "onDisconnected: ")
            }

            override fun onError(camera: CameraDevice, error: Int) {
                Log.e(TAG, "onError: $error")
            }

        }, handler)

    }

    private val surfaceTextureListener by lazy {
        object: TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                openCamera()
            }


            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                bitmap = binding.textureView.bitmap!!
                var image = TensorImage.fromBitmap(bitmap)

                val imageProcessor: ImageProcessor = ImageProcessor.Builder()
                    .add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR))
                    .build()

                image = imageProcessor.process(image)

                val outputs = model.process(image)
                val locations = outputs.locationsAsTensorBuffer.floatArray
                val classes = outputs.classesAsTensorBuffer.floatArray
                val scores = outputs.scoresAsTensorBuffer.floatArray
                val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray

                val rectangleImage = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(rectangleImage)

                val h = rectangleImage.height
                val w = rectangleImage.width

                paint.textSize = h / 15f
                paint.strokeWidth = h / 85f

                scores.forEachIndexed { index, score ->
                    if (score > 0.5) {
                        val x1 = locations[4 * index + 1] * w
                        val y1 = locations[4 * index] * h
                        val x2 = locations[4 * index + 3] * w
                        val y2 = locations[4 * index + 2] * h

                        paint.color = colors[index % colors.size]
                        paint.style = STROKE
                        canvas.drawRect(x1, y1, x2, y2, paint)
                        paint.style = FILL
                        canvas.drawText(
                            labels[classes[index].toInt()] + " " + String.format("%.2f", score),
                            x1,
                            y1 - 10,
                            paint
                        )
                    }
                }

                binding.imageView.setImageBitmap(rectangleImage)
            }

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                return false
            }

        }
    }

    private lateinit var handler: Handler
    private lateinit var labels: List<String>
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        getCameraPermissions()

        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        labels = FileUtil.loadLabels(this, "labels.txt")
        model = SsdMobilenetV11Metadata1.newInstance(this@MainActivity)
        binding.textureView.surfaceTextureListener = surfaceTextureListener
    }


    private fun getCameraPermissions() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            getCameraPermissions()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
