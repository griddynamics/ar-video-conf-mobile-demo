package com.griddynamics.video.conf

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MenuActivity : AppCompatActivity() {
    private val REQUEST_CODE_PERMISSIONS = 1001
    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.READ_EXTERNAL_STORAGE,
        Manifest.permission.WRITE_EXTERNAL_STORAGE
    )

    var btnFaceDetection: Button? = null
    var btnPalmDetection: Button? = null
    var btnCombined: Button? = null
    var btnCustom: Button? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_menu)
        setupViews()
        checkRequiredPermissions()
    }

    private fun setupViews() {
        btnFaceDetection = findViewById(R.id.btn_face_detection)
        btnFaceDetection?.let {
            it.setOnClickListener {
                val intent = Intent(this, MainActivity::class.java)
                startActivity(intent)
            }
        }

        btnPalmDetection = findViewById(R.id.btn_palm_detection)
        btnPalmDetection?.let {
            it.setOnClickListener {
                val intent = Intent(this, PalmDetectionActivity::class.java)
                startActivity(intent)
            }
        }

        btnCombined = findViewById(R.id.btn_video)
        btnCombined?.let {
            it.setOnClickListener {
                val intent = Intent(this, VideoActivity::class.java)
                startActivity(intent)
            }
        }

        btnCustom = findViewById(R.id.btn_custom)
        btnCustom?.let {
            it.setOnClickListener {
                val intent = Intent(this, CustomModelActivity::class.java)
                startActivity(intent)
            }
        }
    }

    private fun checkRequiredPermissions() {
        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

}
