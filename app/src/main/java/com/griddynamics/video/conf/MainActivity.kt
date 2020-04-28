package com.griddynamics.video.conf

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {
    private val REQUEST_CODE_PERMISSIONS = 100
    private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    private lateinit var cameraFragment: CameraFragment5
    private var lensFacing = CameraCharacteristics.LENS_FACING_FRONT

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)
        supportActionBar?.setDisplayHomeAsUpEnabled(false)
        requestPermissionsIfNeeded()
    }

    private fun requestPermissionsIfNeeded() {
        if (allPermissionsGranted()) {
            addCameraFragment()
        } else {

            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }
    }

    private fun addCameraFragment() {
        cameraFragment = CameraFragment5.newInstance(lensFacing)
        supportFragmentManager.popBackStack()
        supportFragmentManager.beginTransaction()
            .replace(R.id.view_finder, cameraFragment)
            .commit()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }
}
