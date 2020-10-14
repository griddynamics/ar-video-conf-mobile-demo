package com.griddynamics.video.conf.utils

import android.os.SystemClock
import android.view.View

fun View.safeClick(listener: View.OnClickListener, blockInMillis: Long = 1000) {
    var lastClickTime: Long = 0
    this.setOnClickListener {
        if (SystemClock.elapsedRealtime() - lastClickTime < blockInMillis) return@setOnClickListener
        lastClickTime = SystemClock.elapsedRealtime()
        listener.onClick(this)
    }
}