package com.griddynamics.video.conf

import com.google.firebase.firestore.FirebaseFirestore
import com.griddynamics.video.conf.pref.DeviceInfo
import com.griddynamics.video.conf.pref.Settings

class Logger(private val db: FirebaseFirestore) {
    var logsEnabled = false
    fun <K, V> log(hashMap: HashMap<K, V>, tag: String) {
        if (logsEnabled)
            db.collection(tag).add(hashMap)
    }

    fun <K> logInfo(tag: String, info: K) {
        if (logsEnabled)
            db.collection("logs").add(
                hashMapOf(
                    "uuid" to DeviceInfo.uuid,
                    "timestamp" to System.currentTimeMillis(),
                    "model" to Settings.modelName,
                    "tag" to tag,
                    "info" to info
                )
            )
    }
}