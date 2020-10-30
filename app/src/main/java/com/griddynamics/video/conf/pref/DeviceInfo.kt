package com.griddynamics.video.conf.pref

import com.chibatching.kotpref.KotprefModel

object DeviceInfo : KotprefModel() {
    var uuid by stringPref()
}