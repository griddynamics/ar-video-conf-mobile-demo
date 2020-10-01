package com.griddynamics.video.conf.pref

import com.chibatching.kotpref.KotprefModel
import com.griddynamics.video.conf.ImageSegmentationFactory

object Settings : KotprefModel() {
    var sendStatic by booleanPref()
    var modelName by stringPref(default = ImageSegmentationFactory.MaskModel.mask32Model.toString())
}