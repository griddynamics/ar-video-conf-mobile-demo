package com.griddynamics.video.conf.data

import com.google.firebase.storage.StorageReference

data class Model(
    val name: String,
    val size: Int,
    var downloaded: Boolean?,
    var isSelected: Boolean,
    val storageRef: StorageReference?,
    var inProgress: Boolean = false
    )