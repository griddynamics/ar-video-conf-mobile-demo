package com.griddynamics.video.conf.calculator

import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList
import android.R.bool
import android.R.attr.x
import android.R.attr.x
import com.griddynamics.video.conf.calculator.HandGesture.FIST
import com.griddynamics.video.conf.calculator.HandGesture.FOUR
import com.griddynamics.video.conf.calculator.HandGesture.ONE
import com.griddynamics.video.conf.calculator.HandGesture.OPEN_HAND
import com.griddynamics.video.conf.calculator.HandGesture.THREE
import com.griddynamics.video.conf.calculator.HandGesture.TWO
import com.griddynamics.video.conf.calculator.HandGesture.UNKNOWN

enum class HandGesture {
    UNKNOWN,
    FIST,
    OPEN_HAND,
    FOUR,
    THREE,
    TWO,
    ONE
}

class HandGestureCalculator {

    companion object {


    }

    fun detectGesture(multiHandLandmarkList: List<NormalizedLandmarkList>) : HandGesture {
        var thumbIsOpen = false
        var firstFingerIsOpen = false
        var secondFingerIsOpen = false
        var thirdFingerIsOpen = false
        var fourthFingerIsOpen = false
        if (multiHandLandmarkList.isEmpty() || multiHandLandmarkList[0].landmarkCount != 21)
            return UNKNOWN
        val landmarkList = multiHandLandmarkList[0].landmarkList

        var pseudoFixKeyPoint = landmarkList[2].x
        if (landmarkList[3].x < pseudoFixKeyPoint && landmarkList[4].x < pseudoFixKeyPoint) {
            thumbIsOpen = true
        }

        pseudoFixKeyPoint = landmarkList[6].y;
        if (landmarkList[7].y < pseudoFixKeyPoint && landmarkList[8].y < pseudoFixKeyPoint)
        {
            firstFingerIsOpen = true;
        }

        pseudoFixKeyPoint = landmarkList[10].y;
        if (landmarkList[11].y < pseudoFixKeyPoint && landmarkList[12].y < pseudoFixKeyPoint)
        {
            secondFingerIsOpen = true;
        }

        pseudoFixKeyPoint = landmarkList[14].y;
        if (landmarkList[15].y < pseudoFixKeyPoint && landmarkList[16].y < pseudoFixKeyPoint)
        {
            thirdFingerIsOpen = true;
        }

        pseudoFixKeyPoint = landmarkList[18].y;
        if (landmarkList[19].y < pseudoFixKeyPoint && landmarkList[20].y < pseudoFixKeyPoint)
        {
            fourthFingerIsOpen = true;
        }

        if (thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen) {
            return OPEN_HAND
        }

        if (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen) {
            return FOUR
        }

        if (thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen) {
            return THREE
        }

        if (thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen) {
            return TWO
        }

        if (!thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen) {
            return ONE
        }

        if (!thumbIsOpen && !firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen) {
            return FIST
        }

        return UNKNOWN
    }
}