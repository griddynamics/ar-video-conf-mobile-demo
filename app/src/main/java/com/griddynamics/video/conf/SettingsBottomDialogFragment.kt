package com.griddynamics.video.conf

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.griddynamics.video.conf.pref.Settings
import kotlinx.android.synthetic.main.fragment_settings_bottom_dialog.view.*

class SettingsBottomDialogFragment(
    private val logger: Logger,
    private val onModelChange: OnModelChange
) : BottomSheetDialogFragment() {
    interface OnModelChange {
        fun onModelChange()
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(
            R.layout.fragment_settings_bottom_dialog, container,
            false
        )
        view.cbSendStatistic.isChecked = Settings.sendStatic
        view.cbSendStatistic.setOnCheckedChangeListener { _, isChecked ->
            Settings.sendStatic = isChecked
            logger.logsEnabled = isChecked
        }

        val modelsList =
            ImageSegmentationFactory.MaskModel.values().map { it.filename }
        ArrayAdapter(
            view.context,
            android.R.layout.simple_spinner_item,
            modelsList
        ).also { adapter ->
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            view.spinner.adapter = adapter
        }
        view.spinner.setSelection(modelsList.indexOfFirst { it == Settings.modelName }, false)
        view.spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                Settings.modelName = modelsList[position]
                onModelChange.onModelChange()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
            }

        }
        return view
    }
}
