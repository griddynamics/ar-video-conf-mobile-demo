package com.griddynamics.video.conf

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.SeekBar
import android.widget.SeekBar.OnSeekBarChangeListener
import android.widget.TextView
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
        ).apply {
            cbSendStatistic.isChecked = Settings.sendStatic
            cbSendStatistic.setOnCheckedChangeListener { _, isChecked ->
                Settings.sendStatic = isChecked
                logger.logsEnabled = isChecked
            }

            val modelsList =
                ImageSegmentationFactory.MaskModel.values().map { it.filename }
            ArrayAdapter(
                context,
                android.R.layout.simple_spinner_item,
                modelsList
            ).also { adapter ->
                adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                spinner.adapter = adapter
            }
            spinner.setSelection(modelsList.indexOfFirst { it == Settings.modelName }, false)
            spinner.onItemSelectedListener =
                MyOnItemSelectedListener { _, _, position, _ ->
                    Settings.modelName = modelsList[position]
                    onModelChange.onModelChange()
                }

            handleRoundTitle(tvRound)
            sbRound.progress = (Settings.modelRound * sbRound.max).toInt()
            sbRound.setOnSeekBarChangeListener(MyOnSeekBarChangeListener { seekBar, progress, _ ->
                seekBar?.let {
                    Settings.modelRound = progress / it.max.toFloat()
                }
                handleRoundTitle(tvRound)
            })

            handleScaleTitle(tvScale)
            sbScale.progress = Settings.modelScale
            sbScale.setOnSeekBarChangeListener(MyOnSeekBarChangeListener { seekBar, progress, _ ->
                seekBar?.let {
                    Settings.modelScale = progress
                }
                handleScaleTitle(tvScale)
            })
            btnUndo.setOnClickListener {
                sbRound.progress = 0
                sbScale.progress = 256
                cbSendStatistic.isChecked = false
            }
        }

        return view
    }

    private fun handleRoundTitle(tv: TextView) {
        tv.text = when (Settings.modelRound) {
            0f -> "Round model value: not in use"
            else -> "Round model value: ${Settings.modelRound}"
        }
    }

    private fun handleScaleTitle(tv: TextView) {
        tv.text = "Model scaling:  ${Settings.modelScale}"
    }

    fun interface MyOnSeekBarChangeListener : OnSeekBarChangeListener {
        override fun onProgressChanged(
            seekBar: SeekBar?,
            progress: Int,
            fromUser: Boolean
        )

        override fun onStartTrackingTouch(seekBar: SeekBar?) {
        }

        override fun onStopTrackingTouch(seekBar: SeekBar?) {
        }
    }

    fun interface MyOnItemSelectedListener : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(
            parent: AdapterView<*>?,
            view: View?,
            position: Int,
            id: Long
        )

        override fun onNothingSelected(parent: AdapterView<*>?) {
        }
    }
}
