package com.griddynamics.video.conf

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import android.widget.SeekBar.OnSeekBarChangeListener
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.gms.tasks.Tasks
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.google.firebase.ktx.Firebase
import com.google.firebase.storage.StorageMetadata
import com.google.firebase.storage.StorageReference
import com.google.firebase.storage.ktx.storage
import com.griddynamics.video.conf.SettingsBottomDialogFragment.MyOnSeekBarChangeListener
import com.griddynamics.video.conf.adapters.ModelsAdapter
import com.griddynamics.video.conf.data.Model
import com.griddynamics.video.conf.pref.Settings
import kotlinx.android.synthetic.main.fragment_settings_bottom_dialog.view.*
import java.io.File

class SettingsBottomDialogFragment(
    private val logger: Logger,
    private val onModelChange: OnModelChange
) : BottomSheetDialogFragment() {
    interface OnModelChange {
        fun onModelChange()
    }

    private lateinit var mAdapter: ModelsAdapter

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
            mAdapter = ModelsAdapter(emptyList(), object : ModelsAdapter.ModelListener {
                override fun onChecked(item: Model) {
                    Settings.modelName = item.name
                    Settings.modelSize = item.size
                    mAdapter.items = mAdapter.items.map { it.isSelected = it == item; it }
                    onModelChange.onModelChange()
                }

                override fun onDownloadToggle(item: Model) {
                    if (item.downloaded == true) {
                        File(context.filesDir, item.name).delete()
                        mAdapter.items = mAdapter.items.map {
                            if (it == item) {
                                it.downloaded = false
                            }; it
                        }
                        if (item.storageRef != null && item.isSelected) onChecked(mAdapter.items.first())
                    } else {
                        mAdapter.items = mAdapter.items.map {
                            if (it == item) {
                                it.inProgress = true
                            }; it
                        }

                        item.storageRef?.let { download(it) }
                    }
                }
            }, context)
            recycler.apply {
                layoutManager = LinearLayoutManager(activity)
                adapter = mAdapter
            }
        }

        listAll()

        return view
    }

    private fun handleRoundTitle(tv: TextView) {
        tv.text = when (Settings.modelRound) {
            0f -> "Threshold: not in use"
            else -> "Threshold: ${Settings.modelRound}"
        }
    }

    private fun handleScaleTitle(tv: TextView) {
        tv.text = "Model scaling:  ${Settings.modelScale}"
    }

    private fun listAll() {
        val listRef = Firebase.storage.reference.child("models")
        val listPageTask = listRef.listAll().continueWithTask { task ->
            val listOfTasks =
                task.result?.items?.map { item -> item.metadata }
            Tasks.whenAllComplete(listOfTasks)
        }
        listPageTask.addOnCompleteListener {

            it.result?.mapNotNull { ref -> (ref.result as StorageMetadata) }?.let { items ->
                onAllListed(items)
            }
        }.addOnFailureListener {
            Toast.makeText(context, it.message, Toast.LENGTH_LONG).show()
            onAllListed()
        }
    }

    private fun onAllListed(items: List<StorageMetadata> = emptyList()) {
        val allModels = ImageSegmentationFactory.MaskModel.values()
            .map {
                Model(
                    it.filename,
                    it.imageSize(),
                    downloaded = null,
                    isSelected = Settings.modelName == it.filename,
                    null
                )
            }
            .toMutableList()
        items.forEach {
            it.getCustomMetadata("size")?.toInt()?.let { size ->
                allModels.add(
                    Model(
                        it.reference?.name ?:"",
                        size,
                        downloaded = context?.let { context ->
                            File(
                                context.filesDir,
                                it.name?:""
                            ).exists()
                        } ?: false,
                        isSelected = Settings.modelName == it.name,
                        it.reference
                    ))
            }
        }
        mAdapter.items = allModels
        view?.progress?.visibility = View.GONE
    }

    fun download(storageRef: StorageReference) {
        context?.let { context ->
            val localFile = File(context.filesDir, storageRef.name)

            storageRef.getFile(localFile).addOnSuccessListener {
                onDownloaded(localFile.name)
            }.addOnFailureListener {
                Toast.makeText(context, it.message, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun onDownloaded(name: String) {
        mAdapter.items = mAdapter.items.map {
            if (it.name == name) {
                it.inProgress = false
                it.downloaded = true
            }; it
        }
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
}
