package com.griddynamics.video.conf.adapters

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.widget.AppCompatButton
import androidx.appcompat.widget.AppCompatCheckBox
import androidx.recyclerview.widget.RecyclerView
import com.griddynamics.video.conf.R
import com.griddynamics.video.conf.data.Model
import kotlinx.android.synthetic.main.model_item.view.*

class ModelsAdapter(
    items: List<Model>,
    private val listener: ModelListener,
    private val context: Context
) :
    RecyclerView.Adapter<ModelsAdapter.ViewHolder>() {

    var items: List<Model> = items
        set(value) {
            field = value
            notifyDataSetChanged()
        }

    interface ModelListener {
        fun onChecked(item: Model)
        fun onDownloadToggle(item: Model)
    }

    override fun getItemCount(): Int {
        return items.size
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        return ViewHolder(LayoutInflater.from(context).inflate(R.layout.model_item, parent, false))
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(items[position])
    }

    inner class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        private val checkBox: AppCompatCheckBox = view.checkbox
        private val name: TextView = view.name
        private val btnDownload: AppCompatButton = view.btnDownload
        private val progress: ProgressBar = view.progress

        fun bind(model: Model) {
            name.text = model.name

            checkBox.setOnCheckedChangeListener(null)
            btnDownload.text = when (model.downloaded) {
                false -> {
                    checkBox.isEnabled = false
                    checkBox.isChecked = false
                    "Download"
                }
                else -> {
                    checkBox.isEnabled = true
                    checkBox.isChecked = model.isSelected
                    "Delete"
                }
            }

            if (model.storageRef == null) {
                btnDownload.visibility = View.GONE
                progress.visibility = View.GONE

            } else {
                if (model.inProgress) {
                    btnDownload.visibility = View.INVISIBLE
                    progress.visibility = View.VISIBLE
                } else {
                    btnDownload.visibility = View.VISIBLE
                    progress.visibility = View.GONE
                }
            }

            checkBox.setOnCheckedChangeListener { _, _ -> listener.onChecked(model) }
            btnDownload.setOnClickListener { listener.onDownloadToggle(model) }
        }
    }
}
