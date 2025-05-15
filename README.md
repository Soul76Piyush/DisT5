# DisT5 🚨📉
**A Text-to-Text Transformer Model for Disaster Event Classification and Understanding**

DisT5 is a specialized NLP model based on T5 (Text-to-Text Transfer Transformer), pre-trained and fine-tuned on disaster-related datasets. It is capable of learning to classify, understand, and respond to emergency and disaster-related text. This project involves both pretraining from scratch using span corruption and fine-tuning for sequence classification.

---

## 🔍 Overview

The purpose of DisT5 is to assist in emergency and humanitarian response efforts by:
- Detecting and classifying disaster-related tweets or messages.
- Understanding the nature and impact of events.
- Improving situational awareness for first responders and analysts.

---

## 📁 Project Structure

```

dist5/
│
├── t5\_pretrain.py            # Pretraining script for T5 using span corruption
├── t5\_finetune.py            # Fine-tuning script on labeled disaster dataset
├── data/                     # Folder for datasets (CSV files)
├── models/                   # Pretrained and fine-tuned model checkpoints
├── logs/                     # TensorBoard logs and training logs
├── README.md                 # Project documentation
└── requirements.txt          # List of dependencies

````

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

Main libraries:

* `transformers`
* `datasets`
* `pytorch-lightning`
* `torch`
* `huggingface_hub`
* `scikit-learn`
* `tensorboard`

---

## 🚀 Pretraining (Masked Span Corruption)

```bash
python t5_pretrain.py --file_path data/disaster_raw.csv --output_dir models/pretrained
```

* Uses unsupervised learning with span corruption (similar to T5 pretraining).
* Generates corrupted input-output pairs with `<extra_id_n>` sentinel tokens.

You can customize the span length, input/output lengths, and batch sizes in the script.

---

## 🎯 Fine-tuning

```bash
python t5_finetune.py
```

* Uses Hugging Face Trainer for supervised sequence classification.
* Trained on a labeled disaster dataset (`kaikouraEarthquake`) with multiple class labels like:

  * `Irrelevant`, `Impact`, `Infrastructure_Damage`, `Volunteer_Support`, etc.

---

## 🧪 Evaluation

After fine-tuning, the model is evaluated using:

* **Accuracy**
* **F1-Score (macro, micro, weighted)**
* **Cohen’s Kappa**

Evaluation predictions and logs are saved to the output directory.

---

## 📊 TensorBoard

You can monitor training with:

```bash
tensorboard --logdir logs/
```

---

## 🤗 Model Hub Integration

The fine-tuned model is automatically pushed to Hugging Face Hub:

➡️ [View Model on Hugging Face](https://huggingface.co/rizvi-rahil786/t5-small-kaikouraEarthquake)

---

## 📈 Results (Example)

```text
Accuracy: 84.5%
F1 Score (macro): 83.9%
Cohen’s Kappa: 0.79
```

---

## ✍️ Contributing

Pull requests and issues are welcome!

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## 👤 Author

**Nishant Kumar**
[GitHub](https://github.com/Nishant0986) • [LinkedIn](https://linkedin.com/in/nishant-singh-a20296325/)

---

## 🆘 Acknowledgments

* Hugging Face Transformers and Datasets
* PyTorch Lightning
* [CrisisNLP Dataset](http://crisisnlp.qcri.org/)
* Open-source community

---

## ⭐️ If you find this project useful, please consider giving it a star!

```

Would you like me to also generate a `requirements.txt` for the project?
```
