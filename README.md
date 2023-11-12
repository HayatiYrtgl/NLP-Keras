# NLP-Keras
This code contains a series of classes used for training a language model on a text dataset. The functionalities are as follows:

1. **PreProcessing Class:**
   - `corpus_processing`: A function that reads the dataset and splits it into lines.
   - `tokenization`: A function that uses a Tokenizer to convert the text into numerical sequences and generates n-gram sequences.
   - `tokenizer_saver`: A function that saves the Tokenizer to a JSON file.

2. **GetTokenizer Class:**
   - `__init__`: A class used to load the Tokenizer from a JSON file.

3. **CreateModel Class:**
   - `__init__`: A class that creates, compiles, and trains the model. It supports both medium and large models.
   - `m_model`: A function that creates a medium-sized language model.
   - `l_model`: A function that creates a large-sized language model.
   - `compiler`: A function that compiles and trains the model.

These classes aim to collectively create a language model learned from a text dataset. After training, the model file (`new_model.h5`) and training history (`new_model.xlsx`) are saved.

If you encounter any issues while running this code or if you need assistance with another topic, feel free to ask.
