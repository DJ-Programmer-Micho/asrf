from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC

# Load model
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m")

# Load tokenizer from your local vocab
tokenizer = Wav2Vec2CTCTokenizer("dataset/vocab/sample_vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# Load feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")

# Combine into processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Save locally
model.save_pretrained("./model/pretrained")
processor.save_pretrained("./model/pretrained")
