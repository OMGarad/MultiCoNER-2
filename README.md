# MultiCoNER-2
For the SemEval 12: MultiCoNER-2 task


###Running the Final Submission File
```
python Final_Submission_File.py --test coarse-en-dev.conll --iob_tagging coarse --gpus 1 --encoder_model xlm-roberta-base --model /home/omkar/coarse_ner_baseline_models/XLMR_Base/xlmr_base_eng_coarse_ner_e10/lightning_logs/version_0  --max_length 500
```

#### Training the model:
```
python train_model.py --train en-train90.conll --dev en-train10.conll --out_dir path/to/output/directory \
                      --iob_tagging coarse --model_name any_name_works --gpus 1 --epochs 2 \
                      --encoder_model hugging-face-encoder-model --batch_size 64 --lr 0.0001
```

#### Evaluating the model:
```
python evaluate.py --test en-dev.conll --out_dir path/to/output/directory/for/results --iob_tagging coarse \
                   --gpus 1 --encoder_model hugging-face-encoder-model  \
                   --model out_dir_path_from_training/model_name_from_training/lightning_logs/version_x --prefix any_prefix
```

#### Predicting the tags:
```
python predict_tags.py --test en-dev.conll --out_dir path/to/directory/for/predictions --iob_tagging coarse \
                    --gpus 1 --encoder_model hugging-face-encoder-model \
                     --model out_dir_path_from_training/model_name_from_training/lightning_logs/version_x \
                     --prefix any_prefix --max_length 500
```

#### Installing all dependencies:
```
pip3 install -r requirements.txt
```
