# auto-punctuation
Итоговая модель BiLSTM с attention вектора fasttext с http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html#fasttext, информация обрабатывалась на уровне слов и символов.

python train.py -d "texts.txt" --epochs 10 --batch-size 10 --train True
