for split in random block; do
    python src/svm_model.py --split $split
    python src/cnn_model.py --split $split
    python src/effnet_model.py --split $split
done
python src/benchmark.py
python src/report.py
python src/check_leak.py
