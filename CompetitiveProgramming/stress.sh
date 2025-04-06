make main
python3 gen.py > bin/in.txt
while ./main < bin/in.txt; do
    echo "OK"
    python3 gen.py > bin/in.txt
done
