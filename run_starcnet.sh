# Demo for classifyng objects from a galaxy or list of galaxies.

DOWNLOAD=${1:-0}
STARTTIME=$(date +%s)

mkdir -p legus/tab_files
mkdir -p legus/frc_fits_files
mkdir -p model
mkdir -p data
mkdir -p output

if [ -d "data/raw_32x32" ]; then rm -Rf data/raw_32x32; fi

if [ $DOWNLOAD -eq 1 ]
then
    echo "downloading galaxy mosaics..."	
    wget -P legus/frc_fits_files -q --show-progress -i frc_fits_links.txt
    
    cd legus/frc_fits_files/
    ls ./*.tar.gz |xargs -n1 tar -xvzf
    rm -r ./*.tar.gz
    cd ../../ 
fi

bash create_dataset.sh 1
echo "classifying objects..."
python src/test_net.py \
                   --test-batch-size 64 \
                   --data_dir data/ \
                   --dataset raw_32x32\
                   --save_dir model/ \
                   --cuda  --gpu 0 \
                   --checkpoint starcnet.pth \

python src/preds2output.py

ENDTIME=$(date +%s)
echo "---------------------------------------------"
echo "| End of classification | time: ( $(($ENDTIME - $STARTTIME))s)"
echo "---------------------------------------------"
