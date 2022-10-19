python dnn_all.py -fd daylight ssi sprotein -fb sequence bbi bprotein -b 256
python dnn_all.py -fd daylight ssi sprotein -fb sequence bbi bprotein -b 128
python dnn_all.py -fd daylight ssi sprotein -fb sequence bbi bprotein -b 64

python dnn_all.py -fd daylight ssi sprotein -fb sequence bbi bprotein -d 0.1
python dnn_all.py -fd daylight ssi sprotein -fb sequence bbi bprotein -d 0.2
python dnn_all.py -fd daylight ssi sprotein -fb sequence bbi bprotein -d 0.3

python dnn_all.py -fd ssi sprotein -fb bbi bprotein
python dnn_all.py -fd daylight ssi -fb sequence bbi
python dnn_all.py -fd daylight sprotein -fb sequence bprotein

python dnn_all.py -fd daylight -fb sequence
python dnn_all.py -fd ssi -fb bbi
python dnn_all.py -fd sprotein -fb bprotein

