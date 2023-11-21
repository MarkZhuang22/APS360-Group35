python SSD/train.py --dest=./ --limit=800
python SSD/eval.py --checkpoint=checkpoint_ssd300_no_change_limi800.pth.tar
python SSD/detect.py --image=./test_photo/test1.png