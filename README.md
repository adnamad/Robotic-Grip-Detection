# robotic-gripping-detection

Train

    python train.py --wb 0 --save_chkpt "../tst_wts" --load_chkpt "../weights/maskrcnn_53.pt" --output_viz "../tst-outs"

Test

    python infer.py --wb 0 --load_chkpt "../weights/maskrcnn_53.pt" --output_viz "../tst-infer"
