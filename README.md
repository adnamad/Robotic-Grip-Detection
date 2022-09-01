# Robotic-Gripping-Detection

- # To Run the code

  1. Download the weight file from the Drive link and put it in the weights folder
  2. Put the data folder from the challenge in the same directory as this code.
  3. To train a new model or restart from checkpoint -

  ```
   python train.py --wb 0 --save_chkpt "../tst_wts"
   --load_chkpt "../weights/maskrcnn_53.pt" --output_viz "../tst-outs"

  ```

  4. To run the code on the Test set and generate heatmap outputs

  ```
   python infer.py --wb 0 --load_chkpt "../weights/maskrcnn_53.pt" --output_viz "../tst-infer"
  ```

- ## Improvements
