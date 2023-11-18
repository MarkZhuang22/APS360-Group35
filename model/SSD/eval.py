from utils import *
from tqdm import tqdm
import sys
from pprint import PrettyPrinter
import argparse
import numpy as np
from pdb import set_trace
from config import Config
from dataload import retrieve_gt
from matplotlib import pyplot as plt
from datasets import FaceMaskDataset

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def plot_precision_recall_curve(precisions_dict, filename, key):
    fig = plt.figure(figsize=(10, 3))
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thresholds = ["0.50", "0.70", "0.90"]

    for i, threshold in enumerate(thresholds):
        plt.subplot(int(f"13{i+1}"))
        if(key == "no_mask"):
            precisions_dict[threshold][0] = precisions_dict[threshold][0].cpu().numpy()
        else:
            precisions_dict[threshold][1] = precisions_dict[threshold][1].cpu().numpy()
        label_ = "threshold_" + threshold
        plt.step(x, precisions_dict[threshold][0], label=label_)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        print("Plotted figure for threshold", threshold)

    fig.tight_layout()
    print("Saving to", filename)
    fig.savefig(filename)
    print("Saved!")
    plt.close()

# Heavily modified from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """
    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    
    precisions_dict = {}
    APs_dict = {}
    mAPs = {}

    with torch.no_grad():
        # Batches
        #for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
        for i, (images, boxes, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)
            
            #set_trace()
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            #difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            #true_difficulties.extend(difficulties)

        # Calculate mAP
        # APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    
        for threshold in np.arange(0.5, 0.95, 0.05):  
            precisions, APs, mAP, _, _, _ = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, threshold)
            threshold = "%.2f" %threshold
            
            mAPs[threshold] = mAP 
            precisions_dict[threshold] = precisions
            APs_dict[threshold] = APs
        


    mAP_50, mAP_70, mAP_90 = mAPs["0.50"], mAPs["0.70"], mAPs["0.90"]
    mean_mAPs = sum(mAPs.values()) / len(mAPs)

    AP_50_no_mask, AP_50_with_mask = APs_dict["0.50"]["no_mask"], APs_dict["0.50"]["mask"]
    AP_70_no_mask, AP_70_with_mask = APs_dict["0.70"]["no_mask"], APs_dict["0.70"]["mask"]
    AP_90_no_mask, AP_90_with_mask = APs_dict["0.90"]["no_mask"], APs_dict["0.90"]["mask"]

    print(f"\nMean Average Precision (mAP@0.50): {mAP_50:.3f}")
    print(f"\nMean Average Precision (mAP@0.70): {mAP_70:.3f}")
    print(f"\nMean Average Precision (mAP@0.90): {mAP_90:.3f}")

    print(f"\nMean Average Precision (mAP@[0.50:0.95]): {mean_mAPs:.3f}")

    print(f"\nAPs[No Mask] (AP@0.50): {AP_50_no_mask:.3f}, APs[With Mask] (AP@0.50): {AP_50_with_mask:.3f}")
    print(f"\nAPs[No Mask] (AP@0.70): {AP_70_no_mask:.3f}, APs[With Mask] (AP@0.70): {AP_70_with_mask:.3f}")
    print(f"\nAPs[No Mask] (AP@0.90): {AP_90_no_mask:.3f}, APs[With Mask] (AP@0.90): {AP_90_with_mask:.3f}")

    mean_APs = [sum(values[i] for values in APs_dict.values()) / len(APs_dict) for i in ["no_mask", "mask"]]
    print(f"\nAPs[No Mask] (AP@[0.50:0.95]): {mean_APs[0]:.3f}, APs[With Mask] (AP@[0.50:0.95]): {mean_APs[1]:.3f}")

    _, _, _, cumul_tps, cumul_fps, n_objects_class = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, 0.5)

    print("n_objects (no_mask): ", n_objects_class[0])
    print("tp (no_mask): ", cumul_tps[0][-1])
    print("fp (no_mask): ", cumul_fps[0][-1])

    print("n_objects (with mask): ", n_objects_class[1])
    print("tp (with mask): ", cumul_tps[1][-1])
    print("fp (with mask): ", cumul_fps[1][-1])

    plot_precision_recall_curve(precisions_dict, "P_R_curve_no_mask.png","no_mask")
    plot_precision_recall_curve(precisions_dict, "P_R_curve_with_mask.png","mask")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="FaceMaskDetection")
    parser.add_argument('--dest', type=str, default="./", help='path to dataset.')
    parser.add_argument('--limit', type=int, default=0, help='limit number of images.')
    parser.add_argument('--checkpoint', type=str, default="./checkpoint_ssd300.pth.tar", help='limit number of images.')
    
    args = parser.parse_args()
    config = Config()
    # Testing Phase
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                     
    checkpoint = args.checkpoint
    
    # Load model checkpoint that is to be evaluated
    try:
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        model = model.to(device)
    except:
        print("Train the model or direct to checkpoint path to start evaluate results")
        sys.exit()	
    #train(config, train_dataset)
    print("loading test images")

    images, bnd_boxes, labels = retrieve_gt(args.dest, "test", limit=args.limit)
    print("%d images has been retrieved" %len(images))
    # set_trace()
    print("finish loading images")
    test_dataset = FaceMaskDataset(images, bnd_boxes, labels, "test")
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                               collate_fn=test_dataset.collate_fn, num_workers=config.workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    
    
    
    # Switch to eval mode
    model.eval()
    evaluate(test_loader, model)

