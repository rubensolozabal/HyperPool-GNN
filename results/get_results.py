# Description: Get the results from the results folder
import argparse
import json

def main():
    # Hyperparameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)

    args = parser.parse_args()

    fielname = args.filename

    # Read json
    with open(fielname, 'r') as f:
        results = json.load(f)

    print('Test Accuracy: {:.2f}±{:.2f} \n\
Test AUC: {:.2f}±{:.2f} \n\
Test NMI: {:.2f}±{:.2f} \n\
Test ARI: {:.2f}±{:.2f} \n\
Test F1: {:.2f}±{:.2f} \n\
Test Recall@1: {:.2f}±{:.2f}\n\
Test Recall@2: {:.2f}±{:.2f}\n\
Duration: {:.2f}s \n'.
        format( 100*results['acc'],
                100*results['acc_std'],
                100*results['auc'],
                100*results['auc_std'],
                100*results['nmi'],
                100*results['nmi_std'],
                100*results['ari'],
                100*results['ari_std'],
                100*results['f1'],
                100*results['f1_std'],
                100*results['recall_1'],
                100*results['recall_1_std'],
                100*results['recall_2'],
                100*results['recall_2_std'],
                results['duration']))

if __name__ == '__main__':
    main()