import torch
import argparse
from classifier import load_from_checkpoint
import json


def main():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(
        description="Predict with your Image classifier")
    parser.add_argument('image_path', action="store",
                        type=str, help='Path to the image to predict')
    parser.add_argument('checkpoint_path', action="store",
                        type=str, help='Path to the classifier checkpoint')
    parser.add_argument('--category_names', default="cat_to_name.json",
                        type=str, help='Path to the classifier checkpoint')
    parser.add_argument('--top_k', type=int, default=5,
                        help='The number of top predictions')
    parser.add_argument('--gpu', action='store_true', help='Predict on gpu')
    in_arg = parser.parse_args()

    # Use GPU if it's available
    if in_arg.gpu:
         if torch.cuda.is_available():
             print('Predict using gpu...')
             device = torch.device("cuda")
         else:
             print('Your system does not have gpu support, switching to cpu..')
             device = torch.device("cpu")
    else:
         print('Predict using cpu...')
         device = torch.device("cpu")
                    
            
            
   
    # LOAD 
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    classifier = load_from_checkpoint(in_arg.checkpoint_path, device)
    
    # PREDICT
    probs, classes = classifier.predict(in_arg.image_path, in_arg.top_k)
    class_indices = list(classifier.model.class_to_idx)
    image_labels = [cat_to_name[class_indices[e]] for e in classes]

    print('\n === Image Predictions ===')
    for prob, label in zip(probs, image_labels):
        print('{}: {}%'.format(label, round(prob * 100, 2)))
    

# Call to main function to run the program
if __name__ == "__main__":
    main()
