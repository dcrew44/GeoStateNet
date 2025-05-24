"""
Simple prediction script for GeoStateNet.
Usage: python predict.py path/to/image.jpg
"""
import argparse
import torch
from PIL import Image
from torchvision.transforms import v2

# Import from the installed package
from state_classifier.models import build_state_classifier
from state_classifier.data.transforms import get_val_transforms
from state_classifier.utils.constants import get_state_index_to_abbrev

# Get state mappings
STATE_ABBREV = get_state_index_to_abbrev()
STATE_NAMES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming"
]


def load_model(checkpoint_path):
    """Load the trained model from checkpoint."""
    model = build_state_classifier(num_classes=50, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def predict_single_image(model, image_path, top_k=5):
    """Make prediction on a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_val_transforms()
    img_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    # Get top predictions
    topk_probs, topk_indices = torch.topk(probabilities, top_k)
    
    # Format results
    predictions = []
    for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
        predictions.append({
            'state': STATE_NAMES[idx],
            'abbrev': STATE_ABBREV[idx],
            'confidence': prob * 100
        })
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Predict US state from street view image')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth', 
                        help='Path to model checkpoint')
    parser.add_argument('--top-k', type=int, default=5, 
                        help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)
    
    # Make prediction
    print(f"Analyzing {args.image}...")
    predictions = predict_single_image(model, args.image, args.top_k)
    
    # Display results
    print("\nPredictions:")
    print("-" * 40)
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['state']} ({pred['abbrev']}) - {pred['confidence']:.1f}%")
    
    print(f"\nTop prediction: {predictions[0]['state']} with {predictions[0]['confidence']:.1f}% confidence")


if __name__ == "__main__":
    main()