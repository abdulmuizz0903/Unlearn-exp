import torch
from model import SimpleMLP
from data import get_mnist_dataloaders
from train import train_model, evaluate_model
from unlearn import unlearn

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    forget_digit = 3  # The digit we want to unlearn
    
    print("\n--- Phase 1: Setup & Baseline Training ---")
    val_pack = get_mnist_dataloaders(batch_size=64, forget_digit=forget_digit)
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    
    model = SimpleMLP().to(device)
    
    # Train baseline model
    print("Training baseline model...")
    model = train_model(model, train_loader, device, epochs=3)
    
    print("\nBaseline Model Evaluation:")
    evaluate_model(model, test_loader, device, name="Overall Test Set")
    evaluate_model(model, test_retain_loader, device, name=f"Retain Set (Digits != {forget_digit})")
    evaluate_model(model, test_forget_loader, device, name=f"Forget Set (Digit == {forget_digit})")
    
    print("\n--- Phase 2 & 3: Activation Profiling and Unlearning ---")
    # Apply targeted iterative penalization based on activation profiling
    model = unlearn(model, forget_loader, device, penalty_scale=0.1, threshold_percentile=95, iter_times=10)
    
    print("\n--- Phase 4: Post-Unlearning Evaluation ---")
    evaluate_model(model, test_loader, device, name="Overall Test Set")
    evaluate_model(model, test_retain_loader, device, name=f"Retain Set (Digits != {forget_digit})")
    evaluate_model(model, test_forget_loader, device, name=f"Forget Set (Digit == {forget_digit})")

if __name__ == "__main__":
    main()
