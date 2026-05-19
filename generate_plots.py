import os
import copy
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from model import SimpleMLP
from data import get_mnist_dataloaders
from train import train_model, evaluate_model
from unlearn import unlearn

def plot_and_save(x_data, retain_y, forget_y, x_label, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x_data, retain_y, marker='o', label='Retain Accuracy', color='blue', linewidth=2)
    plt.plot(x_data, forget_y, marker='X', label='Forget Accuracy (Target: 3)', color='red', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(100, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel(x_label)
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot to {filename}")

def exp_samples_profiling(base_model, val_pack, device, forget_digit):
    print("\n--- Running Exp 1: Number of Samples for Profiling ---")
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    
    # Base dataset from forget_loader
    forget_dataset = forget_loader.dataset
    
    sample_sizes = [10, 50, 100, 250, 500, len(forget_dataset)]
    ret_accs, forg_accs = [], []
    
    for size in sample_sizes:
        print(f"  Testing with profiling subset size: {size}")
        # Create subset dataloader
        subset_indices = list(range(size))
        subset_dataset = Subset(forget_dataset, subset_indices)
        sub_forget_loader = DataLoader(subset_dataset, batch_size=64, shuffle=False)
        
        # Reset model
        model_copy = copy.deepcopy(base_model)
        
        # Unlearn
        model_copy = unlearn(model_copy, sub_forget_loader, device, penalty_scale=0.1, threshold_percentile=90, iter_times=5)
        
        # Evaluate
        ret_acc = evaluate_model(model_copy, test_retain_loader, device, name="silent")
        forg_acc = evaluate_model(model_copy, test_forget_loader, device, name="silent")
        
        ret_accs.append(ret_acc)
        forg_accs.append(forg_acc)
        
    plot_and_save(sample_sizes, ret_accs, forg_accs, 
                  'Number of Samples in Profiling Forward Pass',
                  'Impact of Profiling Sample Size on Unlearning',
                  'plots/exp1_sample_size.png')

def exp_hidden_neurons(device, val_pack, forget_digit):
    print("\n--- Running Exp 2: Number of Neurons in Single Hidden Layer ---")
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    
    hidden_sizes = [32, 64, 128, 256, 512, 1024]
    ret_accs, forg_accs = [], []
    
    for hs in hidden_sizes:
        print(f"  Training and unlearning for Hidden Size: {hs}")
        model = SimpleMLP(hidden_sizes=[hs]).to(device)
        model = train_model(model, train_loader, device, epochs=2) # 2 epochs to save time
        
        model = unlearn(model, forget_loader, device, penalty_scale=0.1, threshold_percentile=90, iter_times=5)
        
        ret_acc = evaluate_model(model, test_retain_loader, device, name="silent")
        forg_acc = evaluate_model(model, test_forget_loader, device, name="silent")
        
        ret_accs.append(ret_acc)
        forg_accs.append(forg_acc)
        
    plot_and_save(hidden_sizes, ret_accs, forg_accs, 
                  'Number of Hidden Neurons',
                  'Model Capacity vs Unlearning Effectiveness',
                  'plots/exp2_hidden_size.png')

def exp_iterations(base_model, val_pack, device, forget_digit):
    print("\n--- Running Exp 3: Number of Unlearning Iterations ---")
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    
    iterations_list = [1, 2, 3, 5, 10, 15]
    ret_accs, forg_accs = [], []
    
    for iters in iterations_list:
        print(f"  Testing with iterations: {iters}")
        model_copy = copy.deepcopy(base_model)
        
        model_copy = unlearn(model_copy, forget_loader, device, penalty_scale=0.1, threshold_percentile=90, iter_times=iters)
        
        ret_acc = evaluate_model(model_copy, test_retain_loader, device, name="silent")
        forg_acc = evaluate_model(model_copy, test_forget_loader, device, name="silent")
        
        ret_accs.append(ret_acc)
        forg_accs.append(forg_acc)
        
    plot_and_save(iterations_list, ret_accs, forg_accs, 
                  'Number of Unlearning Iterations',
                  'Unlearning Depth vs Accuracy',
                  'plots/exp3_iterations.png')

def exp_percentile(base_model, val_pack, device, forget_digit):
    print("\n--- Running Exp 4: Targeting Percentile Threshold ---")
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    
    # 70% means targeting top 30% of neurons. 99% means top 1%.
    percentiles = [50, 70, 80, 90, 95, 99] 
    ret_accs, forg_accs = [], []
    
    for perc in percentiles:
        print(f"  Testing with percentile threshold: {perc}%")
        model_copy = copy.deepcopy(base_model)
        
        model_copy = unlearn(model_copy, forget_loader, device, penalty_scale=0.1, threshold_percentile=perc, iter_times=5)
        
        ret_acc = evaluate_model(model_copy, test_retain_loader, device, name="silent")
        forg_acc = evaluate_model(model_copy, test_forget_loader, device, name="silent")
        
        ret_accs.append(ret_acc)
        forg_accs.append(forg_acc)
        
    plot_and_save(percentiles, ret_accs, forg_accs, 
                  'Target Threshold Percentile (%)',
                  'Percentile Threshold vs Unlearning Quality',
                  'plots/exp4_percentile.png')

def exp_penalty_scale(base_model, val_pack, device, forget_digit):
    print("\n--- Running Exp 5: Penalty Scale Factor ---")
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    
    # 0.0 effectively zeros the weights. 0.9 shrinks them slightly.
    penalty_scales = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    ret_accs, forg_accs = [], []
    
    for scale in penalty_scales:
        print(f"  Testing with penalty scale: {scale}")
        model_copy = copy.deepcopy(base_model)
        
        model_copy = unlearn(model_copy, forget_loader, device, penalty_scale=scale, threshold_percentile=90, iter_times=5)
        
        ret_acc = evaluate_model(model_copy, test_retain_loader, device, name="silent")
        forg_acc = evaluate_model(model_copy, test_forget_loader, device, name="silent")
        
        ret_accs.append(ret_acc)
        forg_accs.append(forg_acc)
        
    plot_and_save(penalty_scales, ret_accs, forg_accs, 
                  'Penalty Scale Factor (0 = prune completely, 1 = no change)',
                  'Penalty Aggressiveness vs Unlearning Quality',
                  'plots/exp5_penalty_scale.png')

def main():
    os.makedirs('plots', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forget_digit = 3
    
    print("Preparing Dataloaders...")
    val_pack = get_mnist_dataloaders(batch_size=64, forget_digit=forget_digit)
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    
    # Train a strong baseline model once to reuse for Exps 1, 3, 4, 5
    print("\nTraining Baseline Reference Model (hidden=256)...")
    base_model = SimpleMLP(hidden_sizes=[256]).to(device)
    base_model = train_model(base_model, train_loader, device, epochs=3)
    
    print("\nBaseline Model Accuracy:")
    evaluate_model(base_model, test_loader, device, name="Base Overall")
    
    # Provide the experiments
    exp_samples_profiling(base_model, val_pack, device, forget_digit)
    exp_iterations(base_model, val_pack, device, forget_digit)
    exp_percentile(base_model, val_pack, device, forget_digit)
    exp_penalty_scale(base_model, val_pack, device, forget_digit)
    
    # Run structural experiment last (requires training multiple models)
    exp_hidden_neurons(device, val_pack, forget_digit)
    
    print("\nAll experiments complete. Check the 'plots' directory.")

if __name__ == "__main__":
    import logging
    # Optional: Suppress standard prints inside loops if they get noisy
    main()
