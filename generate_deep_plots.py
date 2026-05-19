import os
import copy
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from model import SimpleMLP
from data import get_mnist_dataloaders
from train import train_model, evaluate_model
from unlearn import unlearn

def plot_and_save(x_data, retain_y, forget_y, x_label, title, filename, constants_str, overall_y=None, x_ticks_labels=None):
    plt.figure(figsize=(9, 6))
    plt.plot(range(len(x_data)) if x_ticks_labels else x_data, retain_y, marker='o', label='Retain Accuracy', color='blue', linewidth=2)
    plt.plot(range(len(x_data)) if x_ticks_labels else x_data, forget_y, marker='X', label='Forget Accuracy', color='red', linewidth=2)
    
    if overall_y is not None:
        plt.plot(range(len(x_data)) if x_ticks_labels else x_data, overall_y, marker='s', label='Overall Accuracy', color='green', linewidth=2, linestyle='--')
        
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(100, color='gray', linestyle='--', alpha=0.5)
    
    if x_ticks_labels:
        plt.xticks(range(len(x_data)), x_ticks_labels, rotation=45)
    
    plt.xlabel(x_label)
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    
    # Add textbox with constants
    plt.text(0.02, 0.02, f'Fixed Hyperparams:\n{constants_str}', 
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
             
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot to {filename}")

def evaluate_all(model, test_retain_loader, test_forget_loader, test_loader, device):
    ret_acc = evaluate_model(model, test_retain_loader, device, name="silent")
    forg_acc = evaluate_model(model, test_forget_loader, device, name="silent")
    over_acc = evaluate_model(model, test_loader, device, name="silent")
    return ret_acc, forg_acc, over_acc

def exp_samples_profiling(base_model, val_pack, device, forget_digit):
    print("\n--- Running Exp 1: Number of Samples for Profiling ---")
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    forget_dataset = forget_loader.dataset
    
    sample_sizes = [10, 50, 100, 250, 500, len(forget_dataset)]
    ret_accs, forg_accs, over_accs = [], [], []
    
    for size in sample_sizes:
        print(f"  Testing size: {size}")
        subset_dataset = Subset(forget_dataset, list(range(size)))
        sub_forget_loader = DataLoader(subset_dataset, batch_size=64, shuffle=False)
        base_model.hidden_activations = None
        model_copy = copy.deepcopy(base_model)
        model_copy = unlearn(model_copy, sub_forget_loader, device, penalty_scale=0.1, threshold_percentile=90, iter_times=5)
        
        r, f, o = evaluate_all(model_copy, test_retain_loader, test_forget_loader, test_loader, device)
        ret_accs.append(r); forg_accs.append(f); over_accs.append(o)
        
    plot_and_save(sample_sizes, ret_accs, forg_accs, 'Profiling Sample Size',
                  'Impact of Profiling Sample Size (Deep Model)', 'plots/deep_exp1_samples.png',
                  'model=[1024,512,256], iter=5, perc=90%, pen=0.1', over_accs)

def exp_architectures(device, val_pack, forget_digit):
    print("\n--- Running Exp 2: Different Multi-Layer Architectures ---")
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    
    structures = [[256, 128], [512, 256], [1024, 512], [256, 128, 64], [512, 256, 128], [1024, 512, 256]]
    strs_labels = [str(s) for s in structures]
    ret_accs, forg_accs, over_accs = [], [], []
    
    for hs in structures:
        print(f"  Testing architecture: {hs}")
        model = SimpleMLP(hidden_sizes=hs).to(device)
        model = train_model(model, train_loader, device, epochs=2)
        model = unlearn(model, forget_loader, device, penalty_scale=0.1, threshold_percentile=90, iter_times=5)
        
        r, f, o = evaluate_all(model, test_retain_loader, test_forget_loader, test_loader, device)
        ret_accs.append(r); forg_accs.append(f); over_accs.append(o)
        
    plot_and_save(structures, ret_accs, forg_accs, 'Hidden Layer Sizes',
                  'Deep Model Capacity vs Unlearning', 'plots/deep_exp2_architectures.png',
                  'iter=5, perc=90%, pen=0.1', over_accs, x_ticks_labels=strs_labels)

def exp_iterations(base_model, val_pack, device, forget_digit):
    print("\n--- Running Exp 3: Unlearning Iterations ---")
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    iters_list = [1, 2, 3, 5, 10, 15]
    ret_accs, forg_accs, over_accs = [], [], []
    
    for iters in iters_list:
        print(f"  Testing iterations: {iters}")
        base_model.hidden_activations = None
        model_copy = copy.deepcopy(base_model)
        model_copy = unlearn(model_copy, forget_loader, device, penalty_scale=0.1, threshold_percentile=90, iter_times=iters)
        r, f, o = evaluate_all(model_copy, test_retain_loader, test_forget_loader, test_loader, device)
        ret_accs.append(r); forg_accs.append(f); over_accs.append(o)
        
    plot_and_save(iters_list, ret_accs, forg_accs, 'Unlearning Iterations',
                  'Iterations vs Unlearning Quality (Deep Model)', 'plots/deep_exp3_iters.png',
                  'model=[1024,512,256], samples=max, perc=90%, pen=0.1', over_accs)

def exp_percentile(base_model, val_pack, device, forget_digit):
    print("\n--- Running Exp 4: Percentile Threshold ---")
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    percs = [50, 70, 80, 90, 95, 99]
    ret_accs, forg_accs, over_accs = [], [], []
    
    for p in percs:
        print(f"  Testing percentile: {p}%")
        base_model.hidden_activations = None
        model_copy = copy.deepcopy(base_model)
        model_copy = unlearn(model_copy, forget_loader, device, penalty_scale=0.1, threshold_percentile=p, iter_times=5)
        r, f, o = evaluate_all(model_copy, test_retain_loader, test_forget_loader, test_loader, device)
        ret_accs.append(r); forg_accs.append(f); over_accs.append(o)
        
    plot_and_save(percs, ret_accs, forg_accs, 'Percentile Threshold (%)',
                  'Percentile Threshold vs Unlearning (Deep Model)', 'plots/deep_exp4_percentile.png',
                  'model=[1024,512,256], iter=5, samples=max, pen=0.1', over_accs)

def exp_penalty_scale(base_model, val_pack, device, forget_digit):
    print("\n--- Running Exp 5: Penalty Scale ---")
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    scales = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    ret_accs, forg_accs, over_accs = [], [], []
    
    for s in scales:
        print(f"  Testing scale: {s}")
        base_model.hidden_activations = None
        model_copy = copy.deepcopy(base_model)
        model_copy = unlearn(model_copy, forget_loader, device, penalty_scale=s, threshold_percentile=90, iter_times=5)
        r, f, o = evaluate_all(model_copy, test_retain_loader, test_forget_loader, test_loader, device)
        ret_accs.append(r); forg_accs.append(f); over_accs.append(o)
        
    plot_and_save(scales, ret_accs, forg_accs, 'Penalty Scale',
                  'Penalty Scale vs Unlearning (Deep Model)', 'plots/deep_exp5_penalty.png',
                  'model=[1024,512,256], iter=5, perc=90%, samples=max', over_accs)

def exp_target_classes(device):
    print("\n--- Running Exp 6: Unlearning Difficulty across Digits ---")
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ret_accs, forg_accs, over_accs = [], [], []
    
    # Train robust baseline on all data once.
    print("  Training universal baseline model for digit tests...")
    global_val_pack = get_mnist_dataloaders(batch_size=64, forget_digit=0)
    base_model = SimpleMLP(hidden_sizes=[1024, 512, 256]).to(device)
    base_model = train_model(base_model, global_val_pack[0], device, epochs=3)
    
    for d in digits:
        print(f"  Unlearning Digit {d}...")
        val_pack_d = get_mnist_dataloaders(batch_size=64, forget_digit=d)
        _, _, fd_loader, _, te_forg, te_ret = val_pack_d
        _, te_overall, _, _, _, _ = val_pack_d # use overall test set
        
        base_model.hidden_activations = None
        model_copy = copy.deepcopy(base_model)
        model_copy = unlearn(model_copy, fd_loader, device, penalty_scale=0.1, threshold_percentile=90, iter_times=5)
        
        r = evaluate_model(model_copy, te_ret, device, name="silent")
        f = evaluate_model(model_copy, te_forg, device, name="silent")
        o = evaluate_model(model_copy, te_overall, device, name="silent")
        
        ret_accs.append(r); forg_accs.append(f); over_accs.append(o)
        
    plot_and_save(digits, ret_accs, forg_accs, 'Target Forget Digit',
                  'Unlearning Quality across Different Digits', 'plots/deep_exp6_digits.png',
                  'model=[1024,512,256], iter=5, perc=90%, pen=0.1', over_accs, x_ticks_labels=[str(d) for d in digits])

def main():
    os.makedirs('plots', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Preparing Dataloaders...")
    val_pack = get_mnist_dataloaders(batch_size=64, forget_digit=3)
    
    print("\nTraining Deep Baseline Reference Model [1024, 512, 256]...")
    base_model = SimpleMLP(hidden_sizes=[1024, 512, 256]).to(device)
    base_model = train_model(base_model, val_pack[0], device, epochs=3)
    
    exp_samples_profiling(base_model, val_pack, device, 3)
    exp_iterations(base_model, val_pack, device, 3)
    exp_percentile(base_model, val_pack, device, 3)
    exp_penalty_scale(base_model, val_pack, device, 3)
    
    exp_architectures(device, val_pack, 3)
    exp_target_classes(device)
    print("\nDone!")

if __name__ == '__main__':
    main()
