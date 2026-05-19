import torch
import copy
import matplotlib.pyplot as plt
import os
from run_deep_eval import run_experiment
from data import get_mnist_dataloaders
from model import SimpleMLP
from train import train_model, evaluate_model
from unlearn import unlearn

def plot_experiment(ax, x_values, overall, retain, forget, title, xlabel, xticks=None):
    ax.plot(x_values, overall, marker='o', label='Overall Acc', color='blue', linewidth=2)
    ax.plot(x_values, retain, marker='s', label='Retain Acc', color='green', linewidth=2)
    ax.plot(x_values, forget, marker='^', label='Forget Acc (8)', color='red', linewidth=2)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (%)')
    if xticks:
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels(xticks)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_ylim(-5, 105)

def main():
    os.makedirs('plots', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forget_digit = 8
    print(f"Loading data for target digit: {forget_digit}...")
    val_pack = get_mnist_dataloaders(batch_size=64, forget_digit=forget_digit)
    
    base_arch = [1024, 512, 256]
    
    # Train the base model ONCE for experiments 1, 2, and 3
    print(f"\nTraining BASE model {base_arch} once for HP sweeps...")
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    base_model = SimpleMLP(hidden_sizes=base_arch).to(device)
    base_model = train_model(base_model, train_loader, device, epochs=3)
    base_model.hidden_activations = None # Clear hook state for safe copying
    base_state = copy.deepcopy(base_model.state_dict())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Digit 8 Unlearning Resistance Analysis', fontsize=16, fontweight='bold')
    
    # --- Exp 1: Iterations ---
    print("\nRunning Exp 1: Iterations Sweep")
    iters_list = [1, 3, 5, 10, 15, 20]
    overall, retain, forget = [], [], []
    for it in iters_list:
        model = SimpleMLP(hidden_sizes=base_arch).to(device)
        model.load_state_dict(base_state)
        model = unlearn(model, forget_loader, device, penalty_scale=0.1, threshold_percentile=90, iter_times=it)
        
        o_acc = evaluate_model(model, test_loader, device, name=None)
        r_acc = evaluate_model(model, test_retain_loader, device, name=None)
        f_acc = evaluate_model(model, test_forget_loader, device, name=None)
        
        overall.append(o_acc)
        retain.append(r_acc)
        forget.append(f_acc)
        print(f" Iters={it} -> Forget Acc: {f_acc:.2f}%")
        
    plot_experiment(axes[0, 0], iters_list, overall, retain, forget, 
                    'Impact of Unlearning Iterations', 'Iterations')

    # --- Exp 2: Penalty Scale ---
    print("\nRunning Exp 2: Penalty Scale Sweep")
    penalties = [0.0, 0.05, 0.1, 0.2, 0.5]
    overall, retain, forget = [], [], []
    for p in penalties:
        model = SimpleMLP(hidden_sizes=base_arch).to(device)
        model.load_state_dict(base_state)
        model = unlearn(model, forget_loader, device, penalty_scale=p, threshold_percentile=90, iter_times=5)
        
        o_acc = evaluate_model(model, test_loader, device, name=None)
        r_acc = evaluate_model(model, test_retain_loader, device, name=None)
        f_acc = evaluate_model(model, test_forget_loader, device, name=None)
        
        overall.append(o_acc)
        retain.append(r_acc)
        forget.append(f_acc)
        print(f" Penalty={p} -> Forget Acc: {f_acc:.2f}%")
        
    plot_experiment(axes[0, 1], penalties, overall, retain, forget, 
                    'Impact of Penalty Scale', 'Penalty Scale')

    # --- Exp 3: Percentile Threshold ---
    print("\nRunning Exp 3: Percentile Threshold Sweep")
    thresholds = [70, 80, 90, 95, 99]
    overall, retain, forget = [], [], []
    for t in thresholds:
        model = SimpleMLP(hidden_sizes=base_arch).to(device)
        model.load_state_dict(base_state)
        model = unlearn(model, forget_loader, device, penalty_scale=0.1, threshold_percentile=t, iter_times=5)
        
        o_acc = evaluate_model(model, test_loader, device, name=None)
        r_acc = evaluate_model(model, test_retain_loader, device, name=None)
        f_acc = evaluate_model(model, test_forget_loader, device, name=None)
        
        overall.append(o_acc)
        retain.append(r_acc)
        forget.append(f_acc)
        print(f" Threshold={t} -> Forget Acc: {f_acc:.2f}%")
        
    plot_experiment(axes[1, 0], thresholds, overall, retain, forget, 
                    'Impact of Percentile Threshold', 'Percentile Threshold')

    # --- Exp 4: Architectures ---
    print("\nRunning Exp 4: Architecture Sweep")
    # Adding more single hidden layer architectures as requested
    archs = [[64], [128], [256], [512, 256], [1024, 512, 256]]
    arch_labels = ['Tiny', 'Small', 'Med-Single', 'Medium', 'Large']
    overall, retain, forget = [], [], []
    for arch in archs:
        # Architecture sweeps must fully re-train
        res = run_experiment(arch, device, val_pack, forget_digit=forget_digit)
        overall.append(res['Post Overall'])
        retain.append(res['Post Retain'])
        forget.append(res['Post Forget'])
        print(f" Arch={arch} -> Forget Acc: {res['Post Forget']:.2f}%")
        
    plot_experiment(axes[1, 1], range(len(archs)), overall, retain, forget, 
                    'Impact of Network Architecture', 'Architecture Size', xticks=arch_labels)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = 'plots/digit8_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to {save_path}")

if __name__ == '__main__':
    main()
