import torch
from model import SimpleMLP
from data import get_mnist_dataloaders
from train import train_model, evaluate_model
from unlearn import unlearn

def run_experiment(hidden_sizes, device, val_pack, forget_digit):
    train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader = val_pack
    print(f'\n========================================')
    print(f'Running Experiment for Hidden Layers: {hidden_sizes}')
    print(f'========================================')
    model = SimpleMLP(hidden_sizes=hidden_sizes).to(device)
    model = train_model(model, train_loader, device, epochs=3)
    
    base_over = evaluate_model(model, test_loader, device, name='Overall Test Set')
    base_ret = evaluate_model(model, test_retain_loader, device, name='Retain Set')
    base_forg = evaluate_model(model, test_forget_loader, device, name='Forget Set')
    
    model = unlearn(model, forget_loader, device, penalty_scale=0.1, threshold_percentile=90, iter_times=5)
    
    post_over = evaluate_model(model, test_loader, device, name='Overall Test Set')
    post_ret = evaluate_model(model, test_retain_loader, device, name='Retain Set')
    post_forg = evaluate_model(model, test_forget_loader, device, name='Forget Set')
    
    return {
        'Hidden Sizes': str(hidden_sizes),
        'Base Overall': base_over,
        'Base Retain': base_ret,
        'Base Forget': base_forg,
        'Post Overall': post_over,
        'Post Retain': post_ret,
        'Post Forget': post_forg
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forget_digit = 3
    val_pack = get_mnist_dataloaders(batch_size=64, forget_digit=forget_digit)
    
    structures = [
        [256, 128],
        [512, 256],
        [1024, 512],
        [256, 128, 64],
        [512, 256, 128],
        [1024, 512, 256]
    ]
    results = []
    
    for hs in structures:
        results.append(run_experiment(hs, device, val_pack, forget_digit))
        
    md_table = '\n## Multi-Layer Unlearning Results\n\n'
    md_table += 'Target unlearned digit: **3**\n\n'
    md_table += '| Hidden Layers | Base Overall | Base Retain | Base Forget | Post Overall | Post Retain | Post Forget |\n'
    md_table += '|---|---|---|---|---|---|---|\n'
    
    for r in results:
        md_table += f"| {r['Hidden Sizes']} | {r['Base Overall']:.2f}% | {r['Base Retain']:.2f}% | {r['Base Forget']:.2f}% | {r['Post Overall']:.2f}% | {r['Post Retain']:.2f}% | {r['Post Forget']:.2f}% |\n"
        
    print('\n\n' + md_table)
    with open('RESULTS.md', 'a') as f:
        f.write(md_table)

if __name__ == '__main__':
    main()
