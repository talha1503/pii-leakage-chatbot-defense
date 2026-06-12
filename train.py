"""
Main Training Script
Orchestrates the complete experimental pipeline
"""

import os
import json
import torch
from config import Config
from data import DataGenerator
from model import ChatbotModel
from evaluation import ModelEvaluator
from pii_detector import PIIDetector
import warnings
warnings.filterwarnings('ignore')


def setup_directories(config: Config):
    """Create necessary directories"""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print("✓ Directories created")


def phase1_data_preparation(config: Config):
    """
    Phase 1: Dataset Synthesis and PII Injection
    """
    print("\n" + "="*70)
    print("PHASE 1: DATA PREPARATION")
    print("="*70)
    
    generator = DataGenerator(config)
    generator.save_datasets()
    
    print("\n✓ Phase 1 Complete: Datasets generated with PII injection")
    return generator


def phase2_baseline_training(config: Config):
    """
    Phase 2: Train baseline model (SFT without privacy protection)
    """
    print("\n" + "="*70)
    print("PHASE 2: BASELINE MODEL TRAINING (No Protection)")
    print("="*70)
    
    # Initialize model
    model = ChatbotModel(config)
    
    # Train with full dataset (including PII)
    train_path = f"{config.DATA_DIR}/train.json"
    val_path = f"{config.DATA_DIR}/val.json"
    output_dir = f"{config.MODEL_DIR}/baseline"
    
    model.train_sft(
        train_data_path=train_path,
        val_data_path=val_path,
        output_dir=output_dir,
        use_scrubbing=False
    )
    
    print("\n✓ Phase 2 Complete: Baseline model trained")
    return model


def phase3_sft_with_scrubbing(config: Config):
    """
    Phase 3: Train SFT model with PII scrubbing
    """
    print("\n" + "="*70)
    print("PHASE 3: SFT WITH PII SCRUBBING")
    print("="*70)
    
    # Initialize fresh model
    model = ChatbotModel(config)
    
    # Train with PII scrubbing enabled
    train_path = f"{config.DATA_DIR}/train.json"
    val_path = f"{config.DATA_DIR}/val.json"
    output_dir = f"{config.MODEL_DIR}/sft_scrubbed"
    
    model.train_sft(
        train_data_path=train_path,
        val_data_path=val_path,
        output_dir=output_dir,
        use_scrubbing=True
    )
    
    print("\n✓ Phase 3 Complete: SFT with scrubbing trained")
    return model


def phase4_dpo_training(config: Config):
    """
    Phase 4: Train DPO model with privacy preferences
    """
    print("\n" + "="*70)
    print("PHASE 4: DPO TRAINING (Privacy-Aware Preferences)")
    print("="*70)
    
    # Start from baseline or scrubbed model
    baseline_dir = f"{config.MODEL_DIR}/baseline"
    
    if os.path.exists(baseline_dir):
        print(f"Loading baseline model from {baseline_dir}")
        model = ChatbotModel(config)
        model.load_model(baseline_dir)
    else:
        print("Baseline not found, initializing fresh model")
        model = ChatbotModel(config)
    
    # Train with DPO
    dpo_data_path = f"{config.DATA_DIR}/dpo_pairs.json"
    output_dir = f"{config.MODEL_DIR}/dpo"
    
    model.train_dpo(
        dpo_data_path=dpo_data_path,
        output_dir=output_dir
    )
    
    print("\n✓ Phase 4 Complete: DPO model trained")
    return model


def phase5_grpo_training(config: Config):
    """
    Phase 5: Train GRPO model with ranked privacy rewards
    """
    print("\n" + "="*70)
    print("PHASE 5: GRPO TRAINING (Ranked Privacy Rewards)")
    print("="*70)
    
    # Start from baseline
    baseline_dir = f"{config.MODEL_DIR}/baseline"
    
    if os.path.exists(baseline_dir):
        print(f"Loading baseline model from {baseline_dir}")
        model = ChatbotModel(config)
        model.load_model(baseline_dir)
    else:
        print("Baseline not found, initializing fresh model")
        model = ChatbotModel(config)
    
    # Train with GRPO
    train_path = f"{config.DATA_DIR}/train.json"
    output_dir = f"{config.MODEL_DIR}/grpo"
    
    model.train_grpo(
        train_data_path=train_path,
        output_dir=output_dir
    )
    
    print("\n✓ Phase 5 Complete: GRPO model trained")
    return model


def phase6_evaluation(config: Config):
    """
    Phase 6: Comprehensive evaluation of all models
    """
    print("\n" + "="*70)
    print("PHASE 6: COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    evaluator = ModelEvaluator(config)
    test_data_path = f"{config.DATA_DIR}/test.json"
    
    # Models to evaluate
    models_to_eval = {
        "Baseline (No Protection)": f"{config.MODEL_DIR}/baseline",
        "SFT + Scrubbing": f"{config.MODEL_DIR}/sft_scrubbed",
        "DPO (Privacy-Aware)": f"{config.MODEL_DIR}/dpo",
        "GRPO (Ranked Rewards)": f"{config.MODEL_DIR}/grpo"
    }
    
    all_results = {}
    
    # Evaluate each model
    for model_name, model_path in models_to_eval.items():
        if not os.path.exists(model_path):
            print(f"\n⚠ Skipping {model_name} - model not found at {model_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model = ChatbotModel(config)
        model.load_model(model_path)
        
        # Comprehensive evaluation
        results = evaluator.evaluate_model_comprehensive(
            model=model,
            model_name=model_name,
            test_data_path=test_data_path
        )
        
        all_results[model_name] = results
    
    # Compare models
    if len(all_results) > 1:
        comparison = evaluator.compare_models(all_results)
        
        # Save comparison results
        with open(f"{config.RESULTS_DIR}/comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        
        # Generate visualizations
        evaluator.visualize_results(all_results, config.RESULTS_DIR)
    
    # Save detailed results
    with open(f"{config.RESULTS_DIR}/detailed_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✓ Phase 6 Complete: All models evaluated")
    return all_results


def phase7_failure_analysis(config: Config, all_results: dict):
    """
    Phase 7: Failure analysis and ablation studies
    """
    print("\n" + "="*70)
    print("PHASE 7: FAILURE ANALYSIS")
    print("="*70)
    
    failure_analysis = {
        "common_failure_patterns": [],
        "pii_types_most_leaked": {},
        "attack_vulnerabilities": {}
    }
    
    # Analyze which PII types are most frequently leaked
    pii_leak_counts = {}
    
    for model_name, results in all_results.items():
        privacy = results["privacy"]
        
        for pii_type, count in privacy["leakage_by_type"].items():
            if pii_type not in pii_leak_counts:
                pii_leak_counts[pii_type] = {}
            pii_leak_counts[pii_type][model_name] = count
    
    failure_analysis["pii_types_most_leaked"] = pii_leak_counts
    
    # Analyze attack vulnerabilities
    for model_name, results in all_results.items():
        attack_res = results["attack_resistance"]
        
        # Find most vulnerable attack type
        vulnerabilities = {
            "direct": attack_res["direct_attack"],
            "indirect": attack_res["indirect_attack"],
            "contextual": attack_res["contextual_attack"],
            "jailbreak": attack_res["jailbreak_attack"]
        }
        
        most_vulnerable = max(vulnerabilities, key=vulnerabilities.get)
        
        failure_analysis["attack_vulnerabilities"][model_name] = {
            "most_vulnerable_to": most_vulnerable,
            "success_rate": vulnerabilities[most_vulnerable]
        }
    
    # Print failure analysis
    print("\nMost Leaked PII Types (across all models):")
    total_leaks_by_type = {}
    for pii_type, model_counts in pii_leak_counts.items():
        total = sum(model_counts.values())
        total_leaks_by_type[pii_type] = total
    
    sorted_leaks = sorted(total_leaks_by_type.items(), key=lambda x: x[1], reverse=True)
    for pii_type, count in sorted_leaks:
        print(f"  {pii_type}: {count}")
    
    print("\nAttack Vulnerabilities by Model:")
    for model_name, vuln in failure_analysis["attack_vulnerabilities"].items():
        print(f"  {model_name}:")
        print(f"    Most vulnerable to: {vuln['most_vulnerable_to']}")
        print(f"    Success rate: {vuln['success_rate']:.2%}")
    
    # Save failure analysis
    with open(f"{config.RESULTS_DIR}/failure_analysis.json", "w") as f:
        json.dump(failure_analysis, f, indent=2)
    
    print("\n✓ Phase 7 Complete: Failure analysis completed")
    return failure_analysis


def generate_final_report(config: Config, all_results: dict, failure_analysis: dict):
    """
    Generate final comprehensive report
    """
    print("\n" + "="*70)
    print("GENERATING FINAL REPORT")
    print("="*70)
    
    report = []
    report.append("="*70)
    report.append("PII LEAKAGE ANALYSIS IN CHATBOT SETTINGS")
    report.append("Final Report")
    report.append("="*70)
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-"*70)
    report.append(f"Total models evaluated: {len(all_results)}")
    report.append("")
    
    # Find best model
    best_balanced = max(all_results, key=lambda x: all_results[x]["privacy_utility_score"])
    best_privacy = min(all_results, key=lambda x: all_results[x]["privacy"]["leakage_rate"])
    best_utility = max(all_results, key=lambda x: all_results[x]["utility"]["task_success_rate"])
    
    report.append(f"Best balanced model: {best_balanced}")
    report.append(f"  Privacy-Utility Score: {all_results[best_balanced]['privacy_utility_score']:.3f}")
    report.append(f"Best privacy: {best_privacy}")
    report.append(f"  Leakage Rate: {all_results[best_privacy]['privacy']['leakage_rate']:.2%}")
    report.append(f"Best utility: {best_utility}")
    report.append(f"  Task Success: {all_results[best_utility]['utility']['task_success_rate']:.2%}")
    report.append("")
    
    # Detailed Results
    report.append("DETAILED RESULTS BY MODEL")
    report.append("-"*70)
    
    for model_name, results in all_results.items():
        report.append(f"\n{model_name}:")
        report.append(f"  Privacy Metrics:")
        report.append(f"    Leakage Rate: {results['privacy']['leakage_rate']:.2%}")
        report.append(f"    Samples Tested: {results['privacy']['samples_tested']}")
        report.append(f"  Utility Metrics:")
        report.append(f"    Task Success Rate: {results['utility']['task_success_rate']:.2%}")
        report.append(f"    Avg Response Length: {results['utility']['avg_response_length']:.1f} words")
        report.append(f"  Attack Success Rate:")
        report.append(f"    Overall: {results['attack_resistance']['overall_success_rate']:.2%}")
        report.append(f"    Direct: {results['attack_resistance']['direct_attack']:.2%}")
        report.append(f"    Indirect: {results['attack_resistance']['indirect_attack']:.2%}")
        report.append(f"    Contextual: {results['attack_resistance']['contextual_attack']:.2%}")
        report.append(f"    Jailbreak: {results['attack_resistance']['jailbreak_attack']:.2%}")
        report.append(f"  Privacy-Utility Score: {results['privacy_utility_score']:.3f}")
    
    # Key Findings
    report.append("\n" + "="*70)
    report.append("KEY FINDINGS")
    report.append("-"*70)
    report.append("1. Privacy-Utility Trade-offs:")
    report.append("   - Defense mechanisms reduce PII leakage but may impact utility")
    report.append(f"   - Best balanced approach: {best_balanced}")
    report.append("")
    report.append("2. Most Vulnerable PII Types:")
    for pii_type, count in list(sorted(
        failure_analysis["pii_types_most_leaked"].items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    ))[:3]:
        total = sum(count.values())
        report.append(f"   - {pii_type}: {total} leakages")
    report.append("")
    report.append("3. Attack Effectiveness:")
    report.append("   Different defense strategies show varying resistance to attack types")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-"*70)
    report.append("1. Deploy privacy-aware training (DPO/GRPO) for production systems")
    report.append("2. Implement layered defenses combining scrubbing and RL techniques")
    report.append("3. Regular auditing and red-team testing for PII leakage")
    report.append("4. Context-aware PII handling for improved privacy-utility balance")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    with open(f"{config.RESULTS_DIR}/final_report.txt", "w") as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Report saved to {config.RESULTS_DIR}/final_report.txt")


def main():
    """Main execution pipeline"""
    print("\n" + "="*70)
    print("PII LEAKAGE ANALYSIS IN CHATBOT SETTINGS")
    print("CS690F: Trustworthy and Responsible AI")
    print("="*70)
    
    # Initialize config
    config = Config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.SEED)
    
    # Setup
    setup_directories(config)
    
    # Run all phases
    try:
        # Phase 1: Data Preparation
        generator = phase1_data_preparation(config)
        
        # Phase 2: Baseline Training
        baseline_model = phase2_baseline_training(config)
        
        # Phase 3: SFT with Scrubbing
        sft_model = phase3_sft_with_scrubbing(config)
        
        # Phase 4: DPO Training
        dpo_model = phase4_dpo_training(config)
        
        # Phase 5: GRPO Training
        grpo_model = phase5_grpo_training(config)
        
        # Phase 6: Evaluation
        all_results = phase6_evaluation(config)
        
        # Phase 7: Failure Analysis
        failure_analysis = phase7_failure_analysis(config, all_results)
        
        # Generate Final Report
        generate_final_report(config, all_results, failure_analysis)
        
        print("\n" + "="*70)
        print("✓ ALL PHASES COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {config.RESULTS_DIR}")
        print(f"Models saved to: {config.MODEL_DIR}")
        print(f"Data saved to: {config.DATA_DIR}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
