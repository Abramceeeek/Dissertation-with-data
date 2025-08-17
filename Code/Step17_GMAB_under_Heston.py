"""
GMAB Variable Annuity Analysis under Heston Model

This script implements the GMAB (Guaranteed Minimum Accumulation Benefit) product
under the Heston model for the dissertation:
"The Effects of Stochastic Volatility Models and Dynamic Hedging Strategies 
on Capital Requirements for Equity-Linked Variable Annuities: An Enterprise Risk Management Approach"

Author: Abdurakhmonbek Fayzullaev
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

# Paths for imports
sys.path.append('.')
sys.path.append('Code')

# Import configuration and modules
from Step00_Configuration import *
from Step33_GMAB_Product_Module import (
    GMABParams, evolve_account_from_prices, gmab_value_and_delta, gmab_maturity_payoff
)
from Step14_Heston_Simulation import simulate_heston

# Output directories
results_dir = "Output/results"
tables_dir = "Output/tables"
figures_dir = "Output/figures"
results_heston_file = f"{results_dir}/results_heston_gmab.csv"
table_heston_file = f"{tables_dir}/table_heston.tex"
figure_dpi = 300
figure_format = 'png'

def get_rebalance_times(N, rebalance_freq):
    """Get rebalancing time indices (ensure within bounds)."""
    times = list(range(0, N, rebalance_freq))
    if N not in times:
        times.append(N)
    return [min(t, N - 1) for t in times]

def format_risk_metrics(pnl_array):
    """Calculate and format risk metrics from P&L array (insurer perspective)."""
    mean_pnl = float(np.mean(pnl_array))
    std_pnl  = float(np.std(pnl_array))
    var_99_5 = float(np.percentile(pnl_array, 0.5))  # left 0.5% tail
    cte_99_5 = float(np.mean(pnl_array[pnl_array <= var_99_5])) if pnl_array.size else var_99_5
    min_pnl  = float(np.min(pnl_array))
    max_pnl  = float(np.max(pnl_array))
    return {
        'mean_pnl': mean_pnl,
        'std_pnl': std_pnl,
        'var_99_5': var_99_5,
        'cte_99_5': cte_99_5,
        'min_pnl': min_pnl,
        'max_pnl': max_pnl
    }

def load_market_data():
    """Load risk-free rate and dividend yield data (fallback to defaults)."""
    try:
        rf_data = pd.read_csv(riskfree_file)
        rf_data['maturity_days'] = rf_data['maturity_days'].astype(int)

        div_data = pd.read_csv(dividend_file)

        target_days = int(T * 365.25)
        closest = rf_data.iloc[(rf_data['maturity_days'] - target_days).abs().argsort()[:1]]
        r = float(closest['rate'].iloc[0]) / 100.0
        q = float(div_data['dividend_yield'].mean()) / 100.0
        return r, q
    except Exception as e:
        print(f"Warning: Could not load market data: {e}")
        print(f"Using default rates: r={r_default}, q={q_default}")
        return r_default, q_default

def run_gmab_simulation_heston():
    """Run GMAB simulation under Heston model with dynamic hedging."""
    print("=" * 60)
    print("GMAB Analysis under Heston Model")
    print("=" * 60)
    print(f"Contract Parameters: T={T} years, Guarantee={g_annual:.1%} p.a.")
    print(f"Simulation: {n_paths:,} paths, {int(N):,} time steps\n")

    # Set random seed
    if use_common_random_numbers:
        np.random.seed(seed)

    # Market rates
    r, q = load_market_data()
    print(f"Market rates: r={r:.3%}, q={q:.3%}")

    # GMAB parameters
    gmab_params = GMABParams(
        T_years=T,
        g_annual=g_annual,
        fee_annual=fee_annual,
        trans_cost_bps=trans_cost_bps,
        rebalance_freq=rebalance_freq
    )

    # Use S0 as initial premium/notional basis (consistent with your Step16)
    initial_premium = float(S0)
    print(f"GMAB fees: Management={fee_annual:.1%}, Transaction costs={trans_cost_bps} bps")
    print(f"Initial premium (notional basis): ${initial_premium:,.2f}\n")

    # Simulate Heston paths
    print("Simulating Heston paths...")
    mu = r - q  # risk-neutral drift
    S, V = simulate_heston(S0, mu, T, int(N), n_paths, heston_params, seed=seed)
    assert S.shape[0] == int(N) + 1 and S.shape[1] == n_paths, f"S wrong shape: {S.shape}"
    print(f"Initial price: ${S[0, 0]:.2f}")
    print(f"Average final price: ${np.mean(S[-1, :]):.2f}\n")

    # Evolve account values in DOLLARS (your existing function)
    print("Calculating account evolution...")
    A = evolve_account_from_prices(S, gmab_params.fee_annual, dt_years)

    # Convert to RATIO per unit premium for pricer/Greeks and payoff
    A_ratio = A / initial_premium

    # Rebalancing schedule
    print("Running dynamic hedging simulation...")
    rebalance_times = get_rebalance_times(int(N), gmab_params.rebalance_freq)
    n_rebalance = len(rebalance_times)
    max_valid_index = S.shape[0] - 1
    rebalance_times = [min(t, max_valid_index) for t in rebalance_times]
    print(f"Array dimensions: S {S.shape}, A {A.shape}")
    print(f"Rebalancing times (first 10): {rebalance_times[:10]} ... total={n_rebalance}")
    print(f"Max rebalancing index: {max(rebalance_times)}")

    # Storage
    hedge_pnl = np.zeros(n_paths)
    unhedged_pnl = np.zeros(n_paths)
    transaction_costs = np.zeros(n_paths)

    # Hedging loop
    for path in range(n_paths):
        if path % 1000 == 0:
            print(f"Processing path {path:,} / {n_paths:,}")

        path_hedge_pnl = 0.0
        path_trans_cost = 0.0

        # Start flat in SHARES and anchor price
        prev_shares = 0.0
        prev_S = S[rebalance_times[0], path]

        # Rebalance up to (but not including) maturity
        for i, t_idx in enumerate(rebalance_times[:-1]):
            t = t_idx * dt_years
            T_remain = max(T - t, 0.0)

            current_S = S[t_idx, path]
            current_A_ratio = A_ratio[t_idx, path]

            # GMAB value & delta PER UNIT premium (heston model)
            value_per_unit, delta_per_unit = gmab_value_and_delta(
                S=current_S,
                A=current_A_ratio,
                T=T_remain,
                r=r, q=q,
                gmab_params=gmab_params,
                model='heston',
                heston_params=heston_params
            )

            # Convert per-unit delta -> SHARES for trading
            shares_needed = delta_per_unit * initial_premium
            shares_change = shares_needed - prev_shares

            # Transaction cost on traded notional
            trade_notional = abs(shares_change) * current_S
            path_trans_cost += trade_notional * (trans_cost_bps / 10000.0)

            # Realize P&L on previously-held shares
            if i > 0:
                path_hedge_pnl += prev_shares * (current_S - prev_S)

            prev_shares = shares_needed
            prev_S = current_S

        # Final settlement at maturity
        final_S = S[-1, path]
        if len(rebalance_times) > 1:
            last_reb_idx = rebalance_times[-2]
            path_hedge_pnl += prev_shares * (final_S - S[last_reb_idx, path])

        # GMAB payoff at maturity (per-unit → dollars)
        final_A_ratio = A_ratio[-1, path]
        maturity_payoff = gmab_maturity_payoff(final_A_ratio, gmab_params, initial_premium)

        # Store P&L (insurer perspective)
        unhedged_pnl[path] = -maturity_payoff
        hedge_pnl[path]    = -maturity_payoff + path_hedge_pnl - path_trans_cost
        transaction_costs[path] = path_trans_cost

    print("\nSimulation completed. Calculating risk metrics...")
    unhedged_metrics = format_risk_metrics(unhedged_pnl)
    hedged_metrics   = format_risk_metrics(hedge_pnl)

    # Safe hedge effectiveness
    hedge_eff = (np.nan if abs(unhedged_metrics['std_pnl']) < 1e-12
                 else 1 - hedged_metrics['std_pnl'] / unhedged_metrics['std_pnl'])

    # Results dict
    results = {
        'Model': 'Heston',
        'Contract': 'GMAB',
        'Paths': n_paths,
        'T_years': T,
        'Guarantee_Rate': g_annual,
        'Management_Fee': fee_annual,
        'Transaction_Cost_bps': trans_cost_bps,
        'Rebalance_Freq_days': rebalance_freq,
        'Risk_Free_Rate': r,
        'Dividend_Yield': q,
        'Heston_v0': heston_params['v0'],
        'Heston_kappa': heston_params['kappa'],
        'Heston_theta': heston_params['theta'],
        'Heston_sigma_v': heston_params['sigma_v'],
        'Heston_rho': heston_params['rho'],
        # Unhedged metrics
        'Unhedged_Mean_PnL': unhedged_metrics['mean_pnl'],
        'Unhedged_Std_PnL':  unhedged_metrics['std_pnl'],
        'Unhedged_VaR_99_5': unhedged_metrics['var_99_5'],
        'Unhedged_CTE_99_5': unhedged_metrics['cte_99_5'],
        'Unhedged_Min_PnL':  unhedged_metrics['min_pnl'],
        'Unhedged_Max_PnL':  unhedged_metrics['max_pnl'],
        # Hedged metrics
        'Hedged_Mean_PnL':   hedged_metrics['mean_pnl'],
        'Hedged_Std_PnL':    hedged_metrics['std_pnl'],
        'Hedged_VaR_99_5':   hedged_metrics['var_99_5'],
        'Hedged_CTE_99_5':   hedged_metrics['cte_99_5'],
        'Hedged_Min_PnL':    hedged_metrics['min_pnl'],
        'Hedged_Max_PnL':    hedged_metrics['max_pnl'],
        # Other
        'Avg_Transaction_Cost': float(np.mean(transaction_costs)),
        'Hedge_Effectiveness': hedge_eff,
        'Seed': seed
    }

    # Brief debug summary (remove later if noisy)
    print(f"DEBUG: ITM fraction (payoff>0): {np.mean(unhedged_pnl < 0):.2%}")
    print(f"DEBUG: Unhedged std: {unhedged_metrics['std_pnl']:.2f}  |  Hedged std: {hedged_metrics['std_pnl']:.2f}")

    return results, unhedged_pnl, hedge_pnl, S, A

def save_results_and_plots_heston(results, unhedged_pnl, hedge_pnl, S, A):
    """Save simulation results, tables, and plots for Heston model."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # CSV
    pd.DataFrame([results]).to_csv(results_heston_file, index=False)
    print(f"Results saved to: {results_heston_file}")

    # LaTeX table
    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{GMAB Risk Metrics under Heston Model}}
\\label{{tab:gmab_heston_results}}
\\begin{{tabular}}{{lrr}}
\\toprule
Metric & Unhedged & Hedged \\\\
\\midrule
Mean P\\&L & {results['Unhedged_Mean_PnL']:.2f} & {results['Hedged_Mean_PnL']:.2f} \\\\
Std Dev P\\&L & {results['Unhedged_Std_PnL']:.2f} & {results['Hedged_Std_PnL']:.2f} \\\\
99.5\\% VaR & {results['Unhedged_VaR_99_5']:.2f} & {results['Hedged_VaR_99_5']:.2f} \\\\
99.5\\% CTE & {results['Unhedged_CTE_99_5']:.2f} & {results['Hedged_CTE_99_5']:.2f} \\\\
Min P\\&L & {results['Unhedged_Min_PnL']:.2f} & {results['Hedged_Min_PnL']:.2f} \\\\
Max P\\&L & {results['Unhedged_Max_PnL']:.2f} & {results['Hedged_Max_PnL']:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(table_heston_file, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {table_heston_file}")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # P&L distributions
    axes[0, 0].hist(unhedged_pnl, bins=50, alpha=0.7, label='Unhedged', density=True)
    axes[0, 0].hist(hedge_pnl,    bins=50, alpha=0.7, label='Hedged', density=True)
    axes[0, 0].set_xlabel('P&L')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('GMAB P&L Distributions (Heston)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Sample stock price paths
    sample_paths = min(10, S.shape[1])
    for i in range(sample_paths):
        axes[0, 1].plot(S[:, i], alpha=0.7, linewidth=1)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Stock Price')
    axes[0, 1].set_title(f'Sample Stock Price Paths (Heston, {sample_paths} of {n_paths:,})')
    axes[0, 1].grid(True, alpha=0.3)

    # Sample account value paths (DOLLARS)
    for i in range(sample_paths):
        axes[1, 0].plot(A[:, i], alpha=0.7, linewidth=1)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Account Value ($)')
    axes[1, 0].set_title(f'Sample Account Value Paths (GMAB, {sample_paths} of {n_paths:,})')
    axes[1, 0].grid(True, alpha=0.3)

    # Risk metrics comparison
    metrics = ['Mean P&L', 'Std Dev', '99.5% VaR', '99.5% CTE']
    unhedged_values = [results['Unhedged_Mean_PnL'], results['Unhedged_Std_PnL'],
                       results['Unhedged_VaR_99_5'], results['Unhedged_CTE_99_5']]
    hedged_values   = [results['Hedged_Mean_PnL'], results['Hedged_Std_PnL'],
                       results['Hedged_VaR_99_5'], results['Hedged_CTE_99_5']]

    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 1].bar(x - width/2, unhedged_values, width, label='Unhedged', alpha=0.8)
    axes[1, 1].bar(x + width/2, hedged_values, width, label='Hedged', alpha=0.8)
    axes[1, 1].set_xlabel('Risk Metrics')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Risk Metrics Comparison (Heston)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    figure_file = f"{figures_dir}/gmab_analysis_heston.png"
    plt.savefig(figure_file, dpi=figure_dpi, format=figure_format, bbox_inches='tight')
    plt.show()
    print(f"Plots saved to: {figure_file}")

    # Console summary
    print("\n" + "=" * 60)
    print("GMAB HESTON SIMULATION RESULTS")
    print("=" * 60)
    print(f"Contract: T={T} years, Guarantee={g_annual:.1%} p.a.")
    print(f"Simulation: {n_paths:,} paths, Seed={seed}\n")
    print("Risk Metrics:")
    print("-" * 40)
    print(f"{'Metric':<20} {'Unhedged':<12} {'Hedged':<12}")
    print("-" * 40)
    print(f"{'Mean P&L':<20} {results['Unhedged_Mean_PnL']:<12.2f} {results['Hedged_Mean_PnL']:<12.2f}")
    print(f"{'Std Dev P&L':<20} {results['Unhedged_Std_PnL']:<12.2f} {results['Hedged_Std_PnL']:<12.2f}")
    print(f"{'99.5% VaR':<20} {results['Unhedged_VaR_99_5']:<12.2f} {results['Hedged_VaR_99_5']:<12.2f}")
    print(f"{'99.5% CTE':<20} {results['Unhedged_CTE_99_5']:<12.2f} {results['Hedged_CTE_99_5']:<12.2f}")
    print("-" * 40)
    he = results['Hedge_Effectiveness']
    he_str = "n/a" if not np.isfinite(he) else f"{he*100:.2f}%"
    print(f"Hedge Effectiveness: {he_str}")
    print(f"Avg Transaction Cost: {results['Avg_Transaction_Cost']:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    results, unhedged_pnl, hedge_pnl, S, A = run_gmab_simulation_heston()
    save_results_and_plots_heston(results, unhedged_pnl, hedge_pnl, S, A)
    print("\nGMAB Heston analysis completed successfully!")
