"""
GMAB Variable Annuity Analysis under Rough Volatility Model

Author: Abdurakhmonbek Fayzullaev
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from datetime import datetime

# Add paths for imports
sys.path.append('.')
sys.path.append('Code')

# Import configuration and modules
from Step00_Configuration import *
from Step33_GMAB_Product_Module import (
    GMABParams, evolve_account_from_prices, gmab_value_and_delta
)
from Step15_Rough_Volatility_Simulation import simulate_rough_vol

# Output dirs
os.makedirs("Output/results", exist_ok=True)
os.makedirs("Output/tables",  exist_ok=True)
os.makedirs("Output/figures", exist_ok=True)
results_dir = "Output/results"
tables_dir  = "Output/tables"
figures_dir = "Output/figures"
results_file = f"{results_dir}/results_roughvol_gmab.csv"
table_file   = f"{tables_dir}/table_roughvol_gmab.tex"
figure_dpi = 300
figure_format = "png"

# ---------- helpers ----------
def format_risk_metrics(pnl_array: np.ndarray):
    if pnl_array.size == 0:
        return {k: 0.0 for k in ['mean_pnl','std_pnl','var_99_5','cte_99_5','min_pnl','max_pnl']}
    mean_pnl = float(np.mean(pnl_array))
    std_pnl  = float(np.std(pnl_array))
    var_99_5 = float(np.percentile(pnl_array, 0.5))
    cte_99_5 = float(np.mean(pnl_array[pnl_array <= var_99_5])) if np.any(pnl_array <= var_99_5) else var_99_5
    return {
        'mean_pnl': mean_pnl, 'std_pnl': std_pnl, 'var_99_5': var_99_5, 'cte_99_5': cte_99_5,
        'min_pnl': float(np.min(pnl_array)), 'max_pnl': float(np.max(pnl_array))
    }

def load_market_data():
    """Load r, q with safe fallbacks."""
    try:
        rf = pd.read_csv(riskfree_file)
        rf['maturity_days'] = rf['maturity_days'].astype(int)
        div = pd.read_csv(dividend_file)

        target_days = int(T * 365.25)
        idx = (rf['maturity_days'] - target_days).abs().idxmin()
        r = float(rf.loc[idx, 'rate']) / 100.0
        q = float(div['dividend_yield'].mean()) / 100.0
        return r, q
    except Exception as e:
        print(f"Warning: Could not load market data: {e}")
        print(f"Using default rates: r={r_default:.3%}, q={q_default:.3%}")
        return r_default, q_default

def get_rebalance_times(N, rebalance_freq):
    times = list(range(0, N, rebalance_freq))
    if (N - 1) not in times:
        times.append(N - 1)
    return sorted(times)

def rv_get(p: dict, key: str, default=None, aliases: tuple = ()):
    if p is None:
        return default
    if key in p:
        return p[key]
    for a in aliases:
        if a in p:
            return p[a]
    return default

# --- BS proxy for GMAB on the account (per-unit) ---
def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def gmab_value_delta_bs_proxy(
    S: float,
    A_ratio: float,
    T_remain: float,
    r: float, q: float,
    g_annual: float,
    fee_annual: float,
    sigma_proxy: float = 0.20
):
    """
    Per-unit proxy price and delta wrt S for GMAB shortfall using BS on U=A_ratio.
    """
    if T_remain <= 0.0:
        payoff_per_unit = max((1.0 + g_annual)**0.0 - A_ratio, 0.0)  # ~0 at maturity
        return payoff_per_unit, 0.0

    fee_factor = math.exp(-fee_annual * T_remain)
    U = max(A_ratio, 1e-12)
    # Effective strike so that fee-dragged account is compared to guarantee at T
    K_eff = (1.0 + g_annual) ** T / fee_factor

    vol = max(sigma_proxy, 1e-6)
    sqrtT = math.sqrt(T_remain)
    d1 = (math.log(U / K_eff) + (r - q + 0.5 * vol * vol) * T_remain) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT

    # BS put on U with yield q
    put_U = K_eff * math.exp(-r * T_remain) * _phi(-d2) - U * math.exp(-q * T_remain) * _phi(-d1)
    value_per_unit = fee_factor * put_U

    dV_dU = -math.exp(-q * T_remain) * _phi(-d1)      # BS put delta wrt U
    dU_dS = U / max(S, 1e-12)                          # chain rule (U≈proportional to S)
    delta_per_unit = fee_factor * dV_dU * dU_dS        # wrt S
    return value_per_unit, delta_per_unit

# ---------- main ----------
def run_gmab_simulation_roughvol():
    print("=" * 70)
    print("GMAB Analysis under Rough Volatility Model")
    print("=" * 70)
    print(f"Contract: T={T}y, Guarantee={g_annual:.1%} p.a., Fee={fee_annual:.1%}")
    print(f"Simulation: {n_paths:,} paths, {int(N):,} steps, rebalance={rebalance_freq}")
    print()

    # Seed & market
    if use_common_random_numbers:
        np.random.seed(seed)
        print(f"Using seed: {seed}")
    r, q = load_market_data()
    print(f"Market: r={r:.3%}, q={q:.3%}")

    # GMAB params + notional
    gmab_params = GMABParams(T_years=T, g_annual=g_annual, fee_annual=fee_annual,
                             trans_cost_bps=trans_cost_bps, rebalance_freq=rebalance_freq)
    initial_premium = float(S0)
    guarantee_amount = initial_premium * (1.0 + g_annual) ** T
    print(f"Initial premium: ${initial_premium:,.2f} | Maturity guarantee: ${guarantee_amount:,.2f}\n")

    # Simulate rough vol
    print("Simulating Rough Vol paths...")
    mu = r - q
    S, V = simulate_rough_vol(S0, mu, T, int(N), n_paths, roughvol_params, seed=seed)
    assert S.shape == (int(N) + 1, n_paths)
    print(f"Initial S: ${S[0,0]:.2f} | Avg final S: ${np.mean(S[-1,:]):.2f}\n")

    # Account evolution in DOLLARS
    print("Evolving account values with fees...")
    A = evolve_account_from_prices(S, gmab_params.fee_annual, dt_years)
    A_ratio = A / initial_premium  # per-unit for pricing & Greeks

    # Rebalance times
    rebal_idx = get_rebalance_times(int(N), gmab_params.rebalance_freq)
    print(f"Rebalance indices (first 10): {rebal_idx[:10]} ... total={len(rebal_idx)}")

    # Arrays
    hedge_pnl = np.zeros(n_paths)
    unhedged_pnl = np.zeros(n_paths)
    transaction_costs = np.zeros(n_paths)

    # sigma proxy from roughvol params if available
    rv_xi = rv_get(roughvol_params, 'xi', 0.04, aliases=('xi0','v0'))
    sigma_proxy_global = float(np.sqrt(max(rv_xi, 1e-8)))

    # Paths
    for p in range(n_paths):
        if p % 2000 == 0:
            print(f"  path {p:,}/{n_paths:,}")

        path_hedge_pnl = 0.0
        path_trans_cost = 0.0
        prev_shares = 0.0
        prev_S = S[rebal_idx[0], p]

        # Rebalance up to maturity
        for i, t_idx in enumerate(rebal_idx[:-1]):
            t = t_idx * dt_years
            T_remain = max(T - t, 0.0)
            current_S = S[t_idx, p]
            current_A_ratio = A_ratio[t_idx, p]

            # try module pricer first (per-unit)
            val_u, dlt_u = gmab_value_and_delta(
                S=current_S, A=current_A_ratio, T=T_remain, r=r, q=q,
                gmab_params=gmab_params, model='bs'  # proxy under rough vol
            )

            # fallback if delta tiny
            if abs(dlt_u) < 1e-10 and T_remain > 0.0:
                val_u, dlt_u = gmab_value_delta_bs_proxy(
                    S=current_S, A_ratio=current_A_ratio, T_remain=T_remain,
                    r=r, q=q, g_annual=gmab_params.g_annual, fee_annual=gmab_params.fee_annual,
                    sigma_proxy=sigma_proxy_global
                )

            # convert to SHARES
            shares_needed = dlt_u * initial_premium
            shares_change = shares_needed - prev_shares

            # cost on traded notional
            if abs(shares_change) > 1e-14:
                trade_notional = abs(shares_change) * current_S
                path_trans_cost += trade_notional * (trans_cost_bps / 10000.0)

            # realize P&L since last rebalance
            if i > 0:
                path_hedge_pnl += prev_shares * (current_S - prev_S)

            prev_shares = shares_needed
            prev_S = current_S

        # final step P&L on last shares
        final_S = S[-1, p]
        if len(rebal_idx) > 1:
            last_idx = rebal_idx[-2]
            path_hedge_pnl += prev_shares * (final_S - S[last_idx, p])

        # payoff in DOLLARS (robust)
        final_A_dollars = A[-1, p]
        maturity_payoff  = max(guarantee_amount - final_A_dollars, 0.0)

        # store insurer P&L
        unhedged_pnl[p] = -maturity_payoff
        hedge_pnl[p]    = -maturity_payoff + path_hedge_pnl - path_trans_cost
        transaction_costs[p] = path_trans_cost

    # Metrics
    print("\nSimulation complete. Computing risk metrics...")
    unhedged_metrics = format_risk_metrics(unhedged_pnl)
    hedged_metrics   = format_risk_metrics(hedge_pnl)
    hedge_eff = (np.nan if abs(unhedged_metrics['std_pnl']) < 1e-12
                 else 1 - hedged_metrics['std_pnl'] / unhedged_metrics['std_pnl'])

    payoffs = -unhedged_pnl  # positive
    itm_fraction = float(np.mean(payoffs > 0))
    avg_tc = float(np.mean(transaction_costs))

    # Results
    results = {
        'Model': 'RoughVol', 'Contract': 'GMAB',
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Paths': n_paths, 'T_years': T,
        'Guarantee_Rate': g_annual, 'Management_Fee': fee_annual,
        'Transaction_Cost_bps': trans_cost_bps, 'Rebalance_Freq_days': rebalance_freq,
        'Risk_Free_Rate': r, 'Dividend_Yield': q,
        'Initial_Premium': initial_premium, 'Guarantee_Amount': guarantee_amount,
        'RoughVol_xi': rv_xi, 'RoughVol_nu': rv_get(roughvol_params,'nu',None,('eta',)),
        'RoughVol_H': rv_get(roughvol_params,'H',None,('hurst','Hurst')),
        'ITM_Fraction': itm_fraction, 'Avg_Transaction_Cost': avg_tc,
        # Unhedged
        'Unhedged_Mean_PnL': unhedged_metrics['mean_pnl'],
        'Unhedged_Std_PnL':  unhedged_metrics['std_pnl'],
        'Unhedged_VaR_99_5': unhedged_metrics['var_99_5'],
        'Unhedged_CTE_99_5': unhedged_metrics['cte_99_5'],
        'Unhedged_Min_PnL':  unhedged_metrics['min_pnl'],
        'Unhedged_Max_PnL':  unhedged_metrics['max_pnl'] if 'max_pnl' in unhedged_metrics else float(np.max(unhedged_pnl)),
        # Hedged
        'Hedged_Mean_PnL':   hedged_metrics['mean_pnl'],
        'Hedged_Std_PnL':    hedged_metrics['std_pnl'],
        'Hedged_VaR_99_5':   hedged_metrics['var_99_5'],
        'Hedged_CTE_99_5':   hedged_metrics['cte_99_5'],
        'Hedged_Min_PnL':    hedged_metrics['min_pnl'],
        'Hedged_Max_PnL':    hedged_metrics['max_pnl'] if 'max_pnl' in hedged_metrics else float(np.max(hedge_pnl)),
        'Hedge_Effectiveness': hedge_eff,
        'Seed': seed
    }

    print(f"DEBUG: ITM {itm_fraction:.2%} | Unhedged std {unhedged_metrics['std_pnl']:.2f} | Hedged std {hedged_metrics['std_pnl']:.2f} | Avg TC ${avg_tc:.2f}")
    return results, unhedged_pnl, hedge_pnl, S, A, payoffs

def save_results_and_plots(results, unhedged_pnl, hedge_pnl, S, A, payoffs):
    pd.DataFrame([results]).to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")

    latex_table = f"""
\\begin{{table}}[H]
\\centering
\\caption{{GMAB Risk Metrics under Rough Volatility Model}}
\\label{{tab:gmab_roughvol_results}}
\\begin{{tabular}}{{lrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Unhedged}} & \\textbf{{Hedged}} \\\\
\\midrule
Mean P\\&L (\\$) & {results['Unhedged_Mean_PnL']:,.0f} & {results['Hedged_Mean_PnL']:,.0f} \\\\
Std Dev P\\&L (\\$) & {results['Unhedged_Std_PnL']:,.0f} & {results['Hedged_Std_PnL']:,.0f} \\\\
99.5\\% VaR (\\$) & {results['Unhedged_VaR_99_5']:,.0f} & {results['Hedged_VaR_99_5']:,.0f} \\\\
99.5\\% CTE (\\$) & {results['Unhedged_CTE_99_5']:,.0f} & {results['Hedged_CTE_99_5']:,.0f} \\\\
\\midrule
Hedge Effectiveness & \\multicolumn{{2}}{{c}}{{{he_str}}} \\\\
Avg Transaction Cost & \\multicolumn{{2}}{{c}}{{\\${results['Avg_Transaction_Cost']:,.0f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    with open(table_file, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {table_file}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GMAB Analysis under Rough Volatility Model', fontsize=16, fontweight='bold')

    axes[0,0].hist(unhedged_pnl, bins=50, alpha=0.7, density=True, label='Unhedged')
    axes[0,0].hist(hedge_pnl,    bins=50, alpha=0.7, density=True, label='Hedged')
    axes[0,0].set_title('P&L Distributions'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    sample = min(20, S.shape[1])
    for i in range(sample): axes[0,1].plot(S[:, i], alpha=0.5, lw=0.8)
    axes[0,1].set_title(f'Stock Price Paths (sample {sample})'); axes[0,1].grid(True, alpha=0.3)

    for i in range(sample): axes[0,2].plot(A[:, i], alpha=0.5, lw=0.8)
    axes[0,2].axhline(results['Guarantee_Amount'], color='r', ls='--', label='Guarantee')
    axes[0,2].set_title('Account Values'); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

    metrics = ['Mean P&L','Std Dev','99.5% VaR','99.5% CTE']
    unhedged_vals = [results['Unhedged_Mean_PnL'], results['Unhedged_Std_PnL'],
                     results['Unhedged_VaR_99_5'], results['Unhedged_CTE_99_5']]
    hedged_vals   = [results['Hedged_Mean_PnL'],   results['Hedged_Std_PnL'],
                     results['Hedged_VaR_99_5'],   results['Hedged_CTE_99_5']]
    x = np.arange(len(metrics)); w = 0.35
    axes[1,0].bar(x-w/2, unhedged_vals, w, label='Unhedged'); axes[1,0].bar(x+w/2, hedged_vals, w, label='Hedged')
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(metrics, rotation=45); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].hist(payoffs, bins=50, alpha=0.7, density=True, color='green'); axes[1,1].axvline(0, color='r', ls='--')
    axes[1,1].set_title(f'GMAB Payoffs (ITM {results["ITM_Fraction"]:.1%})'); axes[1,1].grid(True, alpha=0.3)

    sc = axes[1,2].scatter(S[-1,:], A[-1,:], s=2, alpha=0.5)
    axes[1,2].axhline(results['Guarantee_Amount'], color='r', ls='--')
    axes[1,2].set_title('Final Stock vs Account'); axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = f"{figures_dir}/gmab_analysis_roughvol.png"
    plt.savefig(fig_path, dpi=figure_dpi, format=figure_format, bbox_inches='tight'); plt.show()
    print(f"Plots saved to: {fig_path}")

    print("\n" + "="*70)
    print("GMAB ROUGH VOLATILITY SIMULATION RESULTS")
    print("="*70)
    print(f"T={results['T_years']}y, Guarantee={results['Guarantee_Rate']:.1%}, r={results['Risk_Free_Rate']:.1%}, q={results['Dividend_Yield']:.1%}")
    print(f"Paths={results['Paths']:,}, Seed={results['Seed']}, ITM={results['ITM_Fraction']:.1%}")
    he = results['Hedge_Effectiveness']; he_str = "n/a" if not np.isfinite(he) else f"{he*100:.2f}%"
    print(f"Hedge Effectiveness: {he_str} | Avg TC ${results['Avg_Transaction_Cost']:.2f}")
    print("="*70)

# ---------- run ----------
if __name__ == "__main__":
    try:
        print(f"Starting GMAB Rough Volatility analysis at {datetime.now()}")
        results, unhedged_pnl, hedge_pnl, S, A, payoffs = run_gmab_simulation_roughvol()
        save_results_and_plots(results, unhedged_pnl, hedge_pnl, S, A, payoffs)
        print(f"\nGMAB Rough Volatility analysis completed successfully at {datetime.now()}!")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
