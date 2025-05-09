from typing import Literal
import numpy as np
import numba as nb
import pandas as pd
import statsmodels.api as sm
from tabulate import tabulate
from matplotlib import cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100 # 设置分辨率

class FactorBacktester():
    def __init__(
        self,
        factor:pd.DataFrame| pd.Series, 
        bt_data:pd.DataFrame,
        time_point:str = '1100',
        period:int = 1,
        stk_pool:str = None,
        del_st:bool = True, # 剔除ST股票
        del_lowPri:bool = True, # 剔除低价股
        del_newList:bool = True, # 剔除次新股
        annualization = 252 
    ):
        # Filter data according to required conditions
        conditions = []
        if del_st:
            conditions.append(~bt_data['is_st'].values)
        if del_lowPri:
            conditions.append(bt_data['pre_vwap'].values > 3)
        if del_newList:
            conditions.append(bt_data['listed_days'].values > 30)
        if stk_pool:
            if stk_pool in bt_data.columns:
                conditions.append(bt_data[stk_pool].fillna(0).values.astype(bool))
            else:
                raise ValueError(f"Index {stk_pool} not Found in bt_data.")
        if conditions:
            mask = np.logical_and.reduce(conditions)
            bt_data = bt_data.loc[mask]
                
        ret_col = f'fwd_ret{period}d_shift0_{time_point}'
        stack_df = factor.stack().to_frame('factor_data').reset_index(names = ['date','symbol'])
        bt_data_merge = bt_data[['date','symbol',ret_col]].merge(stack_df,on = ['date','symbol'], how = 'inner')

        factor = bt_data_merge.pivot_table(index = 'date', columns = 'symbol', values = 'factor_data')
        returns = bt_data_merge.pivot_table(index = 'date', columns = 'symbol', values = ret_col)

        self._factor = factor
        self._f_mean = factor.mean(axis=1)
        self._f_std = factor.std(axis=1)
        self._factor_normalized = self._factor.sub(self._f_mean,axis=0).div(self._f_std,axis=0)
        
        self._returns = returns
        self._annualization = annualization

        self._method = None
        self._weights = None
        self._profit = None
        self._ic = None
        self._quantile_returns = None

    def _performance_summary(self, profit):
        if isinstance(profit, pd.DataFrame):
            return_data = profit.copy()
        elif isinstance(profit, (pd.Series, np.ndarray)):
            return_data = pd.DataFrame(profit, columns=['factor_return'])
        else:
            raise TypeError("profit must be of type pd.Series, pd.DataFrame, or np.ndarray")

        summary_stats = return_data.mean().to_frame('Mean').apply(lambda x: x* self._annualization)
        summary_stats['Volatility'] = return_data.std().apply(lambda x: x*np.sqrt(self._annualization))
        summary_stats['Sharpe Ratio'] = summary_stats['Mean']/summary_stats['Volatility']
        
        summary_stats['Skewness'] = return_data.skew()
        summary_stats['Excess Kurtosis'] = return_data.kurtosis()
        summary_stats['VaR (0.05)'] = return_data.quantile(.05, axis = 0)
        summary_stats['CVaR (0.05)'] = return_data[return_data <= return_data.quantile(.05, axis = 0)].mean()
        summary_stats['Min'] = return_data.min()
        summary_stats['Max'] = return_data.max()
        
        wealth_index = 1000*(1+return_data).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks

        summary_stats['Max Drawdown'] = drawdowns.min()
        summary_stats['Peak'] = [previous_peaks[col][:drawdowns[col].idxmin()].idxmax() for col in previous_peaks.columns]
        summary_stats['Bottom'] = drawdowns.idxmin()
        
        recovery_date = []
        for col in wealth_index.columns:
            prev_max = previous_peaks[col][:drawdowns[col].idxmin()].max()
            recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin():]]).T
            recovery_date.append(recovery_wealth[recovery_wealth[col] >= prev_max].index.min())
        summary_stats['Recovery'] = recovery_date
        
        # Format the summary statistics
        summary_stats = summary_stats.applymap(lambda x: f"{x:.3f}" if isinstance(x, float) else (f"{x.strftime('%Y%m%d')}" if isinstance(x,pd.Timestamp) else x))
        return summary_stats
    
    def _calculate_factor_weights(self, method='long_short'):
        def _get_factor_signs(factors):
            return np.sign(factors)

        def _neutralize_weights(weights,ignore = 0):
            weights = weights.copy()
            if ignore == 0:
                pass
            elif ignore == -1:
                weights = weights.mask(weights>0,other=0)
            elif ignore == 1:
                weights = weights.mask(weights<0,other=0)      
            return weights.div(weights.abs().sum(axis = 1), axis='index')
    
        if method == 'long_short_equal_weights':
            self._weights = _neutralize_weights(_get_factor_signs(self._factor_normalized))

        elif method == 'short_equal_weights':
            self._weights = _neutralize_weights(_get_factor_signs(self._factor_normalized),ignore=-1)
            
        elif method == 'long_equal_weights':
            self._weights = _neutralize_weights(_get_factor_signs(self._factor_normalized),ignore=1)
            
        elif method == 'long_short':
            self._weights = _neutralize_weights(self._factor_normalized)
            
        elif method == 'short':
            self._weights = _neutralize_weights(self._factor_normalized,ignore=-1)
            
        elif method == 'long':
            self._weights = _neutralize_weights(self._factor_normalized,ignore=1)
        
        self._method = method

    def _calculate_factor_returns(self):
        self._profit = self._weights.mul(self._returns).sum(axis = 1)

    def _calculate_ic(self):
        self._ic = self._factor.corrwith(self._returns,axis = 1, method = 'spearman') # Using spearman correlation

    def _calculate_quantile_returns(self, quantiles = 10):
        @nb.njit
        def rank_and_qcut_with_nan_numba(ranked_f, returns, quantiles):
            n_rows, n_cols = ranked_f.shape
            quantile_returns = np.zeros((n_rows, quantiles), dtype=np.float64)
            counts = np.zeros((n_rows, quantiles), dtype=np.int64) 

            for i in range(n_rows):
                row = ranked_f[i, :]
                ret_row = returns[i, :]
                valid_mask = ~np.isnan(row)
                valid_ranks = row[valid_mask]
                valid_returns = ret_row[valid_mask]

                if valid_ranks.size > 0:
                    cuts = np.percentile(valid_ranks, np.linspace(0, 100, quantiles + 1)[1:-1])
                    quantile_indices = np.searchsorted(cuts, valid_ranks)

                    for j in range(valid_ranks.size):
                        q = quantile_indices[j]
                        quantile_returns[i, q] += valid_returns[j]
                        counts[i, q] += 1

                    for q in range(quantiles):
                        if counts[i, q] > 0:
                            quantile_returns[i, q] /= counts[i, q]
            return quantile_returns
        
        quantile_returns = rank_and_qcut_with_nan_numba(self._factor.rank(axis = 1, method = 'first').values, self._returns.fillna(0).values, quantiles) # factor_rank(method=first) return_fillna0
        self._quantile_returns =  pd.DataFrame(quantile_returns, index = self._factor.index, columns=[f'{i+1}' for i in range(quantiles)])

    def _plot_factor_rets(self):  
        pnl = self._profit.copy().to_frame(self._method)

        fig, ax1 = plt.subplots(figsize=(15,5))
        ax1.plot(pnl.loc[:, self._method].cumsum(), label='Cum_Ret', c='blue')
        ax1.set_title(f"Factor Returns: {self._method}")
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Ret')
        ax1.legend()
        plt.show()

    def _plot_ic(self):
        freq = 'W'
        rolling_win = 4
    
        ic = self._ic.copy()

        s_resample_mean = ic.resample(freq).mean()
        s_resample_mean_swm = s_resample_mean.rolling(rolling_win).mean()

        fig = plt.figure(figsize=(15,5))
        colors = np.where(abs(s_resample_mean.values) > 0.2, 'darkred', 
            np.where(abs(s_resample_mean.values) > 0.1, 'red', 
            np.where(abs(s_resample_mean.values) > 0.03, 'blue', 'gray')))
        plt.scatter(s_resample_mean.index, s_resample_mean.values, s=50, c=colors)
        plt.plot(s_resample_mean_swm.index, s_resample_mean_swm.values, linewidth=3, linestyle='-', color='gray', alpha=0.8)
        plt.axhline(0, linestyle='--', color='gray', alpha=0.5)
        plt.title(freq + ' Frequent Mean of ICs')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.xlim(s_resample_mean.index.min(), s_resample_mean.index.max())

        fig, axs = plt.subplots(ncols=3, figsize=(15,4))
        axs[0].hist(ic.values, color='cornflowerblue', alpha=0.5)
        axs[0].set_title('Distribution of IC')
        axs[0].set_xlabel('Values')
        axs[0].set_ylabel('Frequency')

        sm.qqplot(ic.values, line='s', ax=axs[1])
        axs[1].set_title('QQ Plot of IC')
        axs[1].set_xlabel('Theoretical Quantiles')
        axs[1].set_ylabel('Sample Quantiles')

        axs[2].plot(ic.cumsum())
        axs[2].set_title('Cumulative ICs')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Values')
        
        axs[2].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    def _plot_quantile_returns(self):
        pnl = self._quantile_returns.copy()

        quantiles = pnl.shape[1]
        my_colors = cm.RdYlBu(np.arange(quantiles) / quantiles)

        fig, axs = plt.subplots(ncols=2, figsize=(15,4))
        # Plot annualized returns
        pnl.mean().to_frame('Mean').apply(lambda x: x * self._annualization).plot.bar(ax=axs[0], color='blue')
        axs[0].set_title('Quantile Returns of IC')
        axs[0].set_xlabel('Quantiles')
        axs[0].set_ylabel('Annualized Returns')

        # Plot cumulative returns
        lines = []
        for i in range(quantiles):
            line, = axs[1].plot(pnl.cumsum().iloc[:, i], color=my_colors[i], alpha=0.8)
            if i == 0 or i == quantiles - 1:
                line.set_linewidth(2)  # Make the first and last lines thicker
                line.set_alpha(1)  # Make the first and last lines more opaque
            lines.append(line)
        axs[1].set_title('Quantile Returns of IC')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Cumulative Returns')
        
        legend_labels = [f'{i+1}' for i in range(quantiles)]
        legend = axs[1].legend(lines, legend_labels, bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0., frameon=False)
    
    def get_factor_rets(
        self,
        method:Literal['long_short_equal_weights','short_equal_weights','long_equal_weights','long_short','long','short'] = "long_short",  
        plot:bool = False,
        evaluation:bool = False
    ):
        if method != self._method:
            self._calculate_factor_weights(method)
            self._calculate_factor_returns()

        if plot:
            self._plot_factor_rets()
        
        if evaluation:
            perforamnce_table = tabulate(self._performance_summary(self._profit), headers='keys', tablefmt='plain', floatfmt=".3f")
            print(perforamnce_table)

        return self._profit

    def get_ic(
        self,
        plot:bool = False
    ):
        if self._ic is None:
            self._calculate_ic()

        if plot:
            self._plot_ic()

        return self._ic
        
    def get_quantile_rets(
        self,
        quantiles:int = 10,
        plot:bool = False,
        evaluation:bool = False
    ):
        if self._quantile_returns is None or self._quantile_returns.shape[1] != quantiles:
            self._calculate_quantile_returns(quantiles)

        if plot:
            self._plot_quantile_returns()

        if evaluation:
            perforamnce_table = tabulate(self._performance_summary(self._quantile_returns), headers='keys', tablefmt='plain', floatfmt=".3f")
            print(perforamnce_table)

        return self._quantile_returns
    
