import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import plotly.graph_objects as go
import networkx as nx
from IPython.display import display, HTML
import statsmodels.api as sm
from scipy.stats import gaussian_kde
import os

from .FactorBacktester import FactorBacktester
from .FactorRiskController import FactorRiskController
from .const_variables import bt_required_cols
import fds

class FactorFactory:
    # def __init__(self, date_range = ['2020-01-01','2023-12-31']): # 后续数据完全存在fds 便于及时更新
    #     self.bt_data = fds.mid('BT_DATA',date_range = date_range, cols = bt_required_cols) 
    
    def __init__(self, bt_data_path = "/root/data/base_data/bt_data.parq"):
        self.bt_data = pd.read_parquet(bt_data_path,columns = bt_required_cols)

    def reload_bt_data(self, date_range): # 后续数据完全存在fds 便于及时更新
        self.bt_data = fds.mid('BT_DATA',date_range = date_range, cols = bt_required_cols) 
    
    def evaluate_factor(self, factor, num_quantiles=20, factor_ret_method='long_short', time_point = '1100', period = 1, stk_pool = None):
        """Evaluates a factor's comprehensive performance."""
        null_percentage = self.calculate_null_percentage(factor)
        factor.dropna(axis=0, thresh=1, inplace=True) # drop rows with all NaNs
        
        bt = FactorBacktester(factor = factor, bt_data = self.bt_data, time_point = time_point, period = period, stk_pool = stk_pool)
        ic = bt.get_ic(plot=True)
        mean_ic = ic.mean()
        std_ic = ic.std()
        icir = mean_ic / std_ic
        print(f'rank_ic: {mean_ic:.4f}, rank_ic_std: {std_ic:.4f}, rank_ic_ir: {icir:.4f}, null_percentage: {null_percentage}')

        quantile_rets = bt.get_quantile_rets(quantiles=num_quantiles, plot=True, evaluation=True)
        factor_ret = bt.get_factor_rets(method=factor_ret_method, plot=True, evaluation=True)

    def evaluate_factor_icir(self, factor, icir_thresh=0.3, ic_thresh=0.025, return_res = False, show_res = True,  time_point = '1100', period = 1, stk_pool = None):
        """Evaluates a factor based on IC and ICIR thresholds with formatting."""
        if not return_res and not show_res:
            return 
            
        null_percentage = self.calculate_null_percentage(factor)
        factor.dropna(axis=0, thresh=1, inplace=True) # drop rows with all NaNs

        bt = FactorBacktester(factor = factor, bt_data = self.bt_data, time_point = time_point, period = period, stk_pool = stk_pool)
        ic = bt.get_ic(plot=False)
        mean_ic = ic.mean()
        std_ic = ic.std()
        icir = mean_ic / std_ic

        if show_res:
            content = f'rank_ic: {mean_ic:.4f}, rank_ic_std: {std_ic:.4f}, rank_ic_ir: {icir:.4f}, null_percentage: {null_percentage}'
            if abs(icir) > icir_thresh and abs(mean_ic) > ic_thresh:
                display(HTML(f'<p style="color: darkred;">{content}</p>'))
            elif abs(icir) > icir_thresh or abs(mean_ic) > ic_thresh:
                display(HTML(f'<p style="color: salmon;">{content}</p>'))
            else:
                display(HTML(f'<p style="color: gray;">{content}</p>'))
    
        if return_res:
            return icir, mean_ic, std_ic, null_percentage

    def evaluate_factor_extension_stats(self, factor, window=20, min_periods=2, time_point='1100', period=1, stk_pool=None):
        """Quickly evaluates various extensions of a factor."""
        def evaluate_extension(name, extension):
            print(f"{window}{name}:")
            self.evaluate_factor_icir(extension, time_point=time_point, period=period, stk_pool=stk_pool)
        evaluate_extension("Mean", factor.rolling(window, min_periods=min_periods).mean())
        evaluate_extension("RankMean", factor.rank(1).rolling(window, min_periods=min_periods).mean())
        evaluate_extension("ExpMean", factor.rolling(window, min_periods=min_periods, win_type='exponential').mean(center=window, tau=-1/np.log(1-(2/(1+window))), sym=False))
        evaluate_extension("Std", factor.rolling(window, min_periods=min_periods).std())
        evaluate_extension("Skew", factor.rolling(window, min_periods=min_periods).skew())
        evaluate_extension("Kurt", factor.rolling(window, min_periods=min_periods).kurt())
        evaluate_extension("Median", factor.rolling(window, min_periods=min_periods).median())
        evaluate_extension("Min", factor.rolling(window, min_periods=min_periods).min())
        evaluate_extension("Max", factor.rolling(window, min_periods=min_periods).max())

    def evaluate_factor_risk_report(self, factor, time_point='1100', period=1, stk_pool=None, run_grouped_style_analysis=True, run_category_IC_analysis=False, run_top_bottom_analysis=False, run_turnover_decay_analysis=False, **kwargs):
        """
        Generate comprehensive analysis of factor risk.
    
        **kwargs: Additional keyword arguments.
            style_list_ts (list, optional): List of styles for time-series analysis. 
                Defaults to ['Size', 'Liquidity', 'BP', 'ResVol', 'pre_vwap', 'vwap', 'pre_amount', 'amount', 'pre_ret1d_shift0_1500', 'fwd_ret1d_shift0_1500', 'pre_ret1d_std5d', 'fwd_ret1d_std5d'].
            style_list_all (list, optional): List of all styles. 
                Defaults to ['Size', 'Liquidity', 'BP', 'ResVol', 'pre_vwap', 'pre_amount', 'pre_ret1d_shift0_1500', 'pre_ret1d_std5d'].
            cumIC_cat_list (list, optional): List of categories for cumulative IC analysis. 
                Defaults to ['exchange', 'citic_ind_1st', ('Size', 5), ('Liquidity', 5), ('pre_vwap', 5), ('pre_amount', 5), ('pre_ret1d_shift0_1500', 5), ('pre_ret1d_std5d', 5)].
            IC_cat_list (list, optional): List of categories for IC analysis. 
                Defaults to [('Size', 5), ('Liquidity', 5), ('pre_vwap', 5), ('pre_amount', 5), ('pre_ret1d_shift0_1500', 5), ('pre_ret1d_std5d', 5)].
            top_bottom_columns_list (list, optional): List of columns for top-bottom analysis. 
                Defaults to ['Size', 'Liquidity', 'pre_vwap', 'pre_ret1d_shift0_1500', 'pre_ret1d_std5d', 'pre_amount','citic_ind_1st'].
            n_groups_grouped_style_ts (int, optional): Number of groups for grouped style time-series analysis. 
                Defaults to 5.
            n_cols_grouped_style_ts (int, optional): Number of columns for grouped style time-series analysis. 
                Defaults to 4.
            n_groups_grouped_style_all (int, optional): Number of groups for grouped style all analysis. 
                Defaults to 10.
            n_top_bottom (int, optional): Number of top-bottom groups. 
                Defaults to 10.
            top_bottom_plot (bool, optional): Whether to plot top-bottom analysis. 
                Defaults to True.
            top_bottom_save_html (bool, optional): Whether to save top-bottom analysis as HTML. 
                Defaults to True.
            top_bottom_output_file (str, optional): File path to save top-bottom analysis HTML. 
                Defaults to './top_bottom_analysis.html'.
            n_groups_turnover (int, optional): Number of groups for turnover analysis. 
                Defaults to 10.
            signal_decay_k (float, optional): Signal decay parameter. 
                Defaults to 0.1.
        """
        factor.dropna(axis=0, thresh=1, inplace=True)
        
        frc = FactorRiskController(factor = factor, bt_data=self.bt_data, time_point=time_point, period=period, stk_pool=stk_pool)
        
        # 解析 kwargs 中的参数
        style_list_ts = kwargs.get('style_list_ts', ['Size', 'Liquidity', 'BP', 'ResVol', 'pre_vwap', 'vwap', 'pre_amount', 'amount', 'pre_ret1d_shift0_1500', 'fwd_ret1d_shift0_1500', 'pre_ret1d_std5d', 'fwd_ret1d_std5d'])
        style_list_all = kwargs.get('style_list_all', ['Size', 'Liquidity', 'BP', 'ResVol', 'pre_vwap', 'pre_amount', 'pre_ret1d_shift0_1500', 'pre_ret1d_std5d'])
        cumIC_cat_list = kwargs.get('cumIC_cat_list', ['exchange', 'citic_ind_1st', ('Size', 5), ('Liquidity', 5), ('pre_vwap', 5), ('pre_amount', 5), ('pre_ret1d_shift0_1500', 5), ('pre_ret1d_std5d', 5)])
        IC_cat_list = kwargs.get('IC_cat_list', [('Size', 5), ('Liquidity', 5), ('pre_vwap', 5), ('pre_amount', 5), ('pre_ret1d_shift0_1500', 5), ('pre_ret1d_std5d', 5)])
        top_bottom_columns_list = kwargs.get('top_bottom_columns_list', ['Size', 'Liquidity', 'pre_vwap', 'pre_ret1d_shift0_1500', 'pre_ret1d_std5d', 'pre_amount','citic_ind_1st',])
        
        n_groups_grouped_style_ts = kwargs.get('n_groups_grouped_style_ts', 5)
        n_cols_grouped_style_ts = kwargs.get('n_cols_grouped_style_ts', 4)
        n_groups_grouped_style_all = kwargs.get('n_groups_grouped_style_all', 10)
        n_top_bottom = kwargs.get('n_top_bottom', 10)
        top_bottom_plot = kwargs.get('top_bottom_plot', True)
        top_bottom_save_html = kwargs.get('top_bottom_save_html', True)
        top_bottom_output_file = kwargs.get('top_bottom_output_file', './top_bottom_analysis.html')
        n_groups_turnover = kwargs.get('n_groups_turnover', 10)
        signal_decay_k = kwargs.get('signal_decay_k', 0.1)
        
        # 分组属性分析
        if run_grouped_style_analysis:
            frc.plot_grouped_style_ts(style_list_ts, n_groups=n_groups_grouped_style_ts, n_cols = n_cols_grouped_style_ts)
            frc.plot_grouped_style(style_list_all, n_groups=n_groups_grouped_style_all, CRS=True)
    
        # 分类IC分析
        if run_category_IC_analysis:
            if len(cumIC_cat_list) > 0:
                for cat in cumIC_cat_list:
                    frc.plot_category_cumIC(cat)
        
            if len(IC_cat_list) > 0:
                for cat in IC_cat_list:
                    frc.plot_category_IC(cat)
        
        # 首尾分析
        if run_top_bottom_analysis:
            top_stats, bottom_stats = frc.top_bottom_analysis(n=n_top_bottom,columns_list=top_bottom_columns_list, plot=top_bottom_plot, save_html=top_bottom_save_html, output_file=top_bottom_output_file)
            display(f"TOP{n_top_bottom} Info:",top_stats.head(n_top_bottom))
            display(f"BOTTOM{n_top_bottom} Info:",bottom_stats.head(n_top_bottom))
    
        # 换手及信号衰减分析
        if run_turnover_decay_analysis:
            frc.plot_leadlag_rankicir()
            frc.plot_group_turnover(n_groups=n_groups_turnover)
            frc.plot_signal_decay(k=signal_decay_k)

    def save_factor(self, factor, factor_name="factor", time_point = "1100", root_path = "/root/backtest/factors"):
        """Saves the factor data to a specified directory."""
        factor_dir = os.path.join(root_path, factor_name,time_point)
        os.makedirs(factor_dir, exist_ok=True)
        factor_save_path = os.path.join(factor_dir, "data.parq")
        factor.index.name = "date"
        factor.columns.name = "symbol"
        factor_df = factor.stack().to_frame(factor_name).reset_index()  # Unstack and reset index
        factor_df.to_parquet(factor_save_path)

    def read_factor(self, factor_name="factor", time_point="1100", root_path="/root/backtest/factors"):
        """Reads the factor data from a specified directory and restores it to the original DataFrame format. The factor files should be stored in the asked format, seen in the save_factor function."""
        factor_save_path = os.path.join(root_path, factor_name, time_point, "data.parq")
        if not os.path.exists(factor_save_path):
            raise FileNotFoundError(f"File not found: {factor_save_path}")
        factor_df = pd.read_parquet(factor_save_path)
        required_columns = ['date', 'symbol', factor_name]
        if not all(column in factor_df.columns for column in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        factor_df = factor_df.pivot_table(index='date', columns='symbol', values=factor_name)
        factor_df.index = pd.to_datetime(factor_df.index)
        return factor_df

    def extract_style(self, style_name = 'Size'):
        """Get specified style data from bt_data and output in a wide dataframe form."""
        if not style_name in self.bt_data.columns:
            raise ValueError(f"{style_name} not in bt_data.")
        style_df = self.bt_data.pivot_table(index = 'date', columns ='symbol', values = style_name)
        return style_df
        
    def plot_heatmap(self, df, plot_type = 'heatmap' ,title='Heatmap', cmap='coolwarm', annot=True, fmt=".2f", figsize=(10, 8), vmin=-1, vmax=1):
        """Plots a heatmap for the given DataFrame."""
        if plot_type == 'heatmap':
            plt.figure(figsize=figsize)
            sns.heatmap(df, annot=annot, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.title(title)
        elif plot_type == 'clustermap':
            g = sns.clustermap(df, annot=annot, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax)
            g.figure.suptitle(title)
        plt.show()

    def find_correlated_groups(self, df, threshold=0.6):
        """Finds groups of correlated features in the given correlation table."""
        G = nx.Graph()
        G.add_nodes_from(df.columns)
        
        for i in range(len(df.columns)):
            for j in range(i + 1, len(df.columns)):
                if df.iloc[i, j] > threshold:
                    G.add_edge(df.columns[i], df.columns[j])
        
        subgraphs = []
        for component in nx.connected_components(G):
            if len(component) > 1:
                subgraphs.append(G.subgraph(component))
            else:
                subgraphs.append(component)
        
        groups = {}
        for i, subgraph in enumerate(subgraphs):
            if isinstance(subgraph, set):
                groups[f"Group {i+1}"] = list(subgraph)
            else:
                groups[f"Group {i+1}"] = list(subgraph.nodes())
        
        return groups

    def _reorganize_factors(self, factors, factor_names=None, sample_size = None):
        if factor_names is None:
            factor_names = [f"factor_{i}" for i in range(len(factors))]

        intersection_dates = None
        for f in factors:
            if intersection_dates:
                intersection_dates = intersection_dates.intersection(f.index)
            else:
                intersection_dates = set(f.index)
        intersection_dates = list(intersection_dates)
        
        if sample_size is not None:
            if sample_size > len(intersection_dates):
                raise ValueError("Sample size should be less than the number of observations.")
            sampled_groups = np.random.choice(intersection_dates, sample_size, replace=False)
        else:
            sampled_groups = intersection_dates
            
        factor_df = pd.concat([f.loc[sampled_groups].stack().rename(factor_names[i]) for i, f in enumerate(factors)], axis=1)
        factor_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return factor_df
        
    def factor_correlation_analysis(self, factors, factor_names=None, return_all_matrices=False, plot=True, corr_method='spearman', sample_size = None, plot_type='heatmap', annot = True, fmt=".2f", cmap='coolwarm', figsize=(10, 8)):
        """Analyzes correlation between multiple factors."""
        factor_df = self._reorganize_factors(factors, factor_names, sample_size)
        factor_names = factor_df.columns
        
        correlation_matrices = factor_df.groupby(level=0).corr(method=corr_method)
        mean_correlation = correlation_matrices.groupby(level=1).mean().loc[factor_names, factor_names]
        if plot:
            self.plot_heatmap(mean_correlation, plot_type= plot_type, title='Factor Correlation', cmap=cmap, annot=annot, fmt=fmt, figsize=figsize)
        return (correlation_matrices, mean_correlation) if return_all_matrices else mean_correlation

    def factor_overlap_analysis(self, factors, factor_names=None, return_all_matrices=False, plot=True, low_pct=0.8, high_pct=1, sample_size = None, plot_type='heatmap', annot = True, fmt=".2f", cmap='coolwarm', figsize=(10, 8)):
        """Analyzes overlap between multiple factors."""
        def calculate_column_overlap(df):
            data = df.to_numpy()
            
            low_quantiles = np.nanquantile(data, low_pct, axis=0)
            high_quantiles = np.nanquantile(data, high_pct, axis=0)
            
            masks = (data >= low_quantiles) & (data <= high_quantiles)
            masks = masks.astype(float)
            
            overlap_scores = np.dot(masks.T, masks)  # Pairwise intersection counts
            column_sums = masks.sum(axis=0)  # Sum of each column's mask
            overlap_scores = overlap_scores / column_sums  # Normalize by column sums
            
            overlap_df = pd.DataFrame(overlap_scores, index=df.columns, columns=df.columns)
            return overlap_df
        
        factor_df = self._reorganize_factors(factors, factor_names, sample_size)
        factor_names = factor_df.columns

        overlap_matrices = factor_df.groupby(level=0).apply(calculate_column_overlap)
        mean_overlap = overlap_matrices.groupby(level=1).mean().loc[factor_names, factor_names]
        if plot:
            self.plot_heatmap(mean_overlap, plot_type= plot_type, title='Factor Correlation', cmap=cmap, annot=annot, fmt=fmt, figsize=figsize)
        return (overlap_matrices, mean_overlap) if return_all_matrices else mean_overlap

    def plot_joint_pdf_with_outlier_removal(self, series1, series2, name1 = 'f1', name2 = 'f2', lower_bound = 0, upper_bound = 1, interactive = False):
        """Analyzes correlation between multiple factors by plotting joint pdf."""
        df = pd.concat([series1, series2], axis=1)
        df.columns = [name1, name2]
        lower_bound_x = df[name1].quantile(lower_bound)
        upper_bound_x = df[name1].quantile(upper_bound)
        lower_bound_y = df[name2].quantile(lower_bound)
        upper_bound_y = df[name2].quantile(upper_bound)

        filtered_df = df[(df[name1] >= lower_bound_x) & (df[name1] <= upper_bound_x) & (df[name2] >= lower_bound_y) & (df[name2] <= upper_bound_y)]

        x = filtered_df[name1].values
        y = filtered_df[name2].values

        kde = gaussian_kde(np.vstack([x, y]))

        x_fine = np.linspace(x.min(), x.max(), 100)
        y_fine = np.linspace(y.min(), y.max(), 100)
        xgrid_fine, ygrid_fine = np.meshgrid(x_fine, y_fine)

        z_values_fine = kde(np.vstack([xgrid_fine.ravel(), ygrid_fine.ravel()]))
        zgrid_fine = np.reshape(z_values_fine, xgrid_fine.shape)
        if interactive:
            fig = go.Figure(data=[go.Surface(z=zgrid_fine, x=xgrid_fine, y=ygrid_fine)])
            fig.update_layout(title='Joint Probability Density', autosize=True,
                            scene=dict(
                                xaxis_title=f'{name1}',
                                yaxis_title=f'{name2}',
                                zaxis_title='Probability Density'),
                            margin=dict(l=65, r=50, b=65, t=90))
            fig.show()
        else:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot_surface(xgrid_fine, ygrid_fine, zgrid_fine, cmap='viridis')

            ax.set_xlabel(f'{name1}')
            ax.set_ylabel(f'{name2}')
            ax.set_zlabel('Probability Density')
            plt.show()

    def plot_distribution(self, series, bins='auto', log_scale=False):
        """Plots the distribution of a series with optional log scaling."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(series, bins=bins, kde=False, log_scale=log_scale, ax=ax)
        if log_scale:
            ax.set_xscale('log')
        else: # Calculate statistics and plot is not log_scaled
            mean_val = series.mean()
            std_val = series.std()
            skew_val = series.skew()
            kurt_val = series.kurt()
            
            ax.axvline(mean_val, color='red', linestyle='--', label='Mean')
            ax.axvline(mean_val + 3 * std_val, color='red', linestyle=':', label='Mean + 3σ')
            ax.axvline(mean_val - 3 * std_val, color='red', linestyle=':', label='Mean - 3σ')
            
            stats_text = (
                f'Mean: {mean_val:.2f}\n'
                f'Std: {std_val:.2f}\n'
                f'Skew: {skew_val:.2f}\n'
                f'Kurt: {kurt_val:.2f}'
            )
            ax.text(0.85, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray',facecolor='white', alpha=0.8, lw=1))
        ax.grid(True)
        ax.set_title('Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()
 
    def plot_factor_quantiles(self, factor, use_plotly=False, log_scale=False, figsize=(12, 8)):
        """Plots the quantiles of a factor from a dynamic perspective."""
        quantiles = pd.concat([factor.quantile(i / 100, axis=1) for i in range(0, 105, 10)], axis=1)
        quantiles.columns = [f'{i}%' for i in range(0, 105, 10)]
        
        if use_plotly:
            fig = go.Figure()
            for col in quantiles.columns:
                fig.add_trace(go.Scatter(x=quantiles.index, y=quantiles[col], mode='lines', name=col))
            fig.update_layout(
                title="Quantiles of Test Factor",
                xaxis_title="Date",
                yaxis_title="Value",
                yaxis_type='log' if log_scale else 'linear',
                margin=dict(l=20, r=20, t=20, b=20)
            )
            fig.show()
        else:
            colors = cm.RdYlBu(np.linspace(0, 1, len(quantiles.columns)))
            fig, ax = plt.subplots(figsize=figsize)
            for col, color in zip(quantiles.columns, colors):
                ax.plot(quantiles.index, quantiles[col], label=col, color=color)
            ax.set_title("Quantiles of Test Factor")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            if log_scale:
                ax.set_yscale('log')
            csm = plt.cm.ScalarMappable(cmap=cm.RdYlBu, norm=plt.Normalize(vmin=0, vmax=100))
            csm.set_array([])
            cbar = fig.colorbar(csm, ax=ax, pad=0.01, ticks=np.linspace(0, 100, len(quantiles.columns)))
            cbar.set_label("Percentiles")
            plt.tight_layout()  # 为了更好地适应Legend和颜色条
            plt.show()

    def calculate_null_percentage(self, df):
        """Calculates the average percentage of NaNs in the DataFrame."""
        null_percentage = df.isnull().mean() * 100
        return null_percentage.mean()
    
    def standardize_factor(self, factor):
        """Standardizes the factor data by z-scoring across each day."""
        mean = factor.mean(axis=1)
        std = factor.std(axis=1)
        standardized_factor = factor.sub(mean, axis=0).div(std, axis=0)
        return standardized_factor
    
    def replace_outliers(self, df, num_std=3, winsorize = False):
        """Replaces outliers in DataFrame with NaN or n-sigma boundaries, based on a specified number of standard deviations."""
        data = df.astype(float).to_numpy()  # 将 DataFrame 转换为 NumPy 数组
        mean = np.nanmean(data, axis=1, keepdims=True)  # 计算每行的均值
        std = np.nanstd(data, axis=1, ddof=1, keepdims=True)  # 计算每行的标准差
    
        if winsorize:
            # 计算上下限
            upper_limits = mean + num_std * std
            lower_limits = mean - num_std * std
            # 使用 np.clip 进行 Winsorize 处理
            data_capped = np.clip(data, lower_limits, upper_limits)
            return pd.DataFrame(data_capped, columns=df.columns, index=df.index)
        else:
            # 标记异常值并替换为 NaN
            is_outlier = np.abs(data - mean) > num_std * std
            data[is_outlier] = np.nan
            return pd.DataFrame(data, columns=df.columns, index=df.index)

    def qcut_row(self, df, n_groups):
        """Quantile_cut df along horizontal axis."""
        def numpy_qcut(row, n_groups):
            valid_data = row[~np.isnan(row)]  # 去除 NaN 值
            quantiles = np.percentile(valid_data, np.linspace(0, 100, n_groups + 1))[0:-1]  # 计算分位数
            group = np.full(row.shape, np.nan)  # 初始化为 NaN 的浮点数组
            group[~np.isnan(row)] = np.searchsorted(quantiles, row[~np.isnan(row)], side="right") - 1 # 分配到各个分位
            return group
        col_groups = np.apply_along_axis(numpy_qcut, 1, df.values, n_groups) + 1 # 从 1 开始分组
        qcut_res = pd.DataFrame(col_groups, index=df.index, columns=df.columns) 
        return qcut_res
    
    def group_standardize_factor(self, y, x):
        """Calculates group-wise Z-scores for DataFrame y based on groupings in DataFrame x."""
        assert y.shape == x.shape, "DataFrames should have the same dimensions."
        y_stacked = y.stack().to_frame('y').reset_index(names = ['date','symbol'])
        x_stacked = x.stack().to_frame('x').reset_index(names = ['date','symbol'])
        merged = pd.merge(y_stacked, x_stacked, on=['date', 'symbol'])
        grouped = merged.groupby(['date', 'x'])
        mean = grouped['y'].transform('mean')  # 计算每组的均值
        std = grouped['y'].transform('std', ddof=0)  # 计算每组的标准差
        merged['zscore'] = (merged['y'] - mean) / std
        zscore_df = merged.pivot(index='date', columns='symbol', values='zscore')
        return zscore_df.astype('float')
    
    def group_neutralize_factor(self, y, x, n_groups=5):
        """First qcut_row x then group_standardize based on grouping results."""
        x = self.qcut_row(x, n_groups) 
        zscore_df = self.group_standardize_factor(y,x)
        return zscore_df

    def daily_linear_regression_residuals(self, y, x):
        """Computes residuals from a linear regression between two DataFrames using matrix operations."""
        assert y.shape == x.shape, "DataFrames should have the same dimensions."        
        y_values = y.values  # (n_days, n_observations)
        x_values = x.values  # (n_days, n_observations)
        X = np.stack([np.ones_like(x_values), x_values], axis=-1)  # (n_days, n_observations, 2)  # 添加截距
        valid_mask = ~np.isnan(X).any(axis=-1) & ~np.isnan(y_values)  # (n_days, n_observations)
        beta_matrix = np.full((X.shape[0], 2), np.nan)  # (n_days, 2)
        for i in range(X.shape[0]):
            X_valid = X[i, valid_mask[i]]  # (n_valid, 2)
            Y_valid = y_values[i, valid_mask[i]]  # (n_valid,)
            if len(Y_valid) > 1:
                beta_matrix[i] = np.linalg.inv(X_valid.T @ X_valid) @ X_valid.T @ Y_valid
        residuals = y_values - (X[:, :, 0] * beta_matrix[:, 0][:, None] + X[:, :, 1] * beta_matrix[:, 1][:, None])
        residuals[~valid_mask] = np.nan
        residuals_df = pd.DataFrame(residuals, index=y.index, columns=y.columns)
        return residuals_df.astype('float')

    def sm_linear_regression_residuals(self, y, X, categorical_columns, drop_first=True, dummy_na=False):
        """Computes residuals from linear regressions between provided DataFrames."""
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=drop_first, dummy_na=dummy_na)
        X = sm.add_constant(X)
        model = sm.OLS(y, X.astype('float')).fit()
        return model.resid
    
    def daily_linear_regression_residuals_more(self, df_y, continuous_Xs, continuous_X_names=[], categorical_Xs=[], categorical_X_names=[], drop_first=True):
        """Computes daily cross-sectional residuals from linear regressions between provided DataFrames. Using statsmodel and support category vairables."""
        residuals_df = pd.DataFrame(index=df_y.index, columns=df_y.columns)

        if len(continuous_X_names) != len(continuous_Xs):
            continuous_X_names = [f'f_{i}' for i in range(1, 1 + len(continuous_Xs))]
        if len(categorical_Xs) > 0 and len(categorical_X_names) != len(categorical_Xs):
            categorical_X_names = [f'cat_{i}' for i in range(1, 1 + len(categorical_Xs))]
        
        for index in df_y.index:
            Y = df_y.loc[index]
            X_sub = pd.concat([df_x.loc[index] for df_x in continuous_Xs], axis=1)
            X_sub.columns = continuous_X_names
            
            if len(categorical_Xs) > 0:
                X_catg = pd.concat([df_x.loc[index] for df_x in categorical_Xs], axis=1)
                X_catg.columns = categorical_X_names
                X_sub = pd.concat([X_sub, X_catg], axis=1)

            X_sub = sm.add_constant(X_sub)
            valid_indices = ~np.isnan(X_sub[continuous_X_names]).any(axis=1) & ~np.isnan(Y)  # 只检查连续变量和Y
            X_valid = X_sub.loc[valid_indices]
            Y_valid = Y[valid_indices]
            dummy_na = X_sub[categorical_X_names].isnull().any().any()
            if len(Y_valid) > 1:
                residuals = self.sm_linear_regression_residuals(Y_valid, X_valid, categorical_X_names, dummy_na=dummy_na, drop_first=drop_first)
                residuals_df.loc[index, valid_indices] = residuals
            else:
                residuals_df.loc[index] = np.nan
        return residuals_df.astype('float')